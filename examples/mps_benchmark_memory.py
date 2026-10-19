"""Worker-local MPS sampled memory evidence; independent of parent RSS supervision."""
from __future__ import annotations

import csv
import math
import threading
import time

import torch


class MPSMemorySampler:
    def __init__(self, path, effective_rss_budget):
        recommended = torch.mps.recommended_max_memory()
        if recommended <= 0 or effective_rss_budget <= 0:
            raise ValueError("Invalid MPS allocator budget")
        # Round down so floating division/multiplication cannot put the cap above
        # the registered byte budget (even by a fraction of one byte).
        fraction = min(1.0, math.nextafter(effective_rss_budget / recommended, 0.0))
        torch.mps.set_per_process_memory_fraction(fraction)
        self.budget = {"recommended_max_memory_bytes": recommended, "fraction": fraction,
                       "allocator_cap_bytes": fraction * recommended, "effective_rss_budget_bytes": effective_rss_budget}
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.samples = 0
        self.peaks = {"current_allocated_bytes": 0, "driver_allocated_bytes": 0}
        self.stream = path.open("x")
        self.writer = csv.DictWriter(self.stream, fieldnames=["wall_seconds", "phase", *self.peaks])
        self.writer.writeheader()
        self.thread = threading.Thread(target=self._loop, name="mps-memory-sampler", daemon=True)
        self.sample("allocator_configured")
        self.thread.start()

    def sample(self, phase):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"MPS memory sampling failed: {self.error}")
            try:
                values = {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                          "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
                self.writer.writerow({"wall_seconds": time.perf_counter() - self.started, "phase": phase, **values})
                self.stream.flush()
                self.samples += 1
                self.peaks = {key: max(self.peaks[key], value) for key, value in values.items()}
            except BaseException as error:
                self.error = f"{type(error).__name__}:{error}"
                raise

    def _loop(self):
        while not self.stop.wait(.1):
            try:
                self.sample("periodic")
            except BaseException:
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.error = "memory sampler did not terminate"
        try:
            if self.error is None:
                self.sample("finished")
        finally:
            self.stream.close()
        return {"budget": self.budget, "samples": self.samples, "sampling_seconds": .1,
                "sampled_peak_bytes": self.peaks, "error": self.error,
                "interpretation": "Sampled MPS peaks only. Unified-memory RSS and driver values are not additive."}
