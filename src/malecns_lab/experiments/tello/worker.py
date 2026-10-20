"""Spawned brain computation; no aircraft or SDK access."""

from __future__ import annotations

import multiprocessing as mp
import queue
import signal

from .shadow import ShadowBrain


def _brain_main(cfg, requests, results) -> None:
    # The control process owns Ctrl+C and the aircraft. This process only computes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    job_id = -1
    try:
        engine = ShadowBrain(cfg)
        while True:
            job_id, frame, state, t, reset = requests.get()
            if reset:
                engine.retina.reset()
                engine.retina.prev_exp = {"L": 0.0, "R": 0.0}
                engine.last_yaw = engine.last_t = None
                engine.yaw_rate = 0.0
            row = engine.step(frame, state, t)
            row["brain_source"] = str(cfg["brain"]["source"])
            row["brain_neurons"] = engine.brain.n_neurons
            results.put((job_id, row, None))
    except Exception as exc:
        results.put((job_id, None, f"{type(exc).__name__}: {exc}"))


class BrainProcess:
    """One outstanding image, tiny scalar results, no SDK access in the child."""

    def __init__(self, cfg):
        ctx = mp.get_context("spawn")
        self.requests = ctx.Queue(maxsize=1)
        self.results = ctx.Queue(maxsize=1)
        self.process = ctx.Process(target=_brain_main, args=(cfg, self.requests, self.results), daemon=True)
        self.process.start()
        self.sequence = 0

    def alive(self) -> bool:
        return self.process.is_alive()

    def submit(self, frame, state, t, reset=False) -> int:
        self.sequence += 1
        # put_nowait keeps a stalled worker from blocking the aircraft controller.
        self.requests.put_nowait((self.sequence, frame, dict(state), t, reset))
        return self.sequence

    def poll(self):
        try:
            return self.results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        for channel in (self.requests, self.results):
            channel.cancel_join_thread()
            channel.close()
