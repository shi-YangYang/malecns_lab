"""Stateless common Bernoulli input events keyed by seed, body ID and step."""

from __future__ import annotations

import numpy as np

EVENT_PROTOCOL = {
    "algorithm": "domain_separated_splitmix64_counter_v1",
    "domains_hex": {"body": "d1b54a32d192ed03", "step": "94d049bb133111eb", "seed": "bf58476d1ce4e5b9"},
    "uniform": "upper_53_bits_times_2^-53",
    "event": "uniform < rate_hz * dt_ms / 1000",
    "distribution": "independent-counter Bernoulli approximation, at most one event per neuron per step",
    "stream_identity": "run_seed, numeric body_id, segment-local simulation_step",
    "seed_scope": "reset to the same stream at each explicitly independent segment",
}


def _mix(value: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> np.uint64(31))


def bernoulli_events(body_ids: np.ndarray, rates_hz: np.ndarray, *, seed: int, start_step: int,
                     steps: int, dt_ms: float) -> np.ndarray:
    """Return steps × bodies; subset/order changes cannot consume another body's RNG."""
    ids = np.asarray(body_ids)
    rates = np.asarray(rates_hz, dtype=np.float64)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("body_ids must be a nonnegative integer vector")
    if len(np.unique(ids)) != len(ids) or rates.shape != ids.shape:
        raise ValueError("Unique body IDs and one rate per body are required")
    for name, value in (("seed", seed), ("start_step", start_step), ("steps", steps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**64:
            raise ValueError(f"{name} must be an unsigned 64-bit integer")
    if steps == 0 or start_step + steps >= 2**64 or not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("Positive bounded step count and finite positive dt required")
    probability = rates * (dt_ms / 1000)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Bernoulli step probabilities must lie in [0, 1]")
    body_keys = _mix(ids.astype(np.uint64) ^ np.uint64(0xD1B54A32D192ED03))
    step_keys = _mix(np.arange(start_step, start_step + steps, dtype=np.uint64) ^ np.uint64(0x94D049BB133111EB))
    seed_key = _mix(np.asarray(seed, dtype=np.uint64) ^ np.uint64(0xBF58476D1CE4E5B9))
    values = _mix(body_keys[None, :] ^ step_keys[:, None] ^ seed_key)
    uniform = (values >> np.uint64(11)).astype(np.float64) * (2.0**-53)
    return uniform < probability[None, :]
