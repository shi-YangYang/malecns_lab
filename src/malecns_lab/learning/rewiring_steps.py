"""One shared, resumable edge-pair proposal kernel; no graph loading or IO."""
from __future__ import annotations

import copy
import numpy as np

REASONS = ("same_source", "self_edge", "different_source_sign", "same_target", "new_self_edge", "duplicate")
ALGORITHM = "uniform-edge-pair-sign-preserving-sorted-column-v1"


def counters():
    return {"attempts": 0, "successful_swaps": 0, "rejections": dict.fromkeys(REASONS, 0)}


def validate_counters(value, max_attempts, target_swaps):
    if (set(value) != {"attempts", "successful_swaps", "rejections"}
            or set(value["rejections"]) != set(REASONS)):
        raise ValueError("Invalid proposal counter fields")
    numbers = [value["attempts"], value["successful_swaps"], *value["rejections"].values(), max_attempts, target_swaps]
    if any(type(item) is not int or item < 0 for item in numbers):
        raise ValueError("Proposal counters/budgets must be nonboolean nonnegative integers")
    if (value["attempts"] != value["successful_swaps"] + sum(value["rejections"].values())
            or value["attempts"] > max_attempts or value["successful_swaps"] > target_swaps):
        raise ValueError("Proposal counter conservation/budget mismatch")


def restore_rng(state):
    """Validate every PCG64 field, including the cached uint32 half-draw."""
    if (not isinstance(state, dict) or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"}
            or state["bit_generator"] != "PCG64" or not isinstance(state["state"], dict)
            or set(state["state"]) != {"state", "inc"}):
        raise ValueError("Complete PCG64 state required")
    for key, limit in (("has_uint32", 2), ("uinteger", 2**32)):
        if type(state[key]) is not int or not 0 <= state[key] < limit:
            raise ValueError("Invalid PCG64 cached state")
    for key in ("state", "inc"):
        if type(state["state"][key]) is not int or not 0 <= state["state"][key] < 2**128:
            raise ValueError("Invalid PCG64 core state")
    if state["state"]["inc"] % 2 != 1:
        raise ValueError("PCG64 increment must be odd")
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = copy.deepcopy(state)
    if rng.bit_generator.state != state:
        raise ValueError("PCG64 restore changed cached state")
    return rng


def replace_target(matrix, source, position, target):
    start, end = matrix.indptr[source:source + 2]
    insertion = start + int(np.searchsorted(matrix.indices[start:end], target))
    weight = matrix.data[position]
    if insertion > position:
        insertion -= 1
        matrix.indices[position:insertion] = matrix.indices[position + 1:insertion + 1]
        matrix.data[position:insertion] = matrix.data[position + 1:insertion + 1]
    else:
        matrix.indices[insertion + 1:position + 1] = matrix.indices[insertion:position]
        matrix.data[insertion + 1:position + 1] = matrix.data[insertion:position]
    matrix.indices[insertion], matrix.data[insertion] = target, weight


def contains(matrix, source, target):
    start, end = matrix.indptr[source:source + 2]
    position = start + np.searchsorted(matrix.indices[start:end], target)
    return bool(position < end and matrix.indices[position] == target)


def advance(matrix, rng, state, *, max_attempts, target_swaps, proposals):
    """Mutate CSC and counters; exact original draw shape and rejection order."""
    validate_counters(state, max_attempts, target_swaps)
    if type(proposals) is not int or proposals < 0 or not isinstance(rng.bit_generator, np.random.PCG64):
        raise ValueError("Invalid segment proposal budget or RNG")
    end = min(max_attempts, state["attempts"] + proposals)
    while matrix.nnz and state["attempts"] < end and state["successful_swaps"] < target_swaps:
        state["attempts"] += 1
        a, b = rng.integers(matrix.nnz, size=2)
        u, w = np.searchsorted(matrix.indptr, [a, b], side="right") - 1
        v, x = matrix.indices[[a, b]]
        reason = None
        if u == w:
            reason = "same_source"
        elif u == v or w == x:
            reason = "self_edge"
        elif np.sign(matrix.data[a]) != np.sign(matrix.data[b]):
            reason = "different_source_sign"
        elif v == x:
            reason = "same_target"
        elif u == x or w == v:
            reason = "new_self_edge"
        elif contains(matrix, u, x) or contains(matrix, w, v):
            reason = "duplicate"
        if reason:
            state["rejections"][reason] += 1
            continue
        replace_target(matrix, u, a, x)
        replace_target(matrix, w, b, v)
        state["successful_swaps"] += 1
    validate_counters(state, max_attempts, target_swaps)
    return state


def report(matrix, state, *, max_attempts, target_swaps):
    nonself = matrix.nnz - np.count_nonzero(matrix.diagonal())
    return {**copy.deepcopy(state), "target_reached": state["successful_swaps"] == target_swaps,
            "stop_reason": "target_reached" if state["successful_swaps"] == target_swaps else
                "no_edges" if not matrix.nnz else "proposal_budget_exhausted" if state["attempts"] == max_attempts else "segment_quota",
            "effective_edge_moves_per_nonself_edge": 2 * state["successful_swaps"] / nonself if nonself else None,
            "proposal": "independent uniform edge positions, with replacement; all listed invalid proposals rejected",
            "budget_interpretation": "engineering budget, not experimentally validated mixing"}
