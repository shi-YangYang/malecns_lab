"""Pure MPS yaw-policy blocks; short recurrent credit never resets state."""
import torch
from torch.utils.checkpoint import checkpoint

from ...brain.trainable import LIFState
from ...senses.yaw_drive import YawDriveState

BLOCK_STEPS = 50
CREDIT_STEPS = 100


def _block(policy, rates, brain_step, drive_step, frame_start, count):
    delay, half = policy.brain.delay_steps, policy.frame_steps // 2

    def execute(v, g, refractory, phase, events, *pending_sums_counts):
        brain = LIFState(v, g, refractory, tuple(pending_sums_counts[:delay]), brain_step, policy.brain.model_id)
        drive = YawDriveState(phase, events, drive_step)
        sums = list(pending_sums_counts[delay:delay + 2])
        counts = pending_sums_counts[-1]
        for offset in range(count):
            increment, drive = policy.drive.step(rates, drive)
            spikes, brain = policy.brain.step(increment, brain)
            index = (frame_start + offset) // half
            sums[index] = sums[index] + brain.v[policy.dn_indices]
            counts = counts + spikes.detach().to(torch.int64)
        return brain.v, brain.g, brain.refractory_until, drive.phase, drive.event_counts, *brain.pending, *sums, counts

    return execute


def advance_frame(policy, rates, brain, drive, *, frame_offset=0, direct=False):
    if direct and policy.brain.graph.identity["role"] != "numerical_test_only":
        raise ValueError("Direct execution is numerical-fixture only")
    sums = [brain.v.new_zeros(len(policy.dn_indices)) for _ in range(2)]
    counts = torch.zeros(policy.brain.graph.n, device="mps", dtype=torch.int64)
    for start in range(0, policy.frame_steps, BLOCK_STEPS):
        count = min(BLOCK_STEPS, policy.frame_steps - start)
        execute = _block(policy, rates, brain.step, drive.step, start, count)
        inputs = (brain.v, brain.g, brain.refractory_until, drive.phase, drive.event_counts, *brain.pending, *sums, counts)
        values = (execute(*inputs) if direct or not torch.is_grad_enabled() else
                  checkpoint(execute, *inputs, use_reentrant=False, preserve_rng_state=False,
                             determinism_check="default", early_stop=True))
        delay = policy.brain.delay_steps
        brain = LIFState(values[0], values[1], values[2], tuple(values[5:5 + delay]),
                         brain.step + count, policy.brain.model_id)
        drive = YawDriveState(values[3], values[4], drive.step + count)
        sums, counts = list(values[-3:-1]), values[-1]
        if (frame_offset + start + count) % CREDIT_STEPS == 0:
            brain = brain.detach()
    return brain, drive, torch.cat(sums) / (policy.frame_steps // 2), counts
