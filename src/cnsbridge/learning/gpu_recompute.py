"""Pure all-device checkpoint blocks with explicit recurrent credit cuts."""
import torch
from torch.utils.checkpoint import checkpoint
from ..brain.trainable import LIFState
from ..senses.gpu_drive import GPUDriveState

BLOCK_STEPS = 50


def block_function(policy, mask, brain_step, drive_step, frame_step, count):
    brain, drive = policy.brain, policy.drive
    delay, half = brain.delay_steps, policy.frame_steps // 2

    def block(v, g, refractory, phase, counts, *pending_and_sums):
        state = LIFState(v, g, refractory, tuple(pending_and_sums[:delay]), brain_step, brain.model_id)
        sensory = GPUDriveState(phase, counts, drive_step, drive.drive_id, drive.mapping.map_id)
        sums = list(pending_and_sums[delay:])
        for offset in range(count):
            increment, sensory = drive.step(mask, sensory)
            _, state = brain.step(increment, state)
            index = (frame_step + offset) // half
            sums[index] = sums[index] + state.v[policy.dn_indices]
        return (state.v, state.g, state.refractory_until, sensory.phase, sensory.event_counts, *state.pending, *sums)
    return block


def checkpoint_frame(policy, mask, brain, sensory, sums, *, frame_offset, direct=False):
    if direct and brain.model_id != policy.brain.model_id:
        raise ValueError("Foreign numerical state")
    for start in range(0, policy.frame_steps, BLOCK_STEPS):
        count = min(BLOCK_STEPS, policy.frame_steps - start)
        block = block_function(policy, mask, brain.step, sensory.step, start, count)
        inputs = (brain.v, brain.g, brain.refractory_until, sensory.phase, sensory.event_counts, *brain.pending, *sums)
        values = (block(*inputs) if direct else checkpoint(block, *inputs, use_reentrant=False,
                  preserve_rng_state=False, determinism_check="default", early_stop=True))
        brain = LIFState(values[0], values[1], values[2], tuple(values[5:5 + policy.brain.delay_steps]),
                         brain.step + count, brain.model_id)
        sensory = GPUDriveState(values[3], values[4], sensory.step + count, sensory.drive_id, sensory.spatial_map_id)
        sums = list(values[-2:])
        completed = frame_offset + start + count
        if completed % policy.credit_horizon == 0 and completed < 2 * policy.frame_steps:
            brain = brain.detach()
    return brain, sensory, sums
