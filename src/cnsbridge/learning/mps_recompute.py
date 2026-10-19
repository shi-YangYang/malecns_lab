"""Pure, fixed-length activation recomputation for the MPS neural frame.

Only block boundaries remain saved. Each invocation (including backward
recomputation) reconstructs its own sensory clock from an immutable snapshot.
"""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ..brain.mps_identity import ACTIVATION_CHECKPOINT
from ..brain.trainable import LIFState
from ..senses.spatial_drive import SpatialDrive


def _block_function(policy, mask, drive_state, brain_step, frame_step, count):
    # Factory arguments bind each block independently; no loop variable or
    # mutable external drive/state is captured by the backward invocation.
    brain = policy.brain
    mapping, config, indices = policy.spatial_map, policy.drive_config, policy.dn_indices
    half_steps = policy.frame_steps // 2
    delay_steps, model_id = brain.delay_steps, brain.model_id

    def block(v, g, refractory, *pending_and_sums):
        local_drive = SpatialDrive(mapping, config)
        local_drive.restore_state(drive_state)
        state = LIFState(v, g, refractory, tuple(pending_and_sums[:delay_steps]), brain_step, model_id)
        sums = list(pending_and_sums[delay_steps:])
        for offset in range(count):
            increment = local_drive.step(mask).to_torch(dtype=torch.float32).to("mps")
            _, state = brain.step(increment, state)
            half = (frame_step + offset) // half_steps
            # Carry the original running sums through boundaries: block sums
            # followed by a reduction would change float32 addition order.
            sums[half] = sums[half] + state.v[indices]
        return (state.v, state.g, state.refractory_until, *state.pending, *sums, local_drive.export_state())

    return block


def checkpoint_frame(policy, mask, brain, drive_state, sums):
    mask = np.array(mask, copy=True)
    mask.setflags(write=False)
    block_steps = ACTIVATION_CHECKPOINT["neural_steps_per_block"]
    for start in range(0, policy.frame_steps, block_steps):
        count = min(block_steps, policy.frame_steps - start)
        block = _block_function(policy, mask, drive_state, brain.step, start, count)
        values = checkpoint(block, brain.v, brain.g, brain.refractory_until, *brain.pending, *sums,
                            use_reentrant=False, preserve_rng_state=False, determinism_check="default", early_stop=True)
        brain = LIFState(values[0], values[1], values[2], tuple(values[3:3 + policy.brain.delay_steps]),
                         brain.step + count, brain.model_id)
        sums = list(values[-3:-1])
        drive_state = values[-1]
    return brain, drive_state, sums
