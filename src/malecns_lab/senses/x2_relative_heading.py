"""Relative-task input variant: retain odometry evidence, suppress absolute yaw ports."""
from copy import deepcopy

from ..control.x2_behavior import digest, require


class RelativeHeadingEncoder:
    """Same R1–R6 mapping and every other channel; no behavior readout or task rule."""
    def __init__(self, original):
        self.original = original
        identity = original.identity
        self.input_body_ids, self.input_indices = original.input_body_ids, original.input_indices
        self.channel_indices = original.channel_indices
        excluded = {identity["channels"].index(name) for name in ("body.yaw.positive", "body.yaw.negative")}
        self.suppressed_indices = tuple(i for i, channel in enumerate(self.channel_indices) if channel in excluded)
        require(len(self.suppressed_indices) == 118, "Expected both 59-neuron absolute-yaw ports")
        self.parent_input_map_id = identity.pop("input_map_id")
        identity.update(schema="malecns_lab.x2_behavior_input.no_absolute_yaw.v1",
            parent_input_map_id=self.parent_input_map_id,
            intervention={"channels": ["body.yaw.positive", "body.yaw.negative"], "encoded_value": 0.,
                "raw_observation_changed": False, "relative_heading_error_retained": True,
                "reason": "relative tasks must not depend on world-frame absolute yaw"})
        self.input_map_id = digest(identity)
        self._identity = {**identity, "input_map_id": self.input_map_id}

    @property
    def identity(self):
        return deepcopy(self._identity)

    def encode(self, observation):
        values = self.original.encode(observation)
        values[list(self.suppressed_indices)] = 0.
        return values
