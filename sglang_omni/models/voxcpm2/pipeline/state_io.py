from __future__ import annotations

from sglang_omni.models.voxcpm2.io import VoxCPM2State
from sglang_omni.proto import StagePayload


def load_state(payload: StagePayload) -> VoxCPM2State:
    return VoxCPM2State.from_dict(payload.data or {})


def store_state(payload: StagePayload, state: VoxCPM2State) -> StagePayload:
    payload.data = state.to_dict()
    return payload
