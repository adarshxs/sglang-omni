# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni: Multi-stage pipeline framework for omni models."""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_EXPORTS = {
    "Coordinator": ("sglang_omni.pipeline.coordinator", "Coordinator"),
    "Stage": ("sglang_omni.pipeline.stage", "Stage"),
    "Worker": ("sglang_omni.pipeline.worker", "Worker"),
    "Engine": ("sglang_omni.engines.base", "Engine"),
    "Client": ("sglang_omni.client", "Client"),
    "InputHandler": ("sglang_omni.pipeline.stage", "InputHandler"),
    "DirectInput": ("sglang_omni.pipeline.stage", "DirectInput"),
    "AggregatedInput": ("sglang_omni.pipeline.stage", "AggregatedInput"),
    "RequestState": ("sglang_omni.proto", "RequestState"),
    "OmniRequest": ("sglang_omni.proto", "OmniRequest"),
    "StageInfo": ("sglang_omni.proto", "StageInfo"),
    "DataReadyMessage": ("sglang_omni.proto", "DataReadyMessage"),
    "AbortMessage": ("sglang_omni.proto", "AbortMessage"),
    "CompleteMessage": ("sglang_omni.proto", "CompleteMessage"),
    "GenerateRequest": ("sglang_omni.client", "GenerateRequest"),
    "GenerateChunk": ("sglang_omni.client", "GenerateChunk"),
    "SamplingParams": ("sglang_omni.client", "SamplingParams"),
    "Message": ("sglang_omni.client", "Message"),
    "UsageInfo": ("sglang_omni.client", "UsageInfo"),
    "AbortLevel": ("sglang_omni.client", "AbortLevel"),
    "AbortResult": ("sglang_omni.client", "AbortResult"),
}

__all__ = [
    # Core classes
    "Coordinator",
    "Stage",
    "Worker",
    "Engine",
    "Client",
    # Input handlers
    "InputHandler",
    "DirectInput",
    "AggregatedInput",
    # Types
    "RequestState",
    "OmniRequest",
    "StageInfo",
    "DataReadyMessage",
    "AbortMessage",
    "CompleteMessage",
    "GenerateRequest",
    "GenerateChunk",
    "SamplingParams",
    "Message",
    "UsageInfo",
    "AbortLevel",
    "AbortResult",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
