from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.voxcpm2.pipeline.next_stage import (
    AUDIO_DECODE_STAGE,
    GENERATION_STAGE,
    PREPROCESSING_STAGE,
)

_PKG = "sglang_omni.models.voxcpm2.pipeline"


class VoxCPM2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "VoxCPM2ForConditionalGeneration"

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_preprocessing_executor",
            ),
            get_next=f"{_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=GENERATION_STAGE,
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_generation_executor",
                args={
                    "device": "cuda:0",
                    "mem_fraction_static": 0.85,
                },
            ),
            get_next=f"{_PKG}.next_stage.generation_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name=AUDIO_DECODE_STAGE,
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_audio_decode_executor",
                args={"device": "cpu"},
            ),
            get_next=f"{_PKG}.next_stage.audio_decode_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


EntryClass = VoxCPM2PipelineConfig
