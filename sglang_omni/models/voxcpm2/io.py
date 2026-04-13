from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.native_adapter import PatchAudioAccumulator


@dataclass
class VoxCPM2GenerationConfig:
    max_new_tokens: int = 4096
    min_len: int = 2
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@dataclass
class VoxCPM2AdapterState:
    lm_hidden: torch.Tensor | None = None
    residual_hidden: torch.Tensor | None = None
    prefix_feat_cond: torch.Tensor | None = None
    current_embed_for_next: torch.Tensor | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finished: bool = False
    finish_reason: str | None = None
    accumulator: PatchAudioAccumulator | None = None


@dataclass
class VoxCPM2State:
    text: str = ""
    generation: VoxCPM2GenerationConfig = field(
        default_factory=VoxCPM2GenerationConfig
    )
    generated_patches: Any | None = None
    sample_rate: int = 48000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None

    @staticmethod
    def _tensor_to_list(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "min_len": self.generation.min_len,
                "cfg_value": self.generation.cfg_value,
                "inference_timesteps": self.generation.inference_timesteps,
            },
            "generated_patches": self._tensor_to_list(self.generated_patches),
            "sample_rate": self.sample_rate,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxCPM2State":
        generation = data.get("generation") or {}
        patches = data.get("generated_patches")
        if isinstance(patches, list):
            patches = torch.tensor(patches)
        return cls(
            text=data.get("text", ""),
            generation=VoxCPM2GenerationConfig(
                max_new_tokens=generation.get("max_new_tokens", 4096),
                min_len=generation.get("min_len", 2),
                cfg_value=generation.get("cfg_value", 2.0),
                inference_timesteps=generation.get("inference_timesteps", 10),
            ),
            generated_patches=patches,
            sample_rate=data.get("sample_rate", 48000),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            finish_reason=data.get("finish_reason"),
        )
