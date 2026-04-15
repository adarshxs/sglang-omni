from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

@dataclass
class VoxCPM2GenerationConfig:
    max_new_tokens: int = 4096
    min_len: int = 2
    cfg_value: float = 2.0
    inference_timesteps: int = 10


@dataclass
class VoxCPM2State:
    text: str = ""
    prompt_text: str | None = None
    prompt_audio: Any | None = None
    reference_audio: Any | None = None
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

    @staticmethod
    def _audio_to_dict(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, tuple) and len(value) == 2:
            samples, sample_rate = value
            return {
                "samples": VoxCPM2State._tensor_to_list(samples),
                "sample_rate": int(sample_rate),
            }
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
            return {
                "samples": VoxCPM2State._tensor_to_list(value[0]),
                "sample_rate": int(value[1]),
            }
        return value

    @staticmethod
    def _audio_from_dict(value: Any) -> Any:
        if isinstance(value, dict) and "samples" in value and "sample_rate" in value:
            return [value["samples"], int(value["sample_rate"])]
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt_text": self.prompt_text,
            "prompt_audio": self._audio_to_dict(self.prompt_audio),
            "reference_audio": self._audio_to_dict(self.reference_audio),
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
            prompt_text=data.get("prompt_text"),
            prompt_audio=cls._audio_from_dict(data.get("prompt_audio")),
            reference_audio=cls._audio_from_dict(data.get("reference_audio")),
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
