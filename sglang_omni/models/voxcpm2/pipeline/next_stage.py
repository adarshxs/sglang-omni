from __future__ import annotations

from typing import Any

PREPROCESSING_STAGE = "preprocessing"
GENERATION_STAGE = "generation"
AUDIO_DECODE_STAGE = "audio_decode"


def preprocessing_next(request_id: str, output: Any) -> list[str]:
    return [GENERATION_STAGE]


def generation_next(request_id: str, output: Any) -> list[str]:
    return [AUDIO_DECODE_STAGE]


def audio_decode_next(request_id: str, output: Any) -> None:
    return None
