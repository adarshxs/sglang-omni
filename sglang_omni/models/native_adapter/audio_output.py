from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class PatchAudioAccumulator:
    """Accumulate latent patches and decode them into waveform audio."""

    feature_dim: int
    sample_rate: int
    patches: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.patches.clear()

    def append(self, patch: torch.Tensor | None) -> None:
        if patch is None:
            return
        self.patches.append(patch.detach())

    def is_empty(self) -> bool:
        return not self.patches

    def stacked(self) -> torch.Tensor | None:
        if not self.patches:
            return None
        rows = [patch.reshape(-1, self.feature_dim) for patch in self.patches]
        return torch.cat(rows, dim=0)

    def decode_full(
        self,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        all_patches = self.stacked()
        if all_patches is None:
            return None

        from einops import rearrange

        latents = rearrange(all_patches.to(dtype=dtype), "t d -> 1 d t")
        if device is not None:
            latents = latents.to(device=device)
        audio = decode_fn(latents)
        return audio.reshape(-1).detach().cpu().float()
