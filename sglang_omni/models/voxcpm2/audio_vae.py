from __future__ import annotations

from pathlib import Path

import librosa
import torch

from sglang_omni.utils.hf import load_raw_config_json

from .audio_vae_native import AudioVAEConfigV2, AudioVAEV2


def load_voxcpm2_audio_vae(
    model_path: str,
    *,
    device: str | torch.device = "cpu",
):
    """Load just the AudioVAE component from a VoxCPM2 checkpoint."""
    raw_config = load_raw_config_json(model_path)
    if raw_config is None:
        raise FileNotFoundError(f"Could not read VoxCPM2 raw config from {model_path}")

    cfg_dict = raw_config.get("audio_vae_config")
    vae_config = (
        AudioVAEConfigV2.model_validate(cfg_dict)
        if cfg_dict is not None
        else AudioVAEConfigV2()
    )
    audio_vae = AudioVAEV2(config=vae_config)

    model_dir = Path(model_path)
    safetensors_path = model_dir / "audiovae.safetensors"
    pth_path = model_dir / "audiovae.pth"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif pth_path.exists():
        checkpoint = torch.load(
            str(pth_path),
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        raise FileNotFoundError(
            f"Could not find AudioVAE weights under {model_dir}"
        )

    audio_vae.load_state_dict(state_dict, strict=True)
    return audio_vae.to(device).eval()


def _coerce_audio_source(source: object) -> tuple[torch.Tensor, int] | None:
    if source is None:
        return None
    if isinstance(source, dict) and "samples" in source and "sample_rate" in source:
        source = [source["samples"], int(source["sample_rate"])]
    if isinstance(source, list) and len(source) == 2 and isinstance(source[1], int):
        samples, sample_rate = source
        if isinstance(samples, torch.Tensor):
            waveform = samples.detach().cpu().float()
        else:
            waveform = torch.tensor(samples, dtype=torch.float32)
        return waveform, int(sample_rate)
    if isinstance(source, tuple) and len(source) == 2 and isinstance(source[1], int):
        samples, sample_rate = source
        if isinstance(samples, torch.Tensor):
            waveform = samples.detach().cpu().float()
        else:
            waveform = torch.tensor(samples, dtype=torch.float32)
        return waveform, int(sample_rate)
    if isinstance(source, str):
        waveform, sample_rate = librosa.load(source, sr=None, mono=True)
        return torch.from_numpy(waveform).float(), int(sample_rate)
    raise TypeError(f"Unsupported audio source type: {type(source)}")


def compute_audio_patch_count(
    source: object,
    *,
    encode_sample_rate: int,
    chunk_size: int,
    patch_size: int,
) -> int:
    parsed = _coerce_audio_source(source)
    if parsed is None:
        return 0
    waveform, sample_rate = parsed
    if sample_rate != encode_sample_rate:
        waveform = torch.from_numpy(
            librosa.resample(
                waveform.numpy(),
                orig_sr=sample_rate,
                target_sr=encode_sample_rate,
            )
        ).float()
    total_samples = int(waveform.numel())
    patch_len = int(chunk_size * patch_size)
    if total_samples % patch_len != 0:
        total_samples += patch_len - (total_samples % patch_len)
    return total_samples // patch_len
