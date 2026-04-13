from __future__ import annotations

from pathlib import Path

import torch

from sglang_omni.utils.hf import load_raw_config_json

from .import_utils import import_voxcpm_package


def load_voxcpm2_audio_vae(
    model_path: str,
    *,
    device: str | torch.device = "cpu",
):
    """Load just the AudioVAE component from a VoxCPM2 checkpoint."""
    import_voxcpm_package()
    from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2

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
