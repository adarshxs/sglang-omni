from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import LlamaTokenizerFast, PretrainedConfig

from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.utils.hf import load_raw_config_json

_NATIVE_CONFIG_BASENAME = "sglang_voxcpm2_native_config.json"


def _default_encoder_config() -> dict[str, Any]:
    return {
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "num_layers": 4,
        "kv_channels": None,
    }


def _default_dit_config() -> dict[str, Any]:
    return {
        "hidden_dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "num_layers": 4,
        "kv_channels": None,
        "dit_mean_mode": False,
        "cfm_config": {
            "sigma_min": 1e-6,
            "solver": "euler",
            "t_scheduler": "log-norm",
            "training_cfg_rate": 0.1,
            "inference_cfg_rate": 1.0,
            "reg_loss_type": "l1",
            "ratio_r_neq_t_range": (0.25, 0.75),
            "noise_cond_prob_range": (0.0, 0.0),
            "noise_cond_scale": 0.0,
        },
    }


class VoxCPM2HFConfig(PretrainedConfig):
    model_type = "voxcpm2_native"

    def __init__(
        self,
        *,
        lm_config: dict[str, Any] | None = None,
        patch_size: int = 4,
        feat_dim: int = 64,
        residual_lm_num_layers: int = 8,
        residual_lm_no_rope: bool = False,
        scalar_quantization_latent_dim: int = 512,
        scalar_quantization_scale: int = 9,
        encoder_config: dict[str, Any] | None = None,
        dit_config: dict[str, Any] | None = None,
        audio_vae_config: dict[str, Any] | None = None,
        max_length: int = 8192,
        dtype: str = "bfloat16",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        architectures: list[str] | None = None,
        native_model_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        lm_config = dict(lm_config or {})
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=bool(lm_config.get("tie_word_embeddings", True)),
            **kwargs,
        )
        self.lm_config = lm_config
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.residual_lm_num_layers = residual_lm_num_layers
        self.residual_lm_no_rope = residual_lm_no_rope
        self.scalar_quantization_latent_dim = scalar_quantization_latent_dim
        self.scalar_quantization_scale = scalar_quantization_scale
        self.encoder_config = encoder_config or _default_encoder_config()
        self.dit_config = dit_config or _default_dit_config()
        self.audio_vae_config = audio_vae_config
        self.max_length = max_length
        self.dtype = dtype
        self.architectures = architectures or ["VoxCPM2ForCausalLM"]
        self.native_model_path = native_model_path

        # Mirror the LM surface at the top level because SGLang's model config
        # helpers read these fields directly for hidden size and vocab setup.
        self.vocab_size = int(lm_config.get("vocab_size", kwargs.get("vocab_size", 0)))
        self.hidden_size = int(
            lm_config.get("hidden_size", kwargs.get("hidden_size", 0))
        )
        self.intermediate_size = int(
            lm_config.get("intermediate_size", kwargs.get("intermediate_size", 0))
        )
        self.num_attention_heads = int(
            lm_config.get("num_attention_heads", kwargs.get("num_attention_heads", 0))
        )
        self.num_key_value_heads = int(
            lm_config.get("num_key_value_heads", kwargs.get("num_key_value_heads", 0))
        )
        self.num_hidden_layers = int(
            lm_config.get("num_hidden_layers", kwargs.get("num_hidden_layers", 0))
        )
        self.max_position_embeddings = int(
            lm_config.get(
                "max_position_embeddings", kwargs.get("max_position_embeddings", 0)
            )
        )
        self.rms_norm_eps = float(
            lm_config.get("rms_norm_eps", kwargs.get("rms_norm_eps", 1e-5))
        )
        self.rope_theta = float(
            lm_config.get("rope_theta", kwargs.get("rope_theta", 10000.0))
        )
        self.rope_scaling = lm_config.get("rope_scaling")
        self.scale_emb = float(lm_config.get("scale_emb", kwargs.get("scale_emb", 1.0)))
        self.dim_model_base = int(
            lm_config.get("dim_model_base", kwargs.get("dim_model_base", 0))
        )
        self.scale_depth = float(
            lm_config.get("scale_depth", kwargs.get("scale_depth", 1.0))
        )
        self.kv_channels = lm_config.get("kv_channels")

    @classmethod
    def from_voxcpm2_raw_config(
        cls,
        raw_config: dict[str, Any],
        *,
        native_model_path: str,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> "VoxCPM2HFConfig":
        return cls(
            lm_config=raw_config["lm_config"],
            patch_size=int(raw_config.get("patch_size", 4)),
            feat_dim=int(raw_config.get("feat_dim", 64)),
            residual_lm_num_layers=int(raw_config.get("residual_lm_num_layers", 8)),
            residual_lm_no_rope=bool(raw_config.get("residual_lm_no_rope", False)),
            scalar_quantization_latent_dim=int(
                raw_config.get("scalar_quantization_latent_dim", 512)
            ),
            scalar_quantization_scale=int(
                raw_config.get("scalar_quantization_scale", 9)
            ),
            encoder_config=raw_config.get("encoder_config")
            or _default_encoder_config(),
            dit_config=raw_config.get("dit_config") or _default_dit_config(),
            audio_vae_config=raw_config.get("audio_vae_config"),
            max_length=int(raw_config.get("max_length", 8192)),
            dtype=str(raw_config.get("dtype", "bfloat16")),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            native_model_path=native_model_path,
        )


def _resolve_tokenizer_special_ids(model_dir: Path) -> tuple[int, int, int]:
    try:
        tokenizer = LlamaTokenizerFast.from_pretrained(model_dir)
    except Exception:
        return (0, 1, 2)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id

    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    if bos_token_id is None:
        bos_token_id = 1
    if eos_token_id is None:
        eos_token_id = 2
    return (int(pad_token_id), int(bos_token_id), int(eos_token_id))


def ensure_voxcpm2_scaffold_config(
    model_path: str,
    *,
    local_files_only: bool = False,
) -> tuple[str, str]:
    """Write a HF-compatible native config for a raw VoxCPM2 checkpoint."""
    resolved_dir = resolve_model_path(model_path, local_files_only=local_files_only)
    raw_config = load_raw_config_json(str(resolved_dir))
    if raw_config is None:
        raise FileNotFoundError(f"Could not read raw VoxCPM2 config from {resolved_dir}")

    pad_token_id, bos_token_id, eos_token_id = _resolve_tokenizer_special_ids(
        resolved_dir
    )
    native_config = VoxCPM2HFConfig.from_voxcpm2_raw_config(
        raw_config,
        native_model_path=str(resolved_dir),
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    target = resolved_dir / _NATIVE_CONFIG_BASENAME
    target.write_text(
        json.dumps(native_config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(resolved_dir), _NATIVE_CONFIG_BASENAME
