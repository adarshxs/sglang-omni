from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import LlamaTokenizerFast, PretrainedConfig

from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.utils.hf import load_raw_config_json

_SCAFFOLD_CONFIG_BASENAME = "sglang_voxcpm2_scaffold_config.json"


class VoxCPM2HFConfig(PretrainedConfig):
    model_type = "voxcpm2_scaffold"

    def __init__(
        self,
        *,
        vocab_size: int = 73448,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        scale_emb: float = 12.0,
        dim_model_base: int = 256,
        scale_depth: float = 1.4,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        architectures: list[str] | None = None,
        native_model_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.hidden_act = hidden_act
        self.tie_word_embeddings = tie_word_embeddings
        self.architectures = (
            architectures or ["VoxCPM2MiniCPMScaffoldForCausalLM"]
        )
        self.native_model_path = native_model_path

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
        lm_config = raw_config["lm_config"]
        return cls(
            vocab_size=lm_config["vocab_size"],
            hidden_size=lm_config["hidden_size"],
            intermediate_size=lm_config["intermediate_size"],
            num_attention_heads=lm_config["num_attention_heads"],
            num_key_value_heads=lm_config["num_key_value_heads"],
            num_hidden_layers=lm_config["num_hidden_layers"],
            max_position_embeddings=lm_config["max_position_embeddings"],
            rms_norm_eps=lm_config["rms_norm_eps"],
            rope_theta=lm_config["rope_theta"],
            rope_scaling=lm_config.get("rope_scaling"),
            scale_emb=lm_config["scale_emb"],
            dim_model_base=lm_config["dim_model_base"],
            scale_depth=lm_config["scale_depth"],
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
    """Write a HF-compatible scaffold config for a raw VoxCPM2 checkpoint."""
    resolved_dir = resolve_model_path(model_path, local_files_only=local_files_only)
    raw_config = load_raw_config_json(str(resolved_dir))
    if raw_config is None:
        raise FileNotFoundError(f"Could not read raw VoxCPM2 config from {resolved_dir}")

    pad_token_id, bos_token_id, eos_token_id = _resolve_tokenizer_special_ids(
        resolved_dir
    )
    scaffold_config = VoxCPM2HFConfig.from_voxcpm2_raw_config(
        raw_config,
        native_model_path=str(resolved_dir),
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    target = resolved_dir / _SCAFFOLD_CONFIG_BASENAME
    target.write_text(
        json.dumps(scaffold_config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(resolved_dir), _SCAFFOLD_CONFIG_BASENAME
