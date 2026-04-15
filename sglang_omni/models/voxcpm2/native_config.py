from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class RopeScalingConfig(BaseModel):
    type: str
    long_factor: list[float]
    short_factor: list[float]
    original_max_position_embeddings: int


class MiniCPM4Config(BaseModel):
    model_config = ConfigDict(extra="allow")

    bos_token_id: int
    eos_token_id: int
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: RopeScalingConfig
    vocab_size: int
    use_mup: bool = True
    scale_emb: float
    dim_model_base: int
    scale_depth: float
    rope_theta: float
    kv_channels: int | None = None
    no_rope: bool = False


class VoxCPMEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int | None = None


class CfmConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0


class VoxCPMDitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int | None = None
    dit_mean_mode: bool = False
    cfm_config: CfmConfig


class VoxCPM2NativeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    lm_config: MiniCPM4Config
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9
    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: dict[str, Any] | None = None
    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"
