from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .native_config import MiniCPM4Config


class StaticKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        dim_kv_head: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int = 8192,
    ):
        self.max_length = max_length
        self.num_layers = num_layers
        self.kv_cache = torch.zeros(
            2,
            num_layers,
            batch_size,
            num_kv_heads,
            max_length,
            dim_kv_head,
            device=device,
            dtype=dtype,
        )
        self.current_length = 0

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")
        value = self.current_length
        self.current_length += 1
        return value

    def fill_caches(self, kv_caches: list[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.current_length = kv_caches[0][0].size(2)
        self.kv_cache.zero_()
        for layer_idx in range(self.num_layers):
            self.kv_cache[0, layer_idx, :, :, : self.current_length, :] = kv_caches[
                layer_idx
            ][0]
            self.kv_cache[1, layer_idx, :, :, : self.current_length, :] = kv_caches[
                layer_idx
            ][1]


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(orig_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class MiniCPMLongRoPE(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.config = config
        self.dim = (
            config.kv_channels
            if config.kv_channels
            else config.hidden_size // config.num_attention_heads
        )
        self.base = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.short_factor = config.rope_scaling.short_factor
        self.long_factor = config.rope_scaling.long_factor
        self.original_max_position_embeddings = (
            config.rope_scaling.original_max_position_embeddings
        )
        scale = (
            self.max_position_embeddings / self.original_max_position_embeddings
        )
        self.scaling_factor = math.sqrt(
            1 + math.log(scale) / math.log(self.original_max_position_embeddings)
        )
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(
            self.max_position_embeddings,
            self.inv_freq.device,
            torch.float32,
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(
                self.long_factor, dtype=torch.float32, device=device
            )
        else:
            ext_factors = torch.tensor(
                self.short_factor, dtype=torch.float32, device=device
            )
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype) * self.scaling_factor
        self.sin_cached = emb.sin().to(dtype) * self.scaling_factor

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        super().__init__()
        del layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        is_causal: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, (k, v)

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        position_id: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_emb is not None:
            cos, sin = position_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id, :] = k
        value_cache[:, :, position_id, :] = v
        attn_mask = (
            torch.arange(key_cache.size(2), device=key_cache.device) <= position_id
        ).view(1, 1, 1, -1)
        q = q.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class MiniCPMMLP(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        super().__init__()
        self.self_attn = MiniCPMAttention(config=config, layer_idx=layer_idx)
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def _residual_scale(self) -> float:
        if self.use_mup:
            return self.scale_depth / math.sqrt(self.num_hidden_layers)
        return 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        is_causal: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_emb=position_emb,
            is_causal=is_causal,
        )
        hidden_states = residual + hidden_states * self._residual_scale()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self._residual_scale()
        return hidden_states, present_key_value

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor] | None,
        position_id: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_step(
            hidden_states=hidden_states,
            position_emb=position_emb,
            position_id=position_id,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states * self._residual_scale()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self._residual_scale()
        return hidden_states


class MiniCPMModel(nn.Module):
    def __init__(self, config: MiniCPM4Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.config = config
        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList(
            [
                MiniCPMDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope_emb = None if config.no_rope else MiniCPMLongRoPE(config)
        self.kv_cache: StaticKVCache | None = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self.rope_emb is not None:
            position_ids = torch.arange(
                0,
                inputs_embeds.size(1),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_emb = self.rope_emb(position_ids)
        else:
            position_emb = None
        hidden_states = inputs_embeds
        next_decoder_cache = []
        for decoder_layer in self.layers:
            hidden_states, this_cache = decoder_layer(
                hidden_states,
                position_emb,
                is_causal,
            )
            next_decoder_cache.append(this_cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache

    def forward_step(
        self,
        inputs_embeds: torch.Tensor,
        position_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("KV cache is not setup")
        position_emb = self.rope_emb(position_id) if self.rope_emb is not None else None
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer.forward_step(
                hidden_states,
                position_emb,
                position_id,
                self.kv_cache.get_layer_cache(layer_idx),
            )
        return self.norm(hidden_states)

    def setup_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.kv_cache = StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            dim_kv_head=(
                self.config.hidden_size // self.config.num_attention_heads
                if self.config.kv_channels is None
                else self.config.kv_channels
            ),
            batch_size=batch_size,
            device=torch.device(device),
            dtype=dtype,
            max_length=max_length,
        )
