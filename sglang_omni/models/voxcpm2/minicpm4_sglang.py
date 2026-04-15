from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import RadixAttention

from .minicpm4_static import (
    MiniCPMLongRoPE,
    MiniCPMMLP,
    MiniCPMRMSNorm,
    apply_rotary_pos_emb,
)
from .native_config import MiniCPM4Config


class MiniCPMSGLangAttention(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_id: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_key_value_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
        rope_emb: MiniCPMLongRoPE | None = None,
    ) -> Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if rope_emb is not None:
            cos, sin = rope_emb(positions)
            tokens = q.shape[0]
            q_r = q.view(tokens, self.num_heads, self.head_dim)
            k_r = k.view(tokens, self.num_key_value_heads, self.head_dim)
            q_r = q_r.unsqueeze(0).transpose(1, 2)
            k_r = k_r.unsqueeze(0).transpose(1, 2)
            q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin)
            q = q_r.transpose(1, 2).squeeze(0).reshape(tokens, -1)
            k = k_r.transpose(1, 2).squeeze(0).reshape(tokens, -1)

        attn_output = self.attn(q, k, v, forward_batch)
        return self.o_proj(attn_output)


class MiniCPMSGLangDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniCPM4Config,
        *,
        layer_id: int,
        rope_emb: MiniCPMLongRoPE | None,
    ):
        super().__init__()
        self.self_attn = MiniCPMSGLangAttention(config, layer_id=layer_id)
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rope_emb = rope_emb
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def _residual_scale(self) -> float:
        if self.use_mup:
            return self.scale_depth / math.sqrt(self.num_hidden_layers)
        return 1.0

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            forward_batch,
            self.rope_emb,
        )
        hidden_states = residual + hidden_states * self._residual_scale()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self._residual_scale()
        return hidden_states


class MiniCPMSGLangModel(nn.Module):
    def __init__(
        self,
        config: MiniCPM4Config,
        *,
        layer_id_offset: int = 0,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.start_layer = 0
        self.end_layer = self.num_layers
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size)
            if config.vocab_size > 0
            else nn.Identity()
        )
        self.rope_emb = None if config.no_rope else MiniCPMLongRoPE(config)
        self.layers = nn.ModuleList(
            [
                MiniCPMSGLangDecoderLayer(
                    config,
                    layer_id=layer_id_offset + layer_idx,
                    rope_emb=self.rope_emb,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._scale_emb = config.scale_emb if config.use_mup else 1.0

    def embed_input_ids(self, input_ids: Tensor, **_: Any) -> Tensor:
        if isinstance(self.embed_tokens, nn.Identity):
            raise RuntimeError("embed_input_ids is unavailable when vocab_size=0")
        return self.embed_tokens(input_ids) * self._scale_emb

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Tensor | None = None,
    ) -> Tensor:
        if input_embeds is None:
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = input_embeds.to(device=input_ids.device)
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[layer_idx](
                positions,
                hidden_states,
                forward_batch,
            )
        return self.norm(hidden_states)
