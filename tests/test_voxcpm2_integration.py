# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.native_adapter import NativeAdapterRequestState
from sglang_omni.models.voxcpm2.config import VoxCPM2PipelineConfig
from sglang_omni.models.voxcpm2.hf_config import (
    ensure_voxcpm2_scaffold_config,
)
from sglang_omni.models.voxcpm2.pipeline.engine_io import build_voxcpm2_request
from sglang_omni.models.voxcpm2.pipeline.stages import create_preprocessing_executor
from sglang_omni.models.voxcpm2.pipeline.state_io import load_state
from sglang_omni.models.voxcpm2.io import VoxCPM2State
from sglang_omni.proto import OmniRequest, StagePayload


def _write_voxcpm2_raw_config(model_dir: Path) -> None:
    config = {
        "architecture": "voxcpm2",
        "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 128,
            "intermediate_size": 256,
            "max_position_embeddings": 1024,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [1.0] * 64,
                "short_factor": [1.0] * 64,
                "original_max_position_embeddings": 1024,
            },
            "vocab_size": 32000,
            "use_mup": True,
            "scale_emb": 1.0,
            "dim_model_base": 128,
            "scale_depth": 1.0,
            "rope_theta": 10000.0,
        },
        "patch_size": 4,
        "feat_dim": 4,
        "residual_lm_num_layers": 2,
        "residual_lm_no_rope": True,
        "scalar_quantization_latent_dim": 8,
        "scalar_quantization_scale": 9,
        "encoder_config": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "kv_channels": 16,
        },
        "dit_config": {
            "hidden_dim": 32,
            "ffn_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "kv_channels": 16,
            "dit_mean_mode": False,
            "cfm_config": {
                "sigma_min": 1e-6,
                "solver": "euler",
                "t_scheduler": "log-norm",
                "training_cfg_rate": 0.1,
                "inference_cfg_rate": 1.0,
                "reg_loss_type": "l1",
                "ratio_r_neq_t_range": [0.25, 0.75],
                "noise_cond_prob_range": [0.0, 0.0],
                "noise_cond_scale": 0.0,
            },
        },
        "audio_vae_config": {
            "encoder_dim": 8,
            "encoder_rates": [2, 2],
            "latent_dim": 4,
            "decoder_dim": 16,
            "decoder_rates": [2, 2],
            "depthwise": True,
            "sample_rate": 16000,
            "out_sample_rate": 48000,
            "use_noise_block": False,
            "sr_bin_boundaries": [20000],
            "cond_type": "scale_bias",
            "cond_dim": 8,
            "cond_out_layer": False,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_config_manager_resolves_voxcpm2_raw_config(tmp_path: Path) -> None:
    _write_voxcpm2_raw_config(tmp_path)

    manager = ConfigManager.from_model_path(str(tmp_path))

    assert isinstance(manager.config, VoxCPM2PipelineConfig)
    assert manager.config.model_path == str(tmp_path)


def test_ensure_voxcpm2_scaffold_config_writes_hf_bridge(tmp_path: Path) -> None:
    _write_voxcpm2_raw_config(tmp_path)

    resolved, config_basename = ensure_voxcpm2_scaffold_config(str(tmp_path))

    assert resolved == str(tmp_path)
    config_path = tmp_path / config_basename
    assert config_path.exists()

    scaffold_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert scaffold_cfg["model_type"] == "voxcpm2_native"
    assert scaffold_cfg["architectures"] == ["VoxCPM2ForCausalLM"]
    assert scaffold_cfg["native_model_path"] == str(tmp_path)
    assert scaffold_cfg["hidden_size"] == 128
    assert scaffold_cfg["num_attention_heads"] == 8
    assert scaffold_cfg["lm_config"]["hidden_size"] == 128
    assert scaffold_cfg["audio_vae_config"]["sample_rate"] == 16000


def test_voxcpm2_preprocessing_executor_builds_state() -> None:
    executor = create_preprocessing_executor("unused-model-path")
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={
                "text": "Hello VoxCPM2",
                "prompt_text": "Hi",
                "prompt_audio": [[[0.0] * 32, 16000]],
                "additional_information": {
                    "reference_audio": [[[0.1] * 32, 16000]],
                },
            },
            params={
                "max_new_tokens": 123,
                "stage_params": {
                    "generation": {
                        "min_len": 4,
                        "cfg_value": 3.5,
                        "inference_timesteps": 12,
                    }
                },
            },
        ),
        data={},
    )

    result = executor._processor(payload)
    state = load_state(result)

    assert state.text == "Hello VoxCPM2"
    assert state.prompt_text == "Hi"
    assert state.prompt_audio == [[0.0] * 32, 16000]
    assert state.reference_audio == [[0.1] * 32, 16000]
    assert state.generation.max_new_tokens == 123
    assert state.generation.min_len == 4
    assert state.generation.cfg_value == 3.5
    assert state.generation.inference_timesteps == 12


class _DummyTokenizer:
    def __init__(self):
        self.vocab_size = 32000
        self._mapping = {
            "hello": [11, 12],
            "prompt": [21],
        }

    def __call__(self, text: str):
        return list(self._mapping[text])


def test_build_voxcpm2_request_uses_grouped_native_adapter_state() -> None:
    request = build_voxcpm2_request(
        VoxCPM2State(text="hello"),
        tokenizer=_DummyTokenizer(),
        hidden_size=16,
        request_id="req-123",
        encode_sample_rate=16000,
        chunk_size=4,
        patch_size=4,
    )

    assert request.stop_token_ids == (1,)
    assert request.input_embeds_are_projected is True
    assert isinstance(request.native_adapter, NativeAdapterRequestState)
    assert request.native_adapter is not None
    assert request.native_adapter.prompt_text == "hello"
    assert request.native_adapter.metadata["max_new_tokens"] == 4096
    assert request.native_adapter.metadata["min_len"] == 2
    assert request.native_adapter.metadata["mode"] == "zero_shot"
    assert request.native_adapter.metadata["prompt_audio_num_patches"] == 0
    assert request.native_adapter.metadata["reference_audio_num_patches"] == 0
    assert request.input_ids.tolist() == [11, 12, 101]


def test_build_voxcpm2_request_reference_mode_builds_ref_prefix() -> None:
    request = build_voxcpm2_request(
        VoxCPM2State(
            text="hello",
            reference_audio=[[0.1] * 32, 16000],
        ),
        tokenizer=_DummyTokenizer(),
        hidden_size=16,
        request_id="req-ref",
        encode_sample_rate=16000,
        chunk_size=4,
        patch_size=4,
    )

    assert request.native_adapter is not None
    assert request.native_adapter.metadata["mode"] == "reference"
    assert request.native_adapter.metadata["reference_audio_num_patches"] == 2
    assert request.input_ids.tolist() == [103, 0, 0, 104, 11, 12, 101]


def test_build_voxcpm2_request_continuation_mode_adds_prompt_placeholders() -> None:
    request = build_voxcpm2_request(
        VoxCPM2State(
            text="hello",
            prompt_text="prompt",
            prompt_audio=[[0.0] * 32, 16000],
        ),
        tokenizer=_DummyTokenizer(),
        hidden_size=16,
        request_id="req-cont",
        encode_sample_rate=16000,
        chunk_size=4,
        patch_size=4,
    )

    assert request.native_adapter is not None
    assert request.native_adapter.metadata["mode"] == "continuation"
    assert request.native_adapter.metadata["prompt_audio_num_patches"] == 2
    assert request.input_ids.tolist() == [21, 11, 12, 101, 0, 0]
