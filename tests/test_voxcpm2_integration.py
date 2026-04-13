# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.voxcpm2.config import VoxCPM2PipelineConfig
from sglang_omni.models.voxcpm2.hf_config import (
    ensure_voxcpm2_scaffold_config,
)
from sglang_omni.models.voxcpm2.pipeline.stages import create_preprocessing_executor
from sglang_omni.models.voxcpm2.pipeline.state_io import load_state
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
    assert scaffold_cfg["model_type"] == "voxcpm2_scaffold"
    assert scaffold_cfg["architectures"] == ["VoxCPM2MiniCPMScaffoldForCausalLM"]
    assert scaffold_cfg["native_model_path"] == str(tmp_path)
    assert scaffold_cfg["hidden_size"] == 128


def test_voxcpm2_preprocessing_executor_builds_state() -> None:
    executor = create_preprocessing_executor("unused-model-path")
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"text": "Hello VoxCPM2"},
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
    assert state.generation.max_new_tokens == 123
    assert state.generation.min_len == 4
    assert state.generation.cfg_value == 3.5
    assert state.generation.inference_timesteps == 12
