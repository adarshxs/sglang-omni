from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import LlamaTokenizerFast

from sglang_omni.engines import create_sglang_native_adapter_engine
from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.native_adapter import PatchAudioAccumulator
from sglang_omni.models.voxcpm2.audio_vae import load_voxcpm2_audio_vae
from sglang_omni.models.voxcpm2.hf_config import ensure_voxcpm2_scaffold_config
from sglang_omni.models.voxcpm2.io import VoxCPM2GenerationConfig, VoxCPM2State
from sglang_omni.models.voxcpm2.pipeline.engine_io import (
    apply_voxcpm2_result,
    build_voxcpm2_request,
)
from sglang_omni.models.voxcpm2.pipeline.next_stage import GENERATION_STAGE
from sglang_omni.models.voxcpm2.pipeline.state_io import load_state, store_state
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    del model_path

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        params = payload.request.params or {}

        if isinstance(inputs, str):
            text = inputs
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
        else:
            text = str(inputs) if inputs is not None else ""

        if not text or not text.strip():
            raise ValueError("VoxCPM2 requires a non-empty input text.")

        stage_params = params.get("stage_params") or {}
        generation_params = (
            stage_params.get(GENERATION_STAGE)
            or stage_params.get("generation")
            or {}
        )
        state = VoxCPM2State(
            text=text,
            generation=VoxCPM2GenerationConfig(
                max_new_tokens=int(
                    generation_params.get(
                        "max_new_tokens",
                        params.get("max_new_tokens", 4096),
                    )
                ),
                min_len=int(generation_params.get("min_len", 2)),
                cfg_value=float(generation_params.get("cfg_value", 2.0)),
                inference_timesteps=int(
                    generation_params.get("inference_timesteps", 10)
                ),
            ),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


def create_generation_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    mem_fraction_static: float = 0.85,
) -> EngineExecutor:
    from sglang.srt.server_args import ServerArgs
    from transformers import AutoConfig

    from sglang_omni.models.voxcpm2.hf_config import VoxCPM2HFConfig

    # Register the custom config type before ServerArgs.__post_init__ tries
    # AutoConfig.from_pretrained on the scaffold config file.
    try:
        AutoConfig.register("voxcpm2_scaffold", VoxCPM2HFConfig)
    except ValueError:
        pass  # already registered

    resolved_model_path, config_file = ensure_voxcpm2_scaffold_config(model_path)
    tokenizer = LlamaTokenizerFast.from_pretrained(resolved_model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    server_args = ServerArgs(
        model_path=resolved_model_path,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=mem_fraction_static,
        chunked_prefill_size=8192,
        max_prefill_tokens=8192,
        max_running_requests=1,
        disable_cuda_graph=True,
        trust_remote_code=True,
        decrypted_config_file=config_file,
    )

    engine = create_sglang_native_adapter_engine(
        server_args=server_args,
        gpu_id=gpu_id,
        model_arch_override="VoxCPM2MiniCPMScaffoldForCausalLM",
    )
    hidden_size = engine.model_runner.model_worker.model_config.hidden_size

    def _request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_voxcpm2_request(
            state,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            request_id=payload.request_id,
        )

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_voxcpm2_result(state, result)
        return store_state(payload, state)

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
    )


def create_audio_decode_executor(
    model_path: str,
    *,
    device: str = "cpu",
) -> PreprocessingExecutor:
    resolved_model_path = str(resolve_model_path(model_path, local_files_only=False))
    audio_vae = load_voxcpm2_audio_vae(resolved_model_path, device=device)
    default_sample_rate = int(getattr(audio_vae, "out_sample_rate", 48000))

    def _decode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        patches = state.generated_patches
        if patches is None:
            payload = store_state(payload, state)
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = default_sample_rate
            payload.data["modality"] = "audio"
            return payload

        if not isinstance(patches, torch.Tensor):
            patches = torch.tensor(patches)

        accumulator = PatchAudioAccumulator(
            feature_dim=int(patches.shape[-1]),
            sample_rate=default_sample_rate,
        )
        accumulator.append(patches)
        audio = accumulator.decode_full(audio_vae.decode, device=device)
        if audio is None:
            audio = torch.empty(0, dtype=torch.float32)

        payload = store_state(payload, state)
        payload.data["audio_data"] = audio.tolist()
        payload.data["sample_rate"] = default_sample_rate
        payload.data["modality"] = "audio"
        payload.data["usage"] = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        return payload

    return PreprocessingExecutor(_decode)
