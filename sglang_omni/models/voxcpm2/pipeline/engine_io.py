from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData
from sglang_omni.models.native_adapter import NativeAdapterRequestState
from sglang_omni.models.voxcpm2.audio_vae import compute_audio_patch_count
from sglang_omni.models.voxcpm2.io import VoxCPM2State


def _unwrap_audio_source(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        inner = value[0]
        if isinstance(inner, (list, tuple, dict, str)):
            return inner
    return value


def _request_mode(state: VoxCPM2State) -> str:
    has_reference = state.reference_audio is not None
    has_prompt = state.prompt_audio is not None and bool(state.prompt_text)
    if has_reference and has_prompt:
        return "ref_continuation"
    if has_reference:
        return "reference"
    if has_prompt:
        return "continuation"
    return "zero_shot"


def build_voxcpm2_request(
    state: VoxCPM2State,
    *,
    tokenizer: Any,
    hidden_size: int,
    request_id: str,
    encode_sample_rate: int,
    chunk_size: int,
    patch_size: int,
) -> SGLangARRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    mode = _request_mode(state)
    target_ids = list(tokenizer(state.text))
    prompt_ids = (
        list(tokenizer(state.prompt_text))
        if mode in {"continuation", "ref_continuation"} and state.prompt_text
        else []
    )
    text_ids = prompt_ids + target_ids + [101]

    prompt_audio = _unwrap_audio_source(state.prompt_audio)
    reference_audio = _unwrap_audio_source(state.reference_audio)
    prompt_audio_num_patches = compute_audio_patch_count(
        prompt_audio,
        encode_sample_rate=encode_sample_rate,
        chunk_size=chunk_size,
        patch_size=patch_size,
    )
    reference_audio_num_patches = compute_audio_patch_count(
        reference_audio,
        encode_sample_rate=encode_sample_rate,
        chunk_size=chunk_size,
        patch_size=patch_size,
    )

    if mode == "zero_shot":
        input_ids_list = text_ids
    elif mode == "continuation":
        input_ids_list = text_ids + ([0] * prompt_audio_num_patches)
    elif mode == "reference":
        input_ids_list = (
            [103]
            + ([0] * reference_audio_num_patches)
            + [104]
            + text_ids
        )
    else:
        input_ids_list = (
            [103]
            + ([0] * reference_audio_num_patches)
            + [104]
            + text_ids
            + ([0] * prompt_audio_num_patches)
        )

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    prefill_embeds = torch.zeros(len(input_ids_list), hidden_size, dtype=torch.float32)

    sampling_params = SamplingParams(
        max_new_tokens=state.generation.max_new_tokens,
        temperature=0.0,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(tokenizer.vocab_size)

    req = Req(
        rid=request_id,
        origin_input_text=state.text,
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        input_embeds=prefill_embeds.tolist(),
        vocab_size=tokenizer.vocab_size,
    )
    req._input_embeds_are_projected = True

    return SGLangARRequestData(
        input_ids=input_ids,
        max_new_tokens=state.generation.max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        input_embeds_are_projected=True,
        stop_token_ids=(1,),
        native_adapter=NativeAdapterRequestState(
            prompt_text=state.text,
            metadata={
                "max_new_tokens": state.generation.max_new_tokens,
                "min_len": state.generation.min_len,
                "cfg_value": state.generation.cfg_value,
                "inference_timesteps": state.generation.inference_timesteps,
                "mode": mode,
                "prompt_text": state.prompt_text,
                "prompt_audio": prompt_audio,
                "reference_audio": reference_audio,
                "prompt_audio_num_patches": prompt_audio_num_patches,
                "reference_audio_num_patches": reference_audio_num_patches,
            },
        ),
    )


def apply_voxcpm2_result(state: VoxCPM2State, result: SGLangARRequestData) -> None:
    native_adapter = result.native_adapter
    adapter_state = native_adapter.state if native_adapter is not None else None
    if adapter_state is not None and adapter_state.accumulator is not None:
        state.generated_patches = adapter_state.accumulator.stacked()
        state.sample_rate = adapter_state.accumulator.sample_rate
        state.prompt_tokens = adapter_state.prompt_tokens
        state.completion_tokens = adapter_state.completion_tokens
        state.finish_reason = (
            adapter_state.finish_reason
            or (native_adapter.finish_reason if native_adapter is not None else None)
        )
    elif native_adapter is not None and isinstance(
        native_adapter.latest_multimodal_outputs, dict
    ):
        state.finish_reason = native_adapter.finish_reason
