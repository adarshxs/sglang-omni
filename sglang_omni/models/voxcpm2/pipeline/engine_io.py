from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.runtime.native_adapter import NativeAdapterRequestData
from sglang_omni.models.voxcpm2.io import VoxCPM2State


def build_voxcpm2_request(
    state: VoxCPM2State,
    *,
    tokenizer: Any,
    hidden_size: int,
    request_id: str,
) -> NativeAdapterRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids_list = tokenizer.encode(state.text, add_special_tokens=True)
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

    return NativeAdapterRequestData(
        input_ids=input_ids,
        max_new_tokens=state.generation.max_new_tokens,
        temperature=0.0,
        output_ids=req.output_ids,
        req=req,
        prompt_text=state.text,
        adapter_metadata={
            "max_new_tokens": state.generation.max_new_tokens,
            "min_len": state.generation.min_len,
            "cfg_value": state.generation.cfg_value,
            "inference_timesteps": state.generation.inference_timesteps,
        },
        stop_token_ids=(1,),
    )


def apply_voxcpm2_result(state: VoxCPM2State, result: NativeAdapterRequestData) -> None:
    adapter_state = result.adapter_state
    if adapter_state is not None and adapter_state.accumulator is not None:
        state.generated_patches = adapter_state.accumulator.stacked()
        state.sample_rate = adapter_state.accumulator.sample_rate
        state.prompt_tokens = adapter_state.prompt_tokens
        state.completion_tokens = adapter_state.completion_tokens
        state.finish_reason = adapter_state.finish_reason or result.finish_reason
    elif isinstance(result.latest_multimodal_outputs, dict):
        state.finish_reason = result.finish_reason
