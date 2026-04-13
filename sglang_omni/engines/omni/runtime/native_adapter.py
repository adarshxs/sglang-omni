from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..types import RequestOutput, SchedulerOutput, SchedulerRequest
from .sglang_ar import (
    SGLangARRequestData,
    SGLangIterationController,
    SGLangModelRunner,
    SGLangOutputProcessor,
)


@dataclass
class NativeAdapterStepResult:
    """Per-step side results produced by a scaffold-backed native adapter."""

    token_id: int | None = None
    finished: bool = False
    finish_reason: str | None = None
    next_input_embeds: torch.Tensor | None = None
    multimodal_outputs: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeAdapterRequestData(SGLangARRequestData):
    """SGLang AR request data extended with native adapter state."""

    prompt_text: str = ""
    adapter_metadata: dict[str, Any] = field(default_factory=dict)
    adapter_state: Any = None
    latest_multimodal_outputs: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    feedback_embeds: torch.Tensor | None = None
    input_embeds_are_projected: bool = True
    stop_token_ids: tuple[int, ...] = ()


class NativeAdapterOutputProcessor(SGLangOutputProcessor):
    """Collect per-step results from a scaffold-backed native adapter model."""

    def __init__(self, adapter_model: Any):
        super().__init__(capture_hidden=False)
        self._adapter_model = adapter_model

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        token_list = (
            model_output.next_token_ids.tolist()
            if getattr(model_output, "next_token_ids", None) is not None
            else []
        )

        outputs: dict[str, RequestOutput] = {}
        for index, sched_req in enumerate(scheduler_output.requests):
            step_result = self._adapter_model.pop_native_adapter_result(
                sched_req.request_id
            )
            if step_result is None:
                token_id = token_list[index] if index < len(token_list) else None
                step_result = NativeAdapterStepResult(token_id=token_id)
            elif step_result.token_id is None and index < len(token_list):
                step_result.token_id = token_list[index]

            extra = dict(step_result.extra)
            if step_result.next_input_embeds is not None:
                extra["_next_input_embeds"] = step_result.next_input_embeds
            if step_result.multimodal_outputs:
                extra["multimodal_outputs"] = step_result.multimodal_outputs
            if step_result.usage is not None:
                extra["usage"] = step_result.usage

            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=step_result.token_id,
                finished=step_result.finished,
                finish_reason=step_result.finish_reason,
                extra=extra or None,
            )

        return outputs


class NativeAdapterIterationController(SGLangIterationController):
    """Iteration controller for scaffold-backed native adapters."""

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: NativeAdapterRequestData = request.data
        req = data.req

        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        extra = output.extra or {}
        next_input_embeds = extra.get("_next_input_embeds")
        data.feedback_embeds = next_input_embeds

        multimodal_outputs = extra.get("multimodal_outputs")
        if isinstance(multimodal_outputs, dict):
            data.latest_multimodal_outputs = multimodal_outputs

        usage = extra.get("usage")
        if isinstance(usage, dict):
            data.usage = usage

        if output.finish_reason is not None:
            data.finish_reason = output.finish_reason

        if output.data is not None:
            req.output_ids.append(int(output.data))
            data.generation_steps += 1
            if not self.is_finished(request, output) and req.decode_batch_idx == 0:
                self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: NativeAdapterRequestData = request.data

        if output.finished:
            return True
        if output.finish_reason is not None:
            return True
        if output.data is not None and output.data in set(data.stop_token_ids):
            return True

        max_tokens = data.max_new_tokens or getattr(
            data.req.sampling_params, "max_new_tokens", None
        )
        if max_tokens is not None and data.generation_steps >= max_tokens:
            data.finish_reason = data.finish_reason or "length"
            return True

        return False


class NativeAdapterModelRunner(SGLangModelRunner):
    """Thin wrapper that injects active request context into adapter models."""

    def execute(self, scheduler_output: SchedulerOutput):
        set_active = getattr(self._inner_model, "set_native_adapter_requests", None)
        clear_active = getattr(self._inner_model, "clear_native_adapter_requests", None)

        if callable(set_active):
            set_active(scheduler_output.requests)
        try:
            return super().execute(scheduler_output)
        finally:
            if callable(clear_active):
                clear_active()
