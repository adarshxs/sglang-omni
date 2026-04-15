from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass
class NativeAdapterRequestState:
    """Per-request state owned by a scaffold-backed native adapter."""

    prompt_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    state: Any = None
    latest_multimodal_outputs: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


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


@runtime_checkable
class NativeAdapterModel(Protocol):
    """Minimal model-local contract for scaffold-backed native adapters."""

    def set_native_adapter_requests(self, requests: list[Any]) -> None: ...

    def clear_native_adapter_requests(self) -> None: ...

    def pop_native_adapter_result(self, request_id: str) -> Any | None: ...


class NativeAdapterScaffoldMixin:
    """Reusable scaffold helpers for package-backed native AR adapters."""

    def _init_native_adapter_scaffold(self) -> None:
        self._native_adapter_request_ids: list[str] = []
        self._native_adapter_request_data: list[Any] = []
        self._native_adapter_step_results: dict[str, Any] = {}

    def prepare_input_embeds(self, input_embeds, **_: Any):
        """Identity projection for already-projected decode-time embeddings."""
        return input_embeds

    def set_native_adapter_requests(self, requests: list[Any]) -> None:
        self._native_adapter_request_ids = [req.request_id for req in requests]
        self._native_adapter_request_data = [req.data for req in requests]

    def clear_native_adapter_requests(self) -> None:
        self._native_adapter_request_ids = []
        self._native_adapter_request_data = []

    def iter_native_adapter_requests(self):
        return zip(self._native_adapter_request_ids, self._native_adapter_request_data)

    def record_native_adapter_result(self, request_id: str, result: Any) -> None:
        self._native_adapter_step_results[request_id] = result

    def pop_native_adapter_result(self, request_id: str) -> Any | None:
        return self._native_adapter_step_results.pop(request_id, None)
