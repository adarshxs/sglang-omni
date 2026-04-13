from __future__ import annotations

from typing import Any


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
