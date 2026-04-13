from __future__ import annotations

import logging
import importlib
from typing import Any, Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.minicpm import MiniCPMForCausalLM

from sglang_omni.engines.omni.runtime.native_adapter import NativeAdapterStepResult
from sglang_omni.models.native_adapter import NativeAdapterScaffoldMixin, PatchAudioAccumulator
from sglang_omni.models.voxcpm2.import_utils import import_voxcpm_package
from sglang_omni.models.voxcpm2.io import VoxCPM2AdapterState, VoxCPM2GenerationConfig

logger = logging.getLogger(__name__)


class VoxCPM2MiniCPMScaffoldForCausalLM(
    NativeAdapterScaffoldMixin, MiniCPMForCausalLM
):
    """MiniCPM scaffold plus native VoxCPM2 side computation."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self._init_native_adapter_scaffold()
        self._tts = None
        self._native_device: torch.device | None = None
        self._patch_size = 0
        self._feat_dim = 0
        self._sample_rate = 48000

    @property
    def tts(self):
        if self._tts is None:
            raise RuntimeError("Native VoxCPM2 backend is not loaded.")
        return self._tts

    def _iter_scaffold_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if not name.startswith("base_lm."):
                continue
            inner = name[len("base_lm.") :]
            if inner.startswith(("embed_tokens.", "layers.", "norm.")):
                yield f"model.{inner}", tensor

    def _load_native_backend(self) -> None:
        if self._tts is not None:
            return

        import_voxcpm_package()
        VoxCPM = importlib.import_module("voxcpm.core").VoxCPM

        model_path = (
            getattr(self.config, "native_model_path", None)
            or getattr(self.config, "_name_or_path", None)
            or getattr(self.config, "name_or_path", None)
        )
        if not model_path:
            raise RuntimeError("Missing native_model_path for VoxCPM2 scaffold.")

        native = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=False,
            optimize=False,
        )
        self._tts = native.tts_model.eval()
        self._native_device = next(self.parameters()).device
        self._tts = self._tts.to(self._native_device)
        self._patch_size = int(self._tts.patch_size)
        self._feat_dim = int(self._tts.feat_dim)
        self._sample_rate = int(self._tts.sample_rate)
        logger.info(
            "Loaded native VoxCPM2 backend (patch_size=%d, feat_dim=%d, sample_rate=%d)",
            self._patch_size,
            self._feat_dim,
            self._sample_rate,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        super().load_weights(self._iter_scaffold_weights(weights))
        self._load_native_backend()

    def _generation_config_for(
        self, request_data: Any
    ) -> VoxCPM2GenerationConfig:
        meta = dict(getattr(request_data, "adapter_metadata", {}) or {})
        return VoxCPM2GenerationConfig(
            max_new_tokens=int(meta.get("max_new_tokens", 4096)),
            min_len=int(meta.get("min_len", 2)),
            cfg_value=float(meta.get("cfg_value", 2.0)),
            inference_timesteps=int(meta.get("inference_timesteps", 10)),
        )

    def _build_zero_shot_state(
        self,
        *,
        prompt_text: str,
    ) -> VoxCPM2AdapterState:
        tts = self.tts
        device = self._native_device or next(self.parameters()).device
        dtype = tts._dtype()

        text_token = torch.LongTensor(tts.text_tokenizer(prompt_text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor([tts.audio_start_token], dtype=torch.int32),
            ],
            dim=-1,
        )
        text_length = int(text_token.shape[0])
        audio_feat = torch.zeros(
            (text_length, tts.patch_size, tts.audio_vae.latent_dim),
            dtype=torch.float32,
        )
        text_mask = torch.ones(text_length, dtype=torch.int32)
        audio_mask = torch.zeros(text_length, dtype=torch.int32)

        text_token = text_token.unsqueeze(0).to(device)
        text_mask = text_mask.unsqueeze(0).to(device)
        audio_feat = audio_feat.unsqueeze(0).to(device=device, dtype=dtype)
        audio_mask = audio_mask.unsqueeze(0).to(device)

        prefill_encoder = getattr(tts, "_feat_encoder_raw", tts.feat_encoder)
        feat_embed = prefill_encoder(audio_feat)
        feat_embed = tts.enc_to_lm_proj(feat_embed)

        scale_emb = (
            tts.config.lm_config.scale_emb
            if getattr(tts.config.lm_config, "use_mup", False)
            else 1.0
        )
        text_embed = tts.base_lm.embed_tokens(text_token) * scale_emb
        combined_embed = (
            text_mask.unsqueeze(-1) * text_embed
            + audio_mask.unsqueeze(-1) * feat_embed
        )

        enc_outputs, kv_cache_tuple = tts.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        tts.base_lm.kv_cache.fill_caches(kv_cache_tuple)
        enc_outputs = (
            tts.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1)
            + enc_outputs * text_mask.unsqueeze(-1)
        )
        lm_hidden = enc_outputs[:, -1, :]

        residual_inputs = tts.fusion_concat_proj(
            torch.cat(
                (enc_outputs, audio_mask.unsqueeze(-1) * feat_embed),
                dim=-1,
            )
        )
        residual_outputs, residual_kv_cache_tuple = tts.residual_lm(
            inputs_embeds=residual_inputs,
            is_causal=True,
        )
        tts.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_outputs[:, -1, :]

        return VoxCPM2AdapterState(
            lm_hidden=lm_hidden,
            residual_hidden=residual_hidden,
            prefix_feat_cond=audio_feat[:, -1, ...],
            prompt_tokens=text_length,
            accumulator=PatchAudioAccumulator(
                feature_dim=self._feat_dim,
                sample_rate=self._sample_rate,
            ),
        )

    def _run_one_step(
        self,
        state: VoxCPM2AdapterState,
        generation: VoxCPM2GenerationConfig,
    ) -> NativeAdapterStepResult:
        tts = self.tts
        assert state.lm_hidden is not None
        assert state.residual_hidden is not None
        assert state.prefix_feat_cond is not None
        assert state.accumulator is not None

        dit_hidden = torch.cat(
            (
                tts.lm_to_dit_proj(state.lm_hidden),
                tts.res_to_dit_proj(state.residual_hidden),
            ),
            dim=-1,
        )
        pred_feat = tts.feat_decoder(
            mu=dit_hidden,
            patch_size=tts.patch_size,
            cond=state.prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=generation.inference_timesteps,
            cfg_value=generation.cfg_value,
        ).transpose(1, 2)

        curr_embed = tts.feat_encoder(pred_feat.unsqueeze(1))
        curr_embed = tts.enc_to_lm_proj(curr_embed)

        state.accumulator.append(pred_feat.squeeze(0).cpu())
        state.completion_tokens += 1

        stop_flag = (
            tts.stop_head(tts.stop_actn(tts.stop_proj(state.lm_hidden)))
            .argmax(dim=-1)[0]
            .item()
        )
        stop_allowed = state.completion_tokens > generation.min_len + 1
        finished = bool(stop_allowed and stop_flag == 1)

        next_input_embeds = None
        finish_reason = None
        if not finished:
            position_id = torch.tensor(
                [tts.base_lm.kv_cache.step()],
                device=curr_embed.device,
            )
            next_lm_hidden = tts.base_lm.forward_step(
                curr_embed[:, 0, :],
                position_id,
            ).clone()
            next_lm_hidden = tts.fsq_layer(next_lm_hidden)
            residual_input = tts.fusion_concat_proj(
                torch.cat((next_lm_hidden, curr_embed[:, 0, :]), dim=-1)
            )
            next_residual_hidden = tts.residual_lm.forward_step(
                residual_input,
                torch.tensor(
                    [tts.residual_lm.kv_cache.step()],
                    device=curr_embed.device,
                ),
            ).clone()

            state.lm_hidden = next_lm_hidden
            state.residual_hidden = next_residual_hidden
            state.prefix_feat_cond = pred_feat
            state.current_embed_for_next = curr_embed[:, 0, :].detach()
            next_input_embeds = state.current_embed_for_next
        else:
            state.finished = True
            state.finish_reason = "stop"
            state.current_embed_for_next = None
            finish_reason = "stop"

        usage = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        return NativeAdapterStepResult(
            token_id=1 if finished else 0,
            finished=finished,
            finish_reason=finish_reason,
            next_input_embeds=next_input_embeds,
            multimodal_outputs={
                "latent_patch": pred_feat.squeeze(0).detach().cpu(),
                "sample_rate": self._sample_rate,
            },
            usage=usage,
        )

    def _run_prefill_for_request(
        self,
        *,
        request_id: str,
        request_data: Any,
    ) -> NativeAdapterStepResult:
        prompt_text = request_data.prompt_text
        if not prompt_text:
            raise ValueError("VoxCPM2 request is missing prompt_text.")

        request_data.adapter_state = self._build_zero_shot_state(prompt_text=prompt_text)
        return self._run_one_step(
            request_data.adapter_state,
            self._generation_config_for(request_data),
        )

    def _run_decode_for_request(
        self,
        *,
        request_data: Any,
    ) -> NativeAdapterStepResult:
        state = request_data.adapter_state
        if state is None:
            raise RuntimeError("Missing adapter state for VoxCPM2 decode step.")
        return self._run_one_step(
            state,
            self._generation_config_for(request_data),
        )

    @staticmethod
    def _build_control_logits(
        scaffold_logits: torch.Tensor,
        *,
        continue_token_id: int = 0,
        stop_token_id: int = 1,
        should_stop: bool,
    ) -> torch.Tensor:
        logits = torch.full_like(scaffold_logits, float("-inf"))
        if should_stop:
            logits[:, continue_token_id] = 0.0
            logits[:, stop_token_id] = 1.0
        else:
            logits[:, continue_token_id] = 1.0
            logits[:, stop_token_id] = 0.0
        return logits

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        scaffold_output = super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        active_requests = list(self.iter_native_adapter_requests())
        if not active_requests:
            return scaffold_output
        if len(active_requests) != 1:
            raise RuntimeError(
                "VoxCPM2 scaffold currently supports one native request at a time."
            )

        request_id, request_data = active_requests[0]
        is_prefill = forward_batch.forward_mode.is_extend() and (
            getattr(request_data, "generation_steps", 0) == 0
        )
        if is_prefill:
            step_result = self._run_prefill_for_request(
                request_id=request_id,
                request_data=request_data,
            )
        else:
            step_result = self._run_decode_for_request(request_data=request_data)

        self.record_native_adapter_result(request_id, step_result)

        control_logits = self._build_control_logits(
            scaffold_output.next_token_logits,
            should_stop=bool(step_result.finished),
        )
        return LogitsProcessorOutput(
            next_token_logits=control_logits,
            hidden_states=scaffold_output.hidden_states,
        )


EntryClass = VoxCPM2MiniCPMScaffoldForCausalLM
