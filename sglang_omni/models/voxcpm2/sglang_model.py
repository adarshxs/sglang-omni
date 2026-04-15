from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import librosa
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.models.native_adapter import (
    NativeAdapterScaffoldMixin,
    NativeAdapterStepResult,
    PatchAudioAccumulator,
)
from sglang_omni.models.weight_loader import default_weight_loader

from .audio_vae_native import AudioVAEConfigV2, AudioVAEV2
from .components import ScalarQuantizationLayer, UnifiedCFM, VoxCPMLocDiTV2, VoxCPMLocEnc
from .minicpm4_sglang import MiniCPMSGLangModel
from .native_config import VoxCPM2NativeConfig

logger = logging.getLogger(__name__)


@dataclass
class _VoxCPM2RequestState:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    finished: bool = False
    accumulator: PatchAudioAccumulator | None = None
    prev_feat_embed: torch.Tensor | None = None
    prefix_feat_cond: torch.Tensor | None = None
    current_embed_for_next: torch.Tensor | None = None


class VoxCPM2ForCausalLM(NativeAdapterScaffoldMixin, nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del quant_config, prefix
        self.config = config
        self._init_native_adapter_scaffold()

        native_cfg = self._native_config_from_hf(config)
        lm_cfg = native_cfg.lm_config
        self.model = MiniCPMSGLangModel(lm_cfg, layer_id_offset=0)

        residual_cfg = lm_cfg.model_copy(deep=True)
        residual_cfg.num_hidden_layers = native_cfg.residual_lm_num_layers
        residual_cfg.vocab_size = 0
        residual_cfg.no_rope = native_cfg.residual_lm_no_rope
        self.residual_model = MiniCPMSGLangModel(
            residual_cfg,
            layer_id_offset=lm_cfg.num_hidden_layers,
        )

        encoder_cfg = lm_cfg.model_copy(deep=True)
        encoder_cfg.hidden_size = native_cfg.encoder_config.hidden_dim
        encoder_cfg.intermediate_size = native_cfg.encoder_config.ffn_dim
        encoder_cfg.num_attention_heads = native_cfg.encoder_config.num_heads
        encoder_cfg.num_hidden_layers = native_cfg.encoder_config.num_layers
        encoder_cfg.kv_channels = native_cfg.encoder_config.kv_channels
        encoder_cfg.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_cfg, input_dim=native_cfg.feat_dim)

        dit_cfg = lm_cfg.model_copy(deep=True)
        dit_cfg.hidden_size = native_cfg.dit_config.hidden_dim
        dit_cfg.intermediate_size = native_cfg.dit_config.ffn_dim
        dit_cfg.num_attention_heads = native_cfg.dit_config.num_heads
        dit_cfg.num_hidden_layers = native_cfg.dit_config.num_layers
        dit_cfg.kv_channels = native_cfg.dit_config.kv_channels
        dit_cfg.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=native_cfg.feat_dim,
            cfm_params=native_cfg.dit_config.cfm_config,
            estimator=VoxCPMLocDiTV2(dit_cfg, in_channels=native_cfg.feat_dim),
            mean_mode=native_cfg.dit_config.dit_mean_mode,
        )

        self.fsq_layer = ScalarQuantizationLayer(
            lm_cfg.hidden_size,
            lm_cfg.hidden_size,
            native_cfg.scalar_quantization_latent_dim,
            native_cfg.scalar_quantization_scale,
        )
        self.enc_to_lm_proj = nn.Linear(
            native_cfg.encoder_config.hidden_dim,
            lm_cfg.hidden_size,
        )
        self.lm_to_dit_proj = nn.Linear(
            lm_cfg.hidden_size,
            native_cfg.dit_config.hidden_dim,
        )
        self.res_to_dit_proj = nn.Linear(
            lm_cfg.hidden_size,
            native_cfg.dit_config.hidden_dim,
        )
        self.fusion_concat_proj = nn.Linear(lm_cfg.hidden_size * 2, lm_cfg.hidden_size)
        self.stop_proj = nn.Linear(lm_cfg.hidden_size, lm_cfg.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(lm_cfg.hidden_size, 2, bias=False)

        vae_cfg = (
            AudioVAEConfigV2.model_validate(native_cfg.audio_vae_config)
            if native_cfg.audio_vae_config is not None
            else AudioVAEConfigV2()
        )
        self.audio_vae = AudioVAEV2(vae_cfg)
        self.audio_vae = self.audio_vae.to(torch.float32)
        self._audio_vae_loaded = False

        self.patch_size = native_cfg.patch_size
        self.feat_dim = native_cfg.feat_dim
        self.chunk_size = self.audio_vae.chunk_size
        self.sample_rate = int(self.audio_vae.out_sample_rate)
        self._encode_sample_rate = int(self.audio_vae.sample_rate)
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

    @staticmethod
    def _native_config_from_hf(config: Any) -> VoxCPM2NativeConfig:
        payload = {
            "lm_config": dict(getattr(config, "lm_config")),
            "patch_size": int(getattr(config, "patch_size", 4)),
            "feat_dim": int(getattr(config, "feat_dim", 64)),
            "residual_lm_num_layers": int(
                getattr(config, "residual_lm_num_layers", 8)
            ),
            "residual_lm_no_rope": bool(
                getattr(config, "residual_lm_no_rope", False)
            ),
            "scalar_quantization_latent_dim": int(
                getattr(config, "scalar_quantization_latent_dim", 512)
            ),
            "scalar_quantization_scale": int(
                getattr(config, "scalar_quantization_scale", 9)
            ),
            "encoder_config": dict(getattr(config, "encoder_config")),
            "dit_config": dict(getattr(config, "dit_config")),
            "audio_vae_config": getattr(config, "audio_vae_config", None),
            "max_length": int(getattr(config, "max_length", 8192)),
            "dtype": str(getattr(config, "dtype", "bfloat16")),
        }
        return VoxCPM2NativeConfig.model_validate(payload)

    def prepare_input_embeds(self, input_embeds, **_: Any):
        return input_embeds

    @property
    def device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    @property
    def side_dtype(self) -> torch.dtype:
        return self.fusion_concat_proj.weight.dtype

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _native_model_path(self) -> str:
        model_path = (
            getattr(self.config, "native_model_path", None)
            or getattr(self.config, "_name_or_path", None)
            or getattr(self.config, "name_or_path", None)
        )
        if not model_path:
            raise RuntimeError("Missing native_model_path for VoxCPM2")
        return model_path

    def _load_audio_vae_weights(self) -> None:
        if self._audio_vae_loaded:
            return
        model_dir = Path(self._native_model_path())
        safetensors_path = model_dir / "audiovae.safetensors"
        pth_path = model_dir / "audiovae.pth"
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path), device="cpu")
        elif pth_path.exists():
            checkpoint = torch.load(
                str(pth_path),
                map_location="cpu",
                weights_only=True,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"Could not find AudioVAE weights under {model_dir}"
            )
        self.audio_vae.load_state_dict(state_dict, strict=True)
        self.audio_vae = self.audio_vae.to(self.device).eval()
        self._audio_vae_loaded = True

    def _iter_native_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name.startswith("base_lm."):
                yield f"model.{name[len('base_lm.'):]}", tensor
            elif name.startswith("residual_lm."):
                yield f"residual_model.{name[len('residual_lm.'):]}", tensor
            elif name.startswith("audio_vae."):
                continue
            else:
                yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, loaded_weight in self._iter_native_weights(weights):
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)
        self._load_audio_vae_weights()
        logger.info(
            "Loaded native VoxCPM2 model (patch_size=%d, feat_dim=%d, sample_rate=%d)",
            self.patch_size,
            self.feat_dim,
            self.sample_rate,
        )
        return loaded

    @staticmethod
    def _generation_config_for(request_data: Any) -> tuple[int, int, float, int]:
        native_adapter = getattr(request_data, "native_adapter", None)
        meta = dict(native_adapter.metadata if native_adapter is not None else {})
        return (
            int(meta.get("max_new_tokens", 4096)),
            int(meta.get("min_len", 2)),
            float(meta.get("cfg_value", 2.0)),
            int(meta.get("inference_timesteps", 10)),
        )

    def _ensure_request_state(self, request_data: Any) -> _VoxCPM2RequestState:
        native_adapter = getattr(request_data, "native_adapter", None)
        if native_adapter is None:
            raise RuntimeError("VoxCPM2 request missing native adapter state")
        if native_adapter.state is None:
            native_adapter.state = _VoxCPM2RequestState(
                accumulator=PatchAudioAccumulator(
                    feature_dim=self.feat_dim,
                    sample_rate=self.sample_rate,
                )
            )
        return native_adapter.state

    @staticmethod
    def _parse_audio_source(source: Any) -> tuple[torch.Tensor, int] | None:
        if source is None:
            return None
        if isinstance(source, list) and len(source) == 2 and isinstance(source[1], int):
            samples, sample_rate = source
            if isinstance(samples, torch.Tensor):
                waveform = samples.detach().cpu().float()
            else:
                waveform = torch.tensor(samples, dtype=torch.float32)
            return waveform, int(sample_rate)
        if isinstance(source, tuple) and len(source) == 2 and isinstance(source[1], int):
            samples, sample_rate = source
            if isinstance(samples, torch.Tensor):
                waveform = samples.detach().cpu().float()
            else:
                waveform = torch.tensor(samples, dtype=torch.float32)
            return waveform, int(sample_rate)
        if isinstance(source, str):
            waveform, sample_rate = librosa.load(
                source,
                sr=None,
                mono=True,
            )
            return torch.from_numpy(waveform).float(), int(sample_rate)
        raise TypeError(f"Unsupported audio source type: {type(source)}")

    def _encode_audio_source(
        self,
        source: Any,
        *,
        padding_mode: str,
    ) -> torch.Tensor:
        parsed = self._parse_audio_source(source)
        if parsed is None:
            raise ValueError("Cannot encode an empty audio source")
        waveform, sample_rate = parsed
        if sample_rate != self._encode_sample_rate:
            waveform = torch.from_numpy(
                librosa.resample(
                    waveform.numpy(),
                    orig_sr=sample_rate,
                    target_sr=self._encode_sample_rate,
                )
            ).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        patch_len = self.patch_size * self.chunk_size
        if waveform.size(1) % patch_len != 0:
            padding_size = patch_len - waveform.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            waveform = torch.nn.functional.pad(waveform, pad)
        feat = self.audio_vae.encode(
            waveform.to(self.device),
            self._encode_sample_rate,
        )
        return feat.view(self.audio_vae.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def _make_ref_prefix(
        self, ref_feat: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_len = ref_feat.size(0)
        zeros = torch.zeros(
            (1, self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
            device=device,
        )
        tokens = torch.cat(
            [
                torch.tensor([self.ref_audio_start_token], dtype=torch.int32, device=device),
                torch.zeros(ref_len, dtype=torch.int32, device=device),
                torch.tensor([self.ref_audio_end_token], dtype=torch.int32, device=device),
            ]
        )
        feats = torch.cat([zeros, ref_feat, zeros], dim=0)
        text_mask = torch.cat(
            [
                torch.tensor([1], dtype=torch.int32, device=device),
                torch.zeros(ref_len, dtype=torch.int32, device=device),
                torch.tensor([1], dtype=torch.int32, device=device),
            ]
        )
        audio_mask = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.ones(ref_len, dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
            ]
        )
        return tokens, feats, text_mask, audio_mask

    def _build_prefill_inputs(
        self,
        request_tokens: torch.Tensor,
        request_data: Any,
    ) -> dict[str, torch.Tensor]:
        native_adapter = request_data.native_adapter
        if native_adapter is None:
            raise RuntimeError("VoxCPM2 request missing adapter metadata")
        meta = dict(native_adapter.metadata)
        mode = str(meta.get("mode", "zero_shot"))
        prompt_len = int(meta.get("prompt_audio_num_patches", 0))
        ref_len = int(meta.get("reference_audio_num_patches", 0))
        device = request_tokens.device

        if mode == "zero_shot":
            text_len = request_tokens.shape[0]
            audio_feat = torch.zeros(
                (text_len, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=device,
            )
            text_mask = torch.ones(text_len, dtype=torch.int32, device=device)
            audio_mask = torch.zeros(text_len, dtype=torch.int32, device=device)
            return {
                "text_token": request_tokens.unsqueeze(0),
                "audio_feat": audio_feat.unsqueeze(0),
                "text_mask": text_mask.unsqueeze(0),
                "audio_mask": audio_mask.unsqueeze(0),
            }

        if mode == "continuation":
            if prompt_len <= 0:
                raise ValueError("Continuation mode requires prompt_audio_num_patches")
            text_len = request_tokens.shape[0] - prompt_len
            prompt_feat = self._encode_audio_source(
                meta.get("prompt_audio"),
                padding_mode="left",
            )
            audio_feat = torch.cat(
                [
                    torch.zeros(
                        (text_len, self.patch_size, self.audio_vae.latent_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                    prompt_feat.to(device),
                ],
                dim=0,
            )
            text_mask = torch.cat(
                [
                    torch.ones(text_len, dtype=torch.int32, device=device),
                    torch.zeros(prompt_len, dtype=torch.int32, device=device),
                ]
            )
            audio_mask = torch.cat(
                [
                    torch.zeros(text_len, dtype=torch.int32, device=device),
                    torch.ones(prompt_len, dtype=torch.int32, device=device),
                ]
            )
            return {
                "text_token": request_tokens.unsqueeze(0),
                "audio_feat": audio_feat.unsqueeze(0),
                "text_mask": text_mask.unsqueeze(0),
                "audio_mask": audio_mask.unsqueeze(0),
            }

        if mode == "reference":
            if ref_len <= 0:
                raise ValueError("Reference mode requires reference_audio_num_patches")
            ref_feat = self._encode_audio_source(
                meta.get("reference_audio"),
                padding_mode="right",
            )
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(
                ref_feat.to(device), device
            )
            text_len = request_tokens.shape[0] - (ref_len + 2)
            text_only = request_tokens[-text_len:]
            audio_feat = torch.cat(
                [
                    ref_feats,
                    torch.zeros(
                        (text_len, self.patch_size, self.audio_vae.latent_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=0,
            )
            text_mask = torch.cat(
                [ref_t_mask, torch.ones(text_len, dtype=torch.int32, device=device)]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_len, dtype=torch.int32, device=device),
                ]
            )
            return {
                "text_token": torch.cat([ref_tokens, text_only]).unsqueeze(0),
                "audio_feat": audio_feat.unsqueeze(0),
                "text_mask": text_mask.unsqueeze(0),
                "audio_mask": audio_mask.unsqueeze(0),
            }

        if mode == "ref_continuation":
            if ref_len <= 0 or prompt_len <= 0:
                raise ValueError(
                    "ref_continuation mode requires reference and prompt audio lengths"
                )
            ref_feat = self._encode_audio_source(
                meta.get("reference_audio"),
                padding_mode="right",
            )
            prompt_feat = self._encode_audio_source(
                meta.get("prompt_audio"),
                padding_mode="left",
            )
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(
                ref_feat.to(device), device
            )
            text_len = request_tokens.shape[0] - (ref_len + 2 + prompt_len)
            text_only = request_tokens[ref_len + 2 : ref_len + 2 + text_len]
            audio_feat = torch.cat(
                [
                    ref_feats,
                    torch.zeros(
                        (text_len, self.patch_size, self.audio_vae.latent_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                    prompt_feat.to(device),
                ],
                dim=0,
            )
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_len, dtype=torch.int32, device=device),
                    torch.zeros(prompt_len, dtype=torch.int32, device=device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_len, dtype=torch.int32, device=device),
                    torch.ones(prompt_len, dtype=torch.int32, device=device),
                ]
            )
            return {
                "text_token": torch.cat(
                    [ref_tokens, text_only, torch.zeros(prompt_len, dtype=torch.int32, device=device)]
                ).unsqueeze(0),
                "audio_feat": audio_feat.unsqueeze(0),
                "text_mask": text_mask.unsqueeze(0),
                "audio_mask": audio_mask.unsqueeze(0),
            }

        raise ValueError(f"Unsupported VoxCPM2 request mode: {mode}")

    def _build_prefill_embeds(
        self,
        active_requests: list[tuple[str, Any]],
        input_ids: torch.Tensor,
        forward_batch: Any,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        seq_lens = forward_batch.extend_seq_lens.tolist()
        offset = 0
        embeds: list[torch.Tensor] = []
        per_request_inputs: list[dict[str, Any]] = []
        for (request_id, request_data), seq_len in zip(active_requests, seq_lens):
            del request_id
            request_tokens = input_ids[offset : offset + seq_len]
            built = self._build_prefill_inputs(request_tokens, request_data)
            text_embed = self.model.embed_input_ids(built["text_token"].to(self.device))
            audio_feat = built["audio_feat"].to(self.device, dtype=torch.float32)
            feat_embed = self.enc_to_lm_proj(
                self.feat_encoder(audio_feat.to(self.side_dtype))
            )
            combined = (
                built["text_mask"].to(self.device).unsqueeze(-1) * text_embed
                + built["audio_mask"].to(self.device).unsqueeze(-1) * feat_embed
            )
            embeds.append(combined.squeeze(0))
            per_request_inputs.append(
                {
                    "text_mask": built["text_mask"].to(self.device),
                    "audio_mask": built["audio_mask"].to(self.device),
                    "audio_feat": audio_feat.to(self.side_dtype),
                    "feat_embed": feat_embed,
                }
            )
            offset += seq_len
        return torch.cat(embeds, dim=0), per_request_inputs

    def _step_result_from_hidden(
        self,
        state: _VoxCPM2RequestState,
        *,
        lm_hidden: torch.Tensor,
        residual_hidden: torch.Tensor,
        generation: tuple[int, int, float, int],
    ) -> NativeAdapterStepResult:
        max_new_tokens, min_len, cfg_value, inference_timesteps = generation
        dit_hidden = torch.cat(
            [self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(residual_hidden)],
            dim=-1,
        )
        if state.prefix_feat_cond is None:
            raise RuntimeError("Missing prefix conditioning for VoxCPM2 request")
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            patch_size=self.patch_size,
            cond=state.prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=inference_timesteps,
            cfg_value=cfg_value,
        ).transpose(1, 2)
        curr_embed = self.enc_to_lm_proj(self.feat_encoder(pred_feat.unsqueeze(1)))[:, 0, :]
        if state.accumulator is None:
            raise RuntimeError("Missing accumulator for VoxCPM2 request")
        state.accumulator.append(pred_feat.squeeze(0).detach().cpu())
        state.completion_tokens += 1
        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_flag = stop_logits.argmax(dim=-1)[0].item()
        finished = False
        finish_reason = None
        if state.completion_tokens >= max_new_tokens:
            finished = True
            finish_reason = "length"
        elif state.completion_tokens > min_len + 1 and stop_flag == 1:
            finished = True
            finish_reason = "stop"

        if finished:
            state.finished = True
            state.finish_reason = finish_reason
            state.current_embed_for_next = None
        else:
            state.prev_feat_embed = curr_embed.detach()
            state.prefix_feat_cond = pred_feat.detach()
            state.current_embed_for_next = curr_embed.detach()

        usage = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        return NativeAdapterStepResult(
            token_id=1 if finished else 0,
            finished=finished,
            finish_reason=finish_reason,
            next_input_embeds=None if finished else state.current_embed_for_next,
            multimodal_outputs={
                "latent_patch": pred_feat.squeeze(0).detach().cpu(),
                "sample_rate": self.sample_rate,
            },
            usage=usage,
        )

    @staticmethod
    def _request_row_spans(
        forward_batch: Any, num_requests: int
    ) -> list[tuple[int, int]]:
        if forward_batch.forward_mode.is_extend():
            seq_lens = forward_batch.extend_seq_lens.tolist()
        else:
            seq_lens = [1] * num_requests
        spans: list[tuple[int, int]] = []
        offset = 0
        for seq_len in seq_lens:
            spans.append((offset, offset + seq_len))
            offset += seq_len
        return spans

    def _build_control_logits(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        finished_rows: list[bool],
    ) -> torch.Tensor:
        logits = torch.full(
            (batch_size, self.config.vocab_size),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        for row_idx, should_stop in enumerate(finished_rows):
            if should_stop:
                logits[row_idx, 0] = 0.0
                logits[row_idx, 1] = 1.0
            else:
                logits[row_idx, 0] = 1.0
                logits[row_idx, 1] = 0.0
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
        del kwargs
        active_requests = list(self.iter_native_adapter_requests())
        if not active_requests:
            raise RuntimeError("VoxCPM2 forward called without active requests")

        is_prefill = bool(forward_batch.forward_mode.is_extend())
        per_request_inputs: list[dict[str, Any]] = []
        if is_prefill:
            actual_embeds, per_request_inputs = self._build_prefill_embeds(
                active_requests,
                input_ids,
                forward_batch,
            )
            base_hidden = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=actual_embeds.to(self.side_dtype),
            )
        else:
            if input_embeds is None:
                raise RuntimeError("VoxCPM2 decode step requires projected feedback embeds")
            base_hidden = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds.to(self.side_dtype),
            )

        spans = self._request_row_spans(forward_batch, len(active_requests))
        residual_inputs: list[torch.Tensor] = []
        residual_positions: list[torch.Tensor] = []
        pending: list[dict[str, Any]] = []

        for request_idx, ((request_id, request_data), (start, end)) in enumerate(
            zip(active_requests, spans)
        ):
            state = self._ensure_request_state(request_data)
            generation = self._generation_config_for(request_data)
            if is_prefill:
                inputs = per_request_inputs[request_idx]
                req_hidden = base_hidden[start:end].unsqueeze(0)
                text_mask = inputs["text_mask"]
                audio_mask = inputs["audio_mask"]
                audio_feat = inputs["audio_feat"]
                feat_embed = inputs["feat_embed"]
                state.prompt_tokens = req_hidden.shape[1]
                enc_outputs = (
                    self.fsq_layer(req_hidden) * audio_mask.unsqueeze(-1)
                    + req_hidden * text_mask.unsqueeze(-1)
                )
                lm_hidden = enc_outputs[:, -1, :]
                residual_input = self.fusion_concat_proj(
                    torch.cat(
                        [enc_outputs, audio_mask.unsqueeze(-1) * feat_embed],
                        dim=-1,
                    )
                ).squeeze(0)
                state.prefix_feat_cond = audio_feat[:, -1, ...]
                residual_inputs.append(residual_input)
                residual_positions.append(positions[start:end])
                pending.append(
                    {
                        "request_id": request_id,
                        "request_data": request_data,
                        "state": state,
                        "generation": generation,
                        "lm_hidden": lm_hidden,
                        "prefill": True,
                    }
                )
            else:
                if state.prev_feat_embed is None:
                    raise RuntimeError("Missing previous feature embedding for decode")
                lm_hidden = self.fsq_layer(base_hidden[start:end])
                residual_input = self.fusion_concat_proj(
                    torch.cat([lm_hidden, state.prev_feat_embed], dim=-1)
                )
                residual_inputs.append(residual_input)
                residual_positions.append(positions[start:end])
                pending.append(
                    {
                        "request_id": request_id,
                        "request_data": request_data,
                        "state": state,
                        "generation": generation,
                        "lm_hidden": lm_hidden,
                        "prefill": False,
                    }
                )

        residual_batch = torch.cat(residual_inputs, dim=0)
        residual_batch_positions = torch.cat(residual_positions, dim=0)
        residual_hidden = self.residual_model(
            input_ids=residual_batch_positions,
            positions=residual_batch_positions,
            forward_batch=forward_batch,
            input_embeds=residual_batch.to(self.side_dtype),
        )

        step_results: list[NativeAdapterStepResult] = []
        offset = 0
        for item in pending:
            span = item["lm_hidden"].shape[0]
            request_state = item["state"]
            request_residual = residual_hidden[offset : offset + span]
            offset += span
            if item["prefill"]:
                request_residual = request_residual[-1:, :]
            step_result = self._step_result_from_hidden(
                request_state,
                lm_hidden=item["lm_hidden"],
                residual_hidden=request_residual,
                generation=item["generation"],
            )
            self.record_native_adapter_result(item["request_id"], step_result)
            step_results.append(step_result)

        control_logits = self._build_control_logits(
            len(step_results),
            device=base_hidden.device,
            dtype=base_hidden.dtype,
            finished_rows=[bool(result.finished) for result in step_results],
        )
        return LogitsProcessorOutput(
            next_token_logits=control_logits,
            hidden_states=base_hidden,
        )


EntryClass = VoxCPM2ForCausalLM
