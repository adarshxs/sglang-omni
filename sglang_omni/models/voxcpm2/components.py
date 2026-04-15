from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.func import jvp

from .minicpm4_static import MiniCPMModel
from .native_config import CfmConfig, MiniCPM4Config


class ScalarQuantizationLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, latent_dim: int = 64, scale: int = 9
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
        self.scale = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.in_proj(hidden))
        if self.training:
            quantized = torch.round(hidden * self.scale) / self.scale
            hidden = hidden + (quantized - hidden).detach()
        else:
            hidden = torch.round(hidden * self.scale) / self.scale
        return self.out_proj(hidden)


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        if config.vocab_size != 0:
            raise ValueError("vocab_size must be 0 for VoxCPMLocEnc")
        self.special_token = nn.Parameter(
            torch.randn(1, 1, 1, config.hidden_size)
        )
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.encoder = MiniCPMModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = x.shape
        x = self.in_proj(x)
        special_tokens = self.special_token.expand(batch_size, seq_len, 1, -1)
        x = torch.cat([special_tokens, x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")
        outputs, _ = self.encoder(x, is_causal=False)
        cls_output = outputs[:, 0, :]
        return rearrange(cls_output, "(b t) c -> b t c", b=batch_size)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb requires an even dimension")
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=x.dtype, device=device) * -emb
        )
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(
            time_embed_dim,
            out_dim if out_dim is not None else time_embed_dim,
            bias=True,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class VoxCPMLocDiTV2(nn.Module):
    def __init__(self, config: MiniCPM4Config, in_channels: int = 64):
        super().__init__()
        if config.vocab_size != 0:
            raise ValueError("vocab_size must be 0 for VoxCPMLocDiTV2")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)
        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )
        self.decoder = MiniCPMModel(config)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_proj(x.transpose(1, 2).contiguous())
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)
        t = self.time_mlp(self.time_embeddings(t).to(x.dtype))
        dt = self.delta_time_mlp(self.time_embeddings(dt).to(x.dtype))
        x = torch.cat([mu.view(x.size(0), -1, x.size(-1)), (t + dt).unsqueeze(1), cond, x], dim=1)
        hidden, _ = self.decoder(x, is_causal=False)
        hidden = hidden[:, prefix + mu.view(x.size(0), -1, x.size(-1)).size(1) + 1 :, :]
        hidden = self.out_proj(hidden)
        return hidden.transpose(1, 2).contiguous()


class UnifiedCFM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiTV2,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.reg_loss_type = cfm_params.reg_loss_type
        self.ratio_r_neq_t_range = cfm_params.ratio_r_neq_t_range
        self.noise_cond_prob_range = cfm_params.noise_cond_prob_range
        self.noise_cond_scale = cfm_params.noise_cond_scale
        self.in_channels = in_channels
        self.mean_mode = mean_mode
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        batch_size = mu.shape[0]
        z = (
            torch.randn(
                (batch_size, self.in_channels, patch_size),
                device=mu.device,
                dtype=mu.dtype,
            )
            * temperature
        )
        t_span = torch.linspace(
            1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype
        )
        t_span = t_span + sway_sampling_coef * (
            torch.cos(torch.pi / 2 * t_span) - 1 + t_span
        )
        return self.solve_euler(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    @staticmethod
    def optimized_scale(
        positive_flat: torch.Tensor, negative_flat: torch.Tensor
    ) -> torch.Tensor:
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        t = t_span[0]
        dt = t_span[0] - t_span[1]
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                batch_size = x.size(0)
                x_in = torch.zeros(
                    [2 * batch_size, self.in_channels, x.size(2)],
                    device=x.device,
                    dtype=x.dtype,
                )
                mu_in = torch.zeros(
                    [2 * batch_size, mu.size(1)],
                    device=x.device,
                    dtype=x.dtype,
                )
                t_in = torch.zeros(
                    [2 * batch_size], device=x.device, dtype=x.dtype
                )
                dt_in = torch.zeros_like(t_in)
                cond_in = torch.zeros(
                    [2 * batch_size, self.in_channels, cond.size(2)],
                    device=x.device,
                    dtype=x.dtype,
                )
                x_in[:batch_size], x_in[batch_size:] = x, x
                mu_in[:batch_size] = mu
                t_in[:batch_size], t_in[batch_size:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:batch_size], dt_in[batch_size:] = dt.unsqueeze(0), dt.unsqueeze(0)
                if not self.mean_mode:
                    dt_in.zero_()
                cond_in[:batch_size], cond_in[batch_size:] = cond, cond
                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(
                    dphi_dt, [x.size(0), x.size(0)], dim=0
                )
                if use_cfg_zero_star:
                    positive_flat = dphi_dt.view(batch_size, -1)
                    negative_flat = cfg_dphi_dt.view(batch_size, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.view(
                        batch_size,
                        *([1] * (len(dphi_dt.shape) - 1)),
                    )
                else:
                    st_star = 1.0
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (
                    dphi_dt - cfg_dphi_dt * st_star
                )
            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
        return x

    def adaptive_loss_weighting(
        self,
        losses: torch.Tensor,
        mask: torch.Tensor | None = None,
        p: float = 0.0,
        epsilon: float = 1e-3,
    ) -> torch.Tensor:
        weights = 1.0 / ((losses + epsilon).pow(p))
        if mask is not None:
            weights = weights * mask
        return weights.detach()

    def sample_r_t(
        self,
        x: torch.Tensor,
        mu: float = -0.4,
        sigma: float = 1.0,
        ratio_r_neq_t: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        if self.t_scheduler == "log-norm":
            s_r = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            s_t = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            r = torch.sigmoid(s_r)
            t = torch.sigmoid(s_t)
        elif self.t_scheduler == "uniform":
            r = torch.rand(batch_size, device=x.device, dtype=x.dtype)
            t = torch.rand(batch_size, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported t_scheduler: {self.t_scheduler}")
        mask = torch.rand(batch_size, device=x.device, dtype=x.dtype) < ratio_r_neq_t
        r, t = torch.where(
            mask,
            torch.stack([torch.min(r, t), torch.max(r, t)], dim=0),
            torch.stack([t, t], dim=0),
        )
        return r.squeeze(), t.squeeze()

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        batch_size = x1.shape[0]
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(batch_size, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1)
        if cond is None:
            cond = torch.zeros_like(x1)
        noisy_mask = torch.rand(batch_size, device=x1.device) > (
            1.0
            - (
                self.noise_cond_prob_range[0]
                + progress
                * (
                    self.noise_cond_prob_range[1]
                    - self.noise_cond_prob_range[0]
                )
            )
        )
        cond = cond + noisy_mask.view(-1, 1, 1) * torch.randn_like(cond) * self.noise_cond_scale
        ratio_r_neq_t = (
            self.ratio_r_neq_t_range[0]
            + progress
            * (
                self.ratio_r_neq_t_range[1]
                - self.ratio_r_neq_t_range[0]
            )
            if self.mean_mode
            else 0.0
        )
        r, t = self.sample_r_t(x1, ratio_r_neq_t=ratio_r_neq_t)
        r_ = r.detach().clone()
        t_ = t.detach().clone()
        z = torch.randn_like(x1)
        y = (1 - t_.view(-1, 1, 1)) * x1 + t_.view(-1, 1, 1) * z
        v = z - x1

        def model_fn(z_sample, r_sample, t_sample):
            return self.estimator(z_sample, mu, t_sample, cond, dt=t_sample - r_sample)

        if self.mean_mode:
            v_r = torch.zeros_like(r)
            v_t = torch.ones_like(t)
            from torch.backends.cuda import sdp_kernel

            with sdp_kernel(enable_flash=False, enable_mem_efficient=False):
                u_pred, dudt = jvp(model_fn, (y, r, t), (v, v_r, v_t))
            u_tgt = v - (t_ - r_).view(-1, 1, 1) * dudt
        else:
            u_pred = model_fn(y, r, t)
            u_tgt = v
        losses = F.mse_loss(u_pred, u_tgt.detach(), reduction="none").mean(dim=1)
        if tgt_mask is not None:
            weights = self.adaptive_loss_weighting(losses, tgt_mask.squeeze(1))
            return (weights * losses).sum() / torch.clamp(
                torch.sum(tgt_mask), min=1.0
            )
        return losses.mean()
