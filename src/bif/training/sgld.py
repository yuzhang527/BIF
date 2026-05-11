"""Localized SGLD sampler for Bayesian Influence Function.

Aligned with devinterp SGMCMC update rules:

Plain SGLD:
    Δθ = -(ε/2)(nβ·∇L + γ(θ₀-θ) + λθ) + √ε·σ·N(0,1)

RMSprop-SGLD:
    V_t = α·V_{t-1} + (1-α)·g_t²          (g_t = raw p.grad)
    G_t = 1 / (√V_t + eps)
    Δθ = -(ε/2)·G·(nβ·∇L + γ(θ₀-θ) + λθ) + √(ε·G)·σ·N(0,1)

where:
    nβ = beta * N (inverse temperature × dataset size)
    σ = noise_level (default 1.0; changing this breaks SGLD posterior guarantees)
    γ = localization strength (gamma)
    λ = weight_decay

Key: V is computed from raw gradients (NOT scaled by nbeta), so nbeta
correctly controls the loss-gradient / prior / noise balance — matching
devinterp's RMSpropPreconditioner which uses raw grad for square_avg.

Mixed-precision: When model parameters are in bfloat16/float16, SGLD
noise and parameter updates must be computed in float32 to avoid
catastrophic precision loss.  With lr=5e-6 and bf16 params, ~78% of
noise is rounded to zero by p.add_(), destroying the SGLD posterior
guarantee.  The fix: maintain a float32 "master copy" of each parameter,
compute all updates in fp32, then copy back to the model's dtype.
This matches standard AMP practice (Adam in fp32 with bf16 params).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from bif.config import SGLDConfig
from bif.training.loss import per_example_causal_lm_loss


@dataclass
class SamplerDiagnostics:
    loss: float
    grad_norm: float
    noise_norm: float
    scaled_grad_norm: float
    unscaled_grad_norm: float
    localization_norm: float
    weight_decay_norm: float
    prior_norm: float
    distance: float
    dot_grad_prior: float
    dot_grad_noise: float
    dot_prior_noise: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "noise_norm": self.noise_norm,
            "scaled_grad_norm": self.scaled_grad_norm,
            "unscaled_grad_norm": self.unscaled_grad_norm,
            "localization_norm": self.localization_norm,
            "weight_decay_norm": self.weight_decay_norm,
            "prior_norm": self.prior_norm,
            "distance": self.distance,
            "dot_grad_prior": self.dot_grad_prior,
            "dot_grad_noise": self.dot_grad_noise,
            "dot_prior_noise": self.dot_prior_noise,
        }


def _accumulate_gradients(
    model: nn.Module,
    pool_ds,
    train_batch_size: int,
    batch_gen: torch.Generator,
    gradient_accumulation_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    """Accumulate gradients over multiple micro-batches (legacy randperm mode).

    Returns (mean_loss, grad_norm) across all micro-batches.
    """
    from bif.data.dataset import get_batch_by_indices, move_batch_to_device

    model.train()
    model.zero_grad(set_to_none=True)

    total_loss = 0.0
    for _ in range(gradient_accumulation_steps):
        n = len(pool_ds)
        indices = torch.randperm(n, generator=batch_gen)[:train_batch_size].tolist()
        batch = move_batch_to_device(get_batch_by_indices(pool_ds, indices), device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        per_ex_loss = per_example_causal_lm_loss(
            labels=batch["labels"], logits=outputs.logits
        )
        (per_ex_loss.mean() / gradient_accumulation_steps).backward()
        total_loss += per_ex_loss.mean().detach().item() / gradient_accumulation_steps

    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.detach().float().norm().item() ** 2

    return total_loss, grad_norm_sq ** 0.5


def _accumulate_gradients_dataloader(
    model: nn.Module,
    feed,
    gradient_accumulation_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    """Accumulate gradients from a DataLoader feed (aligned with devinterp).

    Draws gradient_accumulation_steps batches from the cycling DataLoader,
    accumulating gradients.  This matches devinterp's sample_single_chain()
    which uses DataLoader(shuffle=True) + itertools.cycle.

    Returns (mean_loss, grad_norm) across all micro-batches.
    """
    from bif.data.dataset import move_batch_to_device

    model.train()
    model.zero_grad(set_to_none=True)

    total_loss = 0.0
    for _ in range(gradient_accumulation_steps):
        batch = move_batch_to_device(next(feed), device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        per_ex_loss = per_example_causal_lm_loss(
            labels=batch["labels"], logits=outputs.logits
        )
        (per_ex_loss.mean() / gradient_accumulation_steps).backward()
        total_loss += per_ex_loss.mean().detach().item() / gradient_accumulation_steps

    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.detach().float().norm().item() ** 2

    return total_loss, grad_norm_sq ** 0.5


class _FP32ParamShadow:
    """Maintain float32 shadow copies of model parameters for precise updates.

    When model params are bf16/fp16, computing updates directly on p.data
    truncates small values (e.g. SGLD noise at lr=5e-6 loses ~78% of
    entries).  Instead, we:
    1. Keep a fp32 shadow copy: shadow[name] = p.data.float()
    2. Compute the update in fp32: shadow += delta_fp32
    3. Copy back: p.data.copy_(shadow[name])
    """

    def __init__(self, params: list[tuple[str, nn.Parameter]]):
        self._data: dict[str, torch.Tensor] = {}
        for name, p in params:
            self._data[name] = p.detach().float().clone()

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._data[name]

    def sync_from_model(self, params: list[tuple[str, nn.Parameter]]) -> None:
        for name, p in params:
            self._data[name].copy_(p.data)

    def sync_to_model(self, params: list[tuple[str, nn.Parameter]]) -> None:
        with torch.no_grad():
            for name, p in params:
                p.copy_(self._data[name])


class LocalizedSGLDSampler:
    def __init__(
        self,
        model: nn.Module,
        anchor_params: dict[str, torch.Tensor],
        config: SGLDConfig,
        source_dataset_size: int,
        effective_batch_size: int = 0,
    ):
        self.model = model
        self.anchor_params = anchor_params
        self.cfg = config
        self.source_dataset_size = source_dataset_size
        self.effective_batch_size = effective_batch_size or source_dataset_size
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.shadow = _FP32ParamShadow(self.params)

    def reset_to_anchor(self) -> None:
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.anchor_params[name])
        self.shadow.sync_from_model(self.params)

    def _compute_nbeta(self) -> float:
        cfg = self.cfg
        if cfg.nbeta >= 0:
            return cfg.nbeta
        if cfg.nbeta_mode == "devinterp":
            bs = max(self.effective_batch_size, 2)
            return bs / math.log(bs)
        return cfg.beta * float(self.source_dataset_size)

    def _sgld_update(
        self,
        step_generator: torch.Generator | None = None,
    ) -> SamplerDiagnostics:
        lr = self.cfg.lr
        gamma = self.cfg.gamma
        nbeta = self._compute_nbeta()
        noise_level = self.cfg.noise_level

        grad_norm_sq = 0.0
        noise_norm_sq = 0.0
        scaled_grad_sq = 0.0
        unscaled_grad_sq = 0.0
        localization_sq = 0.0
        weight_decay_sq = 0.0
        distance_sq = 0.0
        dot_grad_prior = 0.0
        dot_grad_noise = 0.0
        dot_prior_noise = 0.0
        with torch.no_grad():
            for name, p in self.params:
                if p.grad is None:
                    continue

                p_fp32 = self.shadow[name]
                raw_grad = p.grad.detach().float()
                grad_norm_sq += raw_grad.norm().item() ** 2

                loss_grad = raw_grad * nbeta
                distance_vec = p_fp32 - self.anchor_params[name]
                prior_grad = gamma * distance_vec
                localization = 0.5 * lr * prior_grad
                weight_decay = torch.zeros_like(localization)
                if self.cfg.weight_decay != 0.0:
                    prior_grad = prior_grad + self.cfg.weight_decay * p_fp32
                    weight_decay = 0.5 * lr * self.cfg.weight_decay * p_fp32
                grad = loss_grad + prior_grad
                scaled_grad = 0.5 * lr * loss_grad
                prior = localization + weight_decay

                noise = torch.randn(
                    p.shape, device=p.device, dtype=torch.float32,
                    generator=step_generator,
                )
                noise_term = math.sqrt(lr) * noise_level * noise
                delta = -0.5 * lr * grad + noise_term
                p_fp32.add_(delta)
                p.copy_(p_fp32)
                noise_norm_sq += noise_term.norm().item() ** 2
                scaled_grad_sq += scaled_grad.norm().item() ** 2
                unscaled_grad_sq += scaled_grad.norm().item() ** 2
                localization_sq += localization.norm().item() ** 2
                weight_decay_sq += weight_decay.norm().item() ** 2
                distance_sq += distance_vec.norm().item() ** 2
                dot_grad_prior += float((scaled_grad * prior).sum().item())
                dot_grad_noise += float((scaled_grad * noise_term).sum().item())
                dot_prior_noise += float((prior * noise_term).sum().item())

        return SamplerDiagnostics(
            loss=0.0,
            grad_norm=grad_norm_sq ** 0.5,
            noise_norm=noise_norm_sq ** 0.5,
            scaled_grad_norm=scaled_grad_sq ** 0.5,
            unscaled_grad_norm=unscaled_grad_sq ** 0.5,
            localization_norm=localization_sq ** 0.5,
            weight_decay_norm=weight_decay_sq ** 0.5,
            prior_norm=(localization_sq + weight_decay_sq) ** 0.5,
            distance=distance_sq ** 0.5,
            dot_grad_prior=dot_grad_prior,
            dot_grad_noise=dot_grad_noise,
            dot_prior_noise=dot_prior_noise,
        )

    def step(
        self,
        batch: dict[str, torch.Tensor],
        step_generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        per_ex_loss = per_example_causal_lm_loss(
            labels=batch["labels"], logits=outputs.logits
        )
        batch_mean_loss = per_ex_loss.mean()
        batch_mean_loss.backward()

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        diagnostics = self._sgld_update(step_generator)
        diagnostics.loss = float(batch_mean_loss.detach().item())
        return diagnostics.as_dict()

    def step_accumulated(
        self,
        pool_ds,
        train_batch_size: int,
        batch_gen: torch.Generator,
        gradient_accumulation_steps: int,
        device: torch.device,
        step_generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        """SGLD step with gradient accumulation (legacy randperm mode)."""
        total_loss, grad_norm = _accumulate_gradients(
            self.model, pool_ds, train_batch_size,
            batch_gen, gradient_accumulation_steps, device,
        )

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        diagnostics = self._sgld_update(step_generator)
        diagnostics.loss = total_loss
        diagnostics.grad_norm = grad_norm
        return diagnostics.as_dict()

    def step_accumulated_dataloader(
        self,
        pool_ds,
        feed,
        gradient_accumulation_steps: int,
        device: torch.device,
        step_generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        """SGLD step with gradient accumulation from DataLoader feed (devinterp-aligned)."""
        total_loss, grad_norm = _accumulate_gradients_dataloader(
            self.model, feed, gradient_accumulation_steps, device,
        )

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        diagnostics = self._sgld_update(step_generator)
        diagnostics.loss = total_loss
        diagnostics.grad_norm = grad_norm
        return diagnostics.as_dict()


class RMSpropSGLDSampler:
    def __init__(
        self,
        model: nn.Module,
        anchor_params: dict[str, torch.Tensor],
        config: SGLDConfig,
        source_dataset_size: int,
        effective_batch_size: int = 0,
    ):
        self.model = model
        self.anchor_params = anchor_params
        self.cfg = config
        self.source_dataset_size = source_dataset_size
        self.effective_batch_size = effective_batch_size or source_dataset_size
        self.params: list[tuple[str, nn.Parameter]] = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        if not self.params:
            raise ValueError("No trainable parameters for SGLD updates")

        self.alpha = config.rmsprop_alpha
        self.eps = config.rmsprop_eps
        self.square_avg: dict[str, torch.Tensor] = {}
        for name, p in self.params:
            self.square_avg[name] = torch.zeros(
                p.shape, dtype=torch.float32, device=p.device,
            )
        self.shadow = _FP32ParamShadow(self.params)

    def reset_to_anchor(self) -> None:
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(self.anchor_params[name])
        for name in self.square_avg:
            self.square_avg[name].zero_()
        self.shadow.sync_from_model(self.params)

    def _compute_nbeta(self) -> float:
        cfg = self.cfg
        if cfg.nbeta >= 0:
            return cfg.nbeta
        if cfg.nbeta_mode == "devinterp":
            bs = max(self.effective_batch_size, 2)
            return bs / math.log(bs)
        return cfg.beta * float(self.source_dataset_size)

    def _rmsprop_update(
        self,
        step_generator: torch.Generator | None = None,
    ) -> SamplerDiagnostics:
        lr = self.cfg.lr
        gamma = self.cfg.gamma
        nbeta = self._compute_nbeta()
        noise_level = self.cfg.noise_level

        grad_norm_sq = 0.0
        noise_norm_sq = 0.0
        scaled_grad_sq = 0.0
        unscaled_grad_sq = 0.0
        localization_sq = 0.0
        weight_decay_sq = 0.0
        distance_sq = 0.0
        dot_grad_prior = 0.0
        dot_grad_noise = 0.0
        dot_prior_noise = 0.0
        with torch.no_grad():
            for name, p in self.params:
                if p.grad is None:
                    continue

                raw_grad = p.grad.detach().float()
                grad_norm_sq += raw_grad.norm().item() ** 2

                self.square_avg[name].mul_(self.alpha).addcmul_(
                    raw_grad, raw_grad, value=1.0 - self.alpha
                )

                preconditioner = 1.0 / (torch.sqrt(self.square_avg[name]) + self.eps)

                loss_step = preconditioner * (raw_grad * nbeta)
                scaled_grad = 0.5 * lr * loss_step
                unscaled_grad = 0.5 * lr * (raw_grad * nbeta)

                p_fp32 = self.shadow[name]
                distance_vec = p_fp32 - self.anchor_params[name]
                prior_grad = gamma * distance_vec
                localization = 0.5 * lr * (preconditioner * (gamma * distance_vec))
                weight_decay = torch.zeros_like(localization)
                if self.cfg.weight_decay != 0.0:
                    prior_grad = prior_grad + self.cfg.weight_decay * p_fp32
                    weight_decay = 0.5 * lr * (preconditioner * (self.cfg.weight_decay * p_fp32))
                prior_step = preconditioner * prior_grad
                prior = localization + weight_decay

                deterministic_update = -0.5 * lr * (loss_step + prior_step)

                noise = torch.randn(
                    p.shape, device=p.device, dtype=torch.float32,
                    generator=step_generator,
                )
                noise_term = torch.sqrt(lr * preconditioner) * noise_level * noise

                p_fp32.add_(deterministic_update + noise_term)
                p.copy_(p_fp32)
                noise_norm_sq += noise_term.norm().item() ** 2
                scaled_grad_sq += scaled_grad.norm().item() ** 2
                unscaled_grad_sq += unscaled_grad.norm().item() ** 2
                localization_sq += localization.norm().item() ** 2
                weight_decay_sq += weight_decay.norm().item() ** 2
                distance_sq += distance_vec.norm().item() ** 2
                dot_grad_prior += float((scaled_grad * prior).sum().item())
                dot_grad_noise += float((scaled_grad * noise_term).sum().item())
                dot_prior_noise += float((prior * noise_term).sum().item())

        return SamplerDiagnostics(
            loss=0.0,
            grad_norm=grad_norm_sq ** 0.5,
            noise_norm=noise_norm_sq ** 0.5,
            scaled_grad_norm=scaled_grad_sq ** 0.5,
            unscaled_grad_norm=unscaled_grad_sq ** 0.5,
            localization_norm=localization_sq ** 0.5,
            weight_decay_norm=weight_decay_sq ** 0.5,
            prior_norm=(localization_sq + weight_decay_sq) ** 0.5,
            distance=distance_sq ** 0.5,
            dot_grad_prior=dot_grad_prior,
            dot_grad_noise=dot_grad_noise,
            dot_prior_noise=dot_prior_noise,
        )

    def step(
        self,
        batch: dict[str, torch.Tensor],
        step_generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        per_ex_loss = per_example_causal_lm_loss(
            labels=batch["labels"], logits=outputs.logits
        )
        batch_mean_loss = per_ex_loss.mean()
        batch_mean_loss.backward()

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        diagnostics = self._rmsprop_update(step_generator)
        diagnostics.loss = float(batch_mean_loss.detach().item())
        return diagnostics.as_dict()

    def step_accumulated(
        self,
        pool_ds,
        train_batch_size: int,
        batch_gen: torch.Generator,
        gradient_accumulation_steps: int,
        device: torch.device,
        step_generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        """RMSprop-SGLD step with gradient accumulation (legacy randperm mode)."""
        total_loss, _ = _accumulate_gradients(
            self.model, pool_ds, train_batch_size,
            batch_gen, gradient_accumulation_steps, device,
        )

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        diagnostics = self._rmsprop_update(step_generator)
        diagnostics.loss = total_loss
        return diagnostics.as_dict()

    def step_accumulated_dataloader(
        self,
        pool_ds,
        feed,
        gradient_accumulation_steps: int,
        device: torch.device,
        step_generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        """RMSprop-SGLD step with gradient accumulation from DataLoader feed (devinterp-aligned)."""
        total_loss, _ = _accumulate_gradients_dataloader(
            self.model, feed, gradient_accumulation_steps, device,
        )

        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in self.params], self.cfg.grad_clip
            )

        diagnostics = self._rmsprop_update(step_generator)
        diagnostics.loss = total_loss
        return diagnostics.as_dict()


def create_sampler(
    model: nn.Module,
    anchor_params: dict[str, torch.Tensor],
    config: SGLDConfig,
    source_dataset_size: int,
    effective_batch_size: int = 0,
) -> LocalizedSGLDSampler | RMSpropSGLDSampler:
    """Factory function to create the appropriate sampler based on config."""
    if config.sampler_type == "rmsprop_sgld":
        return RMSpropSGLDSampler(model, anchor_params, config, source_dataset_size, effective_batch_size)
    return LocalizedSGLDSampler(model, anchor_params, config, source_dataset_size, effective_batch_size)
