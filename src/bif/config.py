"""Configuration dataclasses for BIF.

Aligned with devinterp's SamplerConfig conventions:
- noise_level (not noise_scale) matches devinterp's SGMCMC.noise_level
- num_steps_bw_draws replaces thinning for explicit step-between-draws control
- num_burnin_steps (not burn_in) matches devinterp naming
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SGLDConfig:
    """Configuration for Localized SGLD sampler.

    Parameter mapping to devinterp:
        lr              ← SamplerConfig.lr
        noise_level     ← SamplerConfig.noise_level (= SGMCMC.noise_level, default 1.0)
        nbeta_mode      ← "devinterp" uses batch_size/log(batch_size), "dataset" uses beta*N
        nbeta           ← explicit override; if >0, takes priority over nbeta_mode
        gamma           ← SamplerConfig.localization  (= GaussianPrior.localization)
        num_steps_bw_draws ← SamplerConfig.num_steps_bw_draws (steps between draws)
        num_burnin_steps   ← SamplerConfig.num_burnin_steps
        weight_decay    ← SamplerConfig.llc_weight_decay (= GaussianPrior.localization, center=None)
        batches_per_draw   ← sample(batches_per_draw=3) — fixed observable mini-batches evaluated per draw
        gradient_accumulation_steps ← sample(gradient_accumulation_steps=1)
    """

    lr: float = 5e-6
    gamma: float = 1e-3
    beta: float = 1.0
    nbeta: float = -1.0
    nbeta_mode: str = "devinterp"
    noise_level: float = 1.0
    num_chains: int = 4
    draws_per_chain: int = 60
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1
    seed: int = 42
    grad_clip: float | None = None
    weight_decay: float = 0.0
    sampler_type: str = "sgld"
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-1
    batches_per_draw: int = 0
    gradient_accumulation_steps: int = 1

    @property
    def total_sampling_steps(self) -> int:
        return self.num_burnin_steps + self.draws_per_chain * self.num_steps_bw_draws

    @property
    def thinning(self) -> int:
        return self.num_steps_bw_draws


@dataclass
class ReplayTrainConfig:
    """Configuration for replay-aware CPT training."""

    schedule: str = "mixed"
    replay_mode: str = "selected"
    replay_ratio: float = 0.2
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 0
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    deepspeed: str | None = None
    fsdp: str = ""
    fsdp_transformer_layer_cls_to_wrap: str | None = None
    seed: int = 42
