"""BIF trace runner: SGLD-based influence function trace collection.

Aligned with devinterp's sampling.py architecture:
- Separate gradient dataset (for SGLD steps) from observables (for loss evaluation)
- Multiple SGLD steps between draws (num_steps_bw_draws) for better mixing
- Efficient observable evaluation: fixed input_ids, batch forward pass per draw
- Output per-token loss traces (shape: chain × draw × sample × token_pos)
  for proper BIF correlation computation

Key difference from old design:
  OLD: at every draw, scan the ENTIRE pool + query datasets → O(N) forward passes per draw
  NEW: at every draw, evaluate a fixed subset of observable samples in a few batches
       → O(batches_per_draw) forward passes per draw
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bif.config import SGLDConfig
from bif.data.dataset import (
    JsonlSequenceDataset,
    get_batch_by_indices,
    move_batch_to_device,
)
from bif.io import ensure_dir, save_json
from bif.training.loss import per_example_causal_lm_loss, per_token_causal_lm_loss
from bif.training.sgld import LocalizedSGLDSampler, RMSpropSGLDSampler, create_sampler
from bif.utils.logging import get_logger
from bif.utils.naming import (
    fmt_ckpt_short,
    fmt_lr,
    make_bif_name,
    make_bif_pipeline_name,
    resolve_model_tag,
)
from bif.utils.naming import (
    guess_model_tag as _guess_model_tag,
)
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import init_run
from bif.utils.tracker import log as swan_log
from bif.utils.tracker import log_line

logger = get_logger("bif.runner")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _sampling_effective_batch_size(
    train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    return max(1, train_batch_size * max(1, gradient_accumulation_steps))


def _observable_sample_budget(
    eval_batch_size: int,
    batches_per_draw: int,
) -> int:
    if batches_per_draw <= 0:
        return 0
    return max(1, eval_batch_size * batches_per_draw)


def _resolve_observable_max_samples(
    eval_batch_size: int,
    batches_per_draw: int,
    explicit_limit: int = 0,
) -> int:
    if explicit_limit > 0:
        return explicit_limit
    return _observable_sample_budget(eval_batch_size, batches_per_draw)


def _compute_nbeta_value(
    sgld_cfg: SGLDConfig,
    source_dataset_size: int,
    sampling_effective_batch_size: int,
) -> float:
    if sgld_cfg.nbeta >= 0:
        return sgld_cfg.nbeta
    if sgld_cfg.nbeta_mode == "devinterp":
        bs = max(sampling_effective_batch_size, 2)
        return bs / math.log(bs)
    return sgld_cfg.beta * float(source_dataset_size)


def _broadcast_plan(
    plan: list[tuple[str, str]], rank: int, world_size: int
) -> list[tuple[str, str]]:
    if world_size <= 1:
        return plan

    import pickle

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if rank == 0:
        data = pickle.dumps(plan)
        size_t = torch.tensor([len(data)], dtype=torch.long, device=device)
    else:
        size_t = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(size_t, src=0)
    n_bytes = int(size_t.item())

    buf = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    if rank == 0:
        buf[:] = torch.frombuffer(data, dtype=torch.uint8).to(device)
    dist.broadcast(buf, src=0)

    if rank != 0:
        plan = pickle.loads(bytes(buf.cpu().tolist()))
    return plan


def _is_checkpoint_complete(
    out_dir: str, expected_world_size: int, draws_per_chain: int, num_chains: int
) -> bool:
    ddp_complete = True
    for rank in range(expected_world_size):
        manifest_path = os.path.join(out_dir, f"manifest_rank{rank:03d}.json")
        if not os.path.isfile(manifest_path):
            ddp_complete = False
            break

    chain_complete = True
    for c in range(num_chains):
        manifest_path = os.path.join(out_dir, f"manifest_chain{c:03d}.json")
        if not os.path.isfile(manifest_path):
            chain_complete = False
            break

    if not ddp_complete and not chain_complete:
        return False

    total_draws = 0
    names = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    for name in names:
        chain_dir = os.path.join(out_dir, name)
        if not (os.path.isdir(chain_dir) and re.fullmatch(r"chain_\d+", name)):
            continue
        trace = os.path.join(chain_dir, "observable_loss_trace.npz")
        if not os.path.isfile(trace):
            trace_jsonl = os.path.join(chain_dir, "pool_loss_trace.jsonl")
            if not os.path.isfile(trace_jsonl):
                return False
            with open(trace_jsonl) as f:
                lines = sum(1 for _ in f)
            if lines < draws_per_chain:
                return False
            total_draws += lines
        else:
            data = np.load(trace)
            n_draws = data.get("seq_loss", data.get("pool_seq_loss")).shape[0]
            if n_draws < draws_per_chain:
                return False
            total_draws += n_draws

    expected_draws = num_chains * draws_per_chain
    return total_draws >= expected_draws


def _discover_checkpoint_plan(
    model_root: str,
    base_model_path: str | None = None,
    include_final_model: bool = True,
    resume_out_dir: str | None = None,
    world_size: int = 1,
    draws_per_chain: int = 60,
    num_chains: int = 4,
    checkpoint_filter: list[str] | None = None,
) -> list[tuple[str, str]]:
    plan: list[tuple[str, str]] = []
    if base_model_path is not None:
        if not os.path.isdir(base_model_path):
            raise FileNotFoundError(base_model_path)
        plan.append(("base_model", base_model_path))

    entries = []
    for name in os.listdir(model_root):
        full = os.path.join(model_root, name)
        if os.path.isdir(full) and re.fullmatch(r"checkpoint-\d+", name):
            entries.append((int(name.split("-")[-1]), name, full))
    for _, name, full in sorted(entries):
        plan.append((name, full))

    final_path = os.path.join(model_root, "final_model")
    if include_final_model and os.path.isdir(final_path):
        plan.append(("final_model", final_path))

    if checkpoint_filter:
        plan = [(n, p) for n, p in plan if n in checkpoint_filter]

    if not plan:
        raise ValueError(f"No checkpoints under {model_root}")

    if resume_out_dir is None:
        return plan

    remaining = []
    for ckpt_name, ckpt_path in plan:
        ckpt_out = os.path.join(resume_out_dir, ckpt_name)
        if _is_checkpoint_complete(ckpt_out, world_size, draws_per_chain, num_chains):
            logger.info("Skipping completed checkpoint: %s", ckpt_name)
        else:
            remaining.append((ckpt_name, ckpt_path))

    if not remaining:
        logger.info("All checkpoints already complete — nothing to do.")
    return remaining


class Observable:
    """Evaluates a fixed set of sequences at each SGLD draw.

    Aligned with devinterp's Observable class:
    - On construction, loads fixed input_ids (same sequences every draw)
    - At each draw, compute_loss(model) returns per-token losses

    Attributes:
        name: Observable identifier (e.g. "pool", "query")
        input_ids: Fixed input_ids tensor, shape (n_samples, seq_len)
        attention_mask: Fixed attention_mask tensor, shape (n_samples, seq_len)
        n_samples: Total number of samples to evaluate per draw
        context_length: Number of predicted positions (seq_len - 1)
    """

    def __init__(
        self,
        name: str,
        dataset: JsonlSequenceDataset,
        eval_batch_size: int,
        device: torch.device,
        max_samples: int = 0,
        seed: int = 1337,
    ):
        self.name = name
        self.eval_batch_size = eval_batch_size
        self.device = device

        n_total = len(dataset)
        if max_samples > 0 and max_samples < n_total:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(seed)
            indices = torch.randperm(n_total, generator=rng)[:max_samples].sort()[0].tolist()
        else:
            indices = list(range(n_total))

        self.n_samples = len(indices)
        self.sample_ids: list[Any] = []
        self.source_types: list[Any] = []
        self.subtypes: list[Any] = []
        self.task_types: list[Any] = []

        all_input_ids = []
        all_attention_mask = []

        for start in range(0, self.n_samples, eval_batch_size):
            batch_indices = indices[start:start + eval_batch_size]
            batch = get_batch_by_indices(dataset, batch_indices)
            self.sample_ids.extend(batch["sample_ids"])
            self.source_types.extend(batch["source_types"])
            self.subtypes.extend(batch["subtypes"])
            self.task_types.extend(batch["task_types"])
            all_input_ids.append(batch["input_ids"])
            all_attention_mask.append(batch["attention_mask"])

        self.input_ids = torch.cat(all_input_ids, dim=0).to(device)
        self.attention_mask = torch.cat(all_attention_mask, dim=0).to(device)
        self.seq_len = self.input_ids.shape[1]
        self.context_length = self.seq_len - 1

    def compute_loss(self, model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute losses at the current model parameters.

        Args:
            model: The model (should be in eval mode, no grad).

        Returns:
            (seq_loss, token_loss) where:
            - seq_loss: shape (n_samples,), per-example mean loss
            - token_loss: shape (n_samples, context_length), per-token loss
        """
        all_token_losses = []
        all_seq_losses = []

        with torch.no_grad():
            for start in range(0, self.n_samples, self.eval_batch_size):
                end = min(start + self.eval_batch_size, self.n_samples)
                ids = self.input_ids[start:end]
                mask = self.attention_mask[start:end]

                outputs = model(input_ids=ids, attention_mask=mask)

                token_loss = per_token_causal_lm_loss(
                    input_ids=ids,
                    logits=outputs.logits,
                )

                labels = ids.clone()
                labels[mask == 0] = -100
                seq_loss = per_example_causal_lm_loss(
                    labels=labels,
                    logits=outputs.logits,
                )

                all_token_losses.append(token_loss.cpu())
                all_seq_losses.append(seq_loss.cpu())

        token_loss = torch.cat(all_token_losses, dim=0)
        seq_loss = torch.cat(all_seq_losses, dim=0)
        return seq_loss, token_loss


def _save_traces_npz(
    chain_dir: str,
    chain_id: int,
    seq_losses: list[torch.Tensor],
    token_losses: list[torch.Tensor],
    observable: Observable,
) -> None:
    """Save trace data as compressed numpy arrays.

    Stores:
    - seq_loss: shape (num_draws, n_samples) — per-example sequence-level loss
    - token_loss: shape (num_draws, n_samples, context_length) — per-token loss
    - sample_ids, source_types, etc. as JSON
    """
    seq_arr = torch.stack(seq_losses, dim=0).float().numpy()
    token_arr = torch.stack(token_losses, dim=0).float().numpy()

    np.savez_compressed(
        os.path.join(chain_dir, "observable_loss_trace.npz"),
        seq_loss=seq_arr,
        token_loss=token_arr,
    )

    meta = {
        "chain_id": chain_id,
        "num_draws": len(seq_losses),
        "n_samples": observable.n_samples,
        "context_length": observable.context_length,
        "sample_ids": [str(s) for s in observable.sample_ids],
        "source_types": [str(s) if s is not None else None for s in observable.source_types],
        "subtypes": [str(s) if s is not None else None for s in observable.subtypes],
        "task_types": [str(s) if s is not None else None for s in observable.task_types],
    }
    save_json(os.path.join(chain_dir, "observable_meta.json"), meta)


def _save_traces_legacy_jsonl(
    chain_dir: str,
    chain_id: int,
    seq_losses: list[torch.Tensor],
    source_types: list[Any],
    subtypes: list[Any],
    task_types: list[Any],
    sample_ids: list[Any],
    dataset_name: str,
) -> None:
    """Save traces in legacy JSONL format for backward compatibility."""
    out_path = os.path.join(chain_dir, f"{dataset_name}_loss_trace.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for draw_idx, loss_tensor in enumerate(seq_losses):
            row = {
                "chain_id": chain_id,
                "draw_in_chain": draw_idx,
                "global_draw": chain_id * len(seq_losses) + draw_idx,
                "dataset": dataset_name,
                "sample_ids": [str(s) for s in sample_ids],
                "dataset_indices": list(range(len(sample_ids))),
                "source_types": source_types,
                "subtypes": subtypes,
                "task_types": task_types,
                "losses": [float(x) for x in loss_tensor.tolist()],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_chain_loss_traces(
    out_dir: str, chain_ids: list[int]
) -> tuple[dict[int, list[float]], dict[int, list[float]], int]:
    all_pool: dict[int, list[float]] = {}
    all_query: dict[int, list[float]] = {}
    max_total_draws = 0

    for cid in chain_ids:
        chain_dir = os.path.join(out_dir, f"chain_{cid:03d}")

        npz_path = os.path.join(chain_dir, "observable_loss_trace.npz")
        if not os.path.isfile(npz_path):
            continue
        data = np.load(npz_path)
        post_pool = data["seq_loss"].mean(axis=1).tolist()

        burnin_pool: list[float] = []
        burnin_query: list[float] = []
        burnin_npz = os.path.join(chain_dir, "burnin_loss_trace.npz")
        if os.path.isfile(burnin_npz):
            bdata = np.load(burnin_npz)
            burnin_pool = bdata["pool_loss_mean"].tolist()
            burnin_query = bdata["query_loss_mean"].tolist()

        all_pool[cid] = burnin_pool + post_pool
        all_query[cid] = burnin_query

        query_npz = os.path.join(chain_dir, "query_loss_trace.npz")
        if os.path.isfile(query_npz):
            qdata = np.load(query_npz)
            post_query = qdata["seq_loss"].mean(axis=1).tolist()
            all_query[cid] = burnin_query + post_query

        max_total_draws = max(max_total_draws, len(all_pool[cid]))

    return all_pool, all_query, max_total_draws


def _log_all_chains_overlay(
    out_dir: str,
    chain_ids: list[int],
    num_burnin_draws: int,
) -> None:
    """Log multi-chain overlay as Line charts with per-chain colors (includes burnin).

    Reads saved .npz traces (post-burnin) and burnin_loss_trace.npz, then
    reconstructs the full draw-by-draw series.  Uses log_line() so all chains
    appear on the same chart with distinct colours.
    """
    all_pool, all_query, max_total_draws = _read_chain_loss_traces(
        out_dir, chain_ids
    )

    if len(all_pool) < 2:
        return

    xaxis = [str(i) for i in range(max_total_draws)]

    pool_series: dict[str, list] = {}
    query_series: dict[str, list] = {}
    for cid in chain_ids:
        pool_list = all_pool.get(cid)
        if pool_list is not None:
            padded = pool_list + [None] * (max_total_draws - len(pool_list))
            pool_series[f"chain{cid}"] = padded
        q_list = all_query.get(cid)
        if q_list is not None:
            padded = q_list + [None] * (max_total_draws - len(q_list))
            query_series[f"chain{cid}"] = padded

    log_line("4_1_bif_overlay/pool_loss", xaxis, pool_series, smooth=True)
    log_line("4_1_bif_overlay/query_loss", xaxis, query_series, smooth=True)


def _log_training_summary_charts(out_dir: str, chain_ids: list[int]) -> None:
    """Log per-draw training metrics as overlay Line charts with per-chain colors.

    Reads draw_metrics.npz from each chain and creates one Line chart per metric
    with all chains overlaid.
    """
    metrics_keys = [
        ("pool_loss_mean", "4_1_bif_summary/pool_loss"),
        ("query_loss_mean", "4_1_bif_summary/query_loss"),
        ("grad_norm_mean", "4_1_bif_summary/grad_norm"),
        ("scaled_grad_norm_mean", "4_1_bif_summary/scaled_grad"),
        ("noise_norm_mean", "4_1_bif_summary/noise_norm"),
        ("localization_norm_mean", "4_1_bif_summary/localization_norm"),
        ("weight_decay_norm_mean", "4_1_bif_summary/weight_decay_norm"),
        ("prior_norm_mean", "4_1_bif_summary/prior_norm"),
        ("snr_mean", "4_1_bif_summary/snr"),
        ("step_loss_mean", "4_1_bif_summary/step_loss"),
        ("step_distance_mean", "4_1_bif_summary/step_distance"),
        ("param_dist", "4_1_bif_summary/param_dist"),
    ]

    per_metric: dict[str, dict[str, list]] = {k: {} for _, k in metrics_keys}
    max_draws = 0

    for cid in chain_ids:
        npz_path = os.path.join(out_dir, f"chain_{cid:03d}", "draw_metrics.npz")
        if not os.path.isfile(npz_path):
            continue
        data = np.load(npz_path)
        n = len(data["pool_loss_mean"])
        max_draws = max(max_draws, n)
        for arr_key, _ in metrics_keys:
            if arr_key not in data:
                continue
            per_metric[_][f"chain{cid}"] = data[arr_key].tolist()

    if max_draws == 0:
        return

    xaxis = [str(i) for i in range(max_draws)]

    for _, chart_key in metrics_keys:
        series = per_metric[chart_key]
        if not series:
            continue
        for cid_str in series:
            vals = series[cid_str]
            series[cid_str] = vals + [None] * (max_draws - len(vals))
        log_line(chart_key, xaxis, series, smooth=True)


def run_bif(
    model_name_or_path: str,
    pool_jsonl: str,
    query_jsonl: str,
    out_dir: str,
    sgld_cfg: SGLDConfig | None = None,
    tokenizer_path: str | None = None,
    max_length: int = 256,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    pool_eval_subset: int = 0,
    device: str | None = None,
    dtype: str = "float32",
    pool_text_key: str = "text",
    pool_id_key: str = "id",
    pool_source_type_key: str = "source",
    pool_subtype_key: str = "subtype",
    query_text_key: str = "text",
    query_id_key: str = "id",
    query_source_type_key: str = "source",
    query_subtype_key: str = "subtype",
    query_task_type_key: str = "task_type",
    experiment_name: str | None = None,
    run_name: str | None = None,
    manage_tracking: bool = True,
    chain_id: int | None = None,
    model_tag: str | None = None,
) -> None:
    if sgld_cfg is None:
        sgld_cfg = SGLDConfig()

    single_chain_mode = chain_id is not None
    if single_chain_mode:
        rank = 0
        world_size = 1
        local_rank = 0
    else:
        rank, world_size, local_rank = _get_distributed_context()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    _set_seed(sgld_cfg.seed + (chain_id if single_chain_mode else rank))

    ckpt_name = os.path.basename(model_name_or_path)
    tag = resolve_model_tag(model_tag, model_name_or_path)
    auto_exp_name = make_bif_name(
        tag, ckpt_name, sgld_cfg.lr, sgld_cfg.gamma,
        sgld_cfg.draws_per_chain, sgld_cfg.num_burnin_steps,
    )
    if manage_tracking and rank == 0:
        init_run(
            experiment_name=experiment_name or auto_exp_name,
            run_name=run_name,
            config={
                "checkpoint": ckpt_name,
                "model": model_name_or_path,
                "sgld": asdict(sgld_cfg),
                "max_length": max_length,
                "chain_id": chain_id,
                "model_tag": tag,
            },
            tags=["bif", ckpt_name, tag],
        )

    if device is None:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    ensure_dir(out_dir)

    logger.info("Loading tokenizer and model (rank=%d)", rank)
    tok_src = tokenizer_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pt_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=pt_dtype if device.type == "cuda" else torch.float32,
    )
    model.to(device)

    pool_ds = JsonlSequenceDataset(
        pool_jsonl,
        tokenizer,
        max_length=max_length,
        text_key=pool_text_key,
        id_key=pool_id_key,
        source_type_key=pool_source_type_key,
        subtype_key=pool_subtype_key,
    )
    query_ds = JsonlSequenceDataset(
        query_jsonl,
        tokenizer,
        max_length=max_length,
        text_key=query_text_key,
        id_key=query_id_key,
        source_type_key=query_source_type_key,
        subtype_key=query_subtype_key,
        task_type_key=query_task_type_key,
    )

    sampling_effective_batch_size = _sampling_effective_batch_size(
        train_batch_size,
        sgld_cfg.gradient_accumulation_steps,
    )
    observable_budget = _observable_sample_budget(
        eval_batch_size,
        sgld_cfg.batches_per_draw,
    )
    computed_nbeta = _compute_nbeta_value(
        sgld_cfg,
        source_dataset_size=len(pool_ds),
        sampling_effective_batch_size=sampling_effective_batch_size,
    )

    if rank == 0:
        save_json(
            f"{out_dir}/run_config.json",
            {
                "model_name_or_path": model_name_or_path,
                "max_length": max_length,
                "pool_jsonl": pool_jsonl,
                "query_jsonl": query_jsonl,
                "nbeta_mode": sgld_cfg.nbeta_mode,
                "nbeta": sgld_cfg.nbeta,
                "nbeta_computed": computed_nbeta,
                "sampling_effective_batch_size": sampling_effective_batch_size,
                "observable_sample_budget": observable_budget,
                "sgld_config": asdict(sgld_cfg),
            },
        )

    anchor_params = {
        name: p.detach().clone().to(device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    sampler = create_sampler(
        model,
        anchor_params,
        sgld_cfg,
        source_dataset_size=len(pool_ds),
        effective_batch_size=sampling_effective_batch_size,
    )

    obs_seed = sgld_cfg.seed + 1337
    pool_max = _resolve_observable_max_samples(
        eval_batch_size,
        sgld_cfg.batches_per_draw,
        explicit_limit=pool_eval_subset,
    )
    query_max = _resolve_observable_max_samples(
        eval_batch_size,
        sgld_cfg.batches_per_draw,
    )
    pool_obs = Observable(
        name="pool",
        dataset=pool_ds,
        eval_batch_size=eval_batch_size,
        device=device,
        max_samples=pool_max,
        seed=obs_seed,
    )
    query_obs = Observable(
        name="query",
        dataset=query_ds,
        eval_batch_size=eval_batch_size,
        device=device,
        max_samples=query_max,
        seed=obs_seed + 1,
    )

    if single_chain_mode:
        assigned_chains = [chain_id]
    else:
        assigned_chains = list(range(rank, sgld_cfg.num_chains, world_size))
    if not assigned_chains:
        logger.info("No chains assigned to rank %d", rank)
        if not single_chain_mode:
            save_json(
                f"{out_dir}/manifest_rank{rank:03d}.json",
                {"rank": rank, "assigned_chains": []},
            )
            _barrier()
        return

    for chain_id in assigned_chains:
        logger.info("Starting chain %d on rank %d", chain_id, rank)
        sampler.reset_to_anchor()

        dataloader_rng = torch.Generator(device="cpu")
        dataloader_rng.manual_seed(sgld_cfg.seed + chain_id)
        noise_gen = None
        if device.type == "cuda":
            noise_gen = torch.Generator(device=device)
            noise_gen.manual_seed(sgld_cfg.seed + chain_id)

        from itertools import cycle as _cycle

        from torch.utils.data import DataLoader as _DataLoader

        from bif.data.dataset import collate_bif_batch as _collate

        loader = _DataLoader(
            pool_ds,
            batch_size=train_batch_size,
            shuffle=True,
            generator=dataloader_rng,
            drop_last=True,
            collate_fn=_collate,
        )
        feed = _cycle(loader)

        chain_dir = f"{out_dir}/chain_{chain_id:03d}"
        ensure_dir(chain_dir)

        pool_seq_losses: list[torch.Tensor] = []
        pool_token_losses: list[torch.Tensor] = []
        query_seq_losses: list[torch.Tensor] = []
        query_token_losses: list[torch.Tensor] = []

        burnin_pool_means: list[float] = []
        burnin_query_means: list[float] = []

        total_steps = sgld_cfg.num_burnin_steps + sgld_cfg.draws_per_chain * sgld_cfg.num_steps_bw_draws
        num_burnin_draws = sgld_cfg.num_burnin_steps // sgld_cfg.num_steps_bw_draws
        draw_count = 0
        burnin_draw_count = 0
        grad_accum = sgld_cfg.gradient_accumulation_steps
        steps_since_draw = sgld_cfg.num_steps_bw_draws

        draw_grad_norms: list[float] = []
        draw_scaled_grad_norms: list[float] = []
        draw_noise_norms: list[float] = []
        draw_localization_norms: list[float] = []
        draw_weight_decay_norms: list[float] = []
        draw_prior_norms: list[float] = []
        draw_step_distances: list[float] = []
        draw_step_losses: list[float] = []
        draw_snrs: list[float] = []

        all_draw_pool_means: list[float] = []
        all_draw_query_means: list[float] = []
        all_draw_grad_norm: list[float] = []
        all_draw_scaled_grad_norm: list[float] = []
        all_draw_noise_norm: list[float] = []
        all_draw_localization_norm: list[float] = []
        all_draw_weight_decay_norm: list[float] = []
        all_draw_prior_norm: list[float] = []
        all_draw_snr: list[float] = []
        all_draw_step_loss: list[float] = []
        all_draw_step_distance: list[float] = []
        all_draw_param_dist: list[float] = []
        all_draw_is_burnin: list[int] = []

        def _sgld_draw_summary(chain_id: int, step: int) -> dict[str, float]:
            if not draw_grad_norms:
                return {}
            gn = np.array(draw_grad_norms)
            sgn = np.array(draw_scaled_grad_norms)
            nn = np.array(draw_noise_norms)
            ln = np.array(draw_localization_norms)
            wd = np.array(draw_weight_decay_norms)
            pn = np.array(draw_prior_norms)
            dn = np.array(draw_step_distances)
            sl = np.array(draw_step_losses)
            sr = np.array(draw_snrs)
            return {
                f"4_1_bif_sgld/chain{chain_id}/draw/grad_norm_mean": float(gn.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/grad_norm_max": float(gn.max()),
                f"4_1_bif_sgld/chain{chain_id}/draw/scaled_grad_norm_mean": float(sgn.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/noise_norm_mean": float(nn.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/noise_norm_max": float(nn.max()),
                f"4_1_bif_sgld/chain{chain_id}/draw/localization_norm_mean": float(ln.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/weight_decay_norm_mean": float(wd.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/prior_norm_mean": float(pn.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/distance_mean": float(dn.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/step_loss_mean": float(sl.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/snr_mean": float(sr.mean()),
                f"4_1_bif_sgld/chain{chain_id}/draw/snr_min": float(sr.min()),
                f"4_1_bif_sgld/chain{chain_id}/draw/num_steps": len(draw_grad_norms),
                f"4_1_bif_sgld/chain{chain_id}/draw/actual_sgld_step": step,
            }

        def _append_draw_aggregates(param_dist: float, is_burnin_draw: int) -> None:
            all_draw_pool_means.append(float(pool_seq.mean()))
            all_draw_query_means.append(float(query_seq.mean()))
            if draw_grad_norms:
                all_draw_grad_norm.append(float(np.mean(draw_grad_norms)))
                all_draw_scaled_grad_norm.append(float(np.mean(draw_scaled_grad_norms)))
                all_draw_noise_norm.append(float(np.mean(draw_noise_norms)))
                all_draw_localization_norm.append(float(np.mean(draw_localization_norms)))
                all_draw_weight_decay_norm.append(float(np.mean(draw_weight_decay_norms)))
                all_draw_prior_norm.append(float(np.mean(draw_prior_norms)))
                all_draw_step_loss.append(float(np.mean(draw_step_losses)))
                all_draw_step_distance.append(float(np.mean(draw_step_distances)))
                all_draw_snr.append(float(np.mean(draw_snrs)))
            else:
                all_draw_grad_norm.append(0.0)
                all_draw_scaled_grad_norm.append(0.0)
                all_draw_noise_norm.append(0.0)
                all_draw_localization_norm.append(0.0)
                all_draw_weight_decay_norm.append(0.0)
                all_draw_prior_norm.append(0.0)
                all_draw_step_loss.append(0.0)
                all_draw_step_distance.append(0.0)
                all_draw_snr.append(0.0)
            all_draw_param_dist.append(param_dist)
            all_draw_is_burnin.append(is_burnin_draw)

        def _clear_draw_buffers() -> None:
            draw_grad_norms.clear()
            draw_scaled_grad_norms.clear()
            draw_noise_norms.clear()
            draw_localization_norms.clear()
            draw_weight_decay_norms.clear()
            draw_prior_norms.clear()
            draw_step_distances.clear()
            draw_step_losses.clear()
            draw_snrs.clear()

        pbar = tqdm(range(total_steps), desc=f"Chain {chain_id}", disable=(rank != 0))
        for step in pbar:
            is_burnin = step < sgld_cfg.num_burnin_steps

            if grad_accum > 1:
                step_info = sampler.step_accumulated_dataloader(
                    pool_ds, feed, grad_accum, device, step_generator=noise_gen,
                )
            else:
                batch = move_batch_to_device(next(feed), device)
                step_info = sampler.step(batch, step_generator=noise_gen)

            steps_since_draw += 1

            draw_grad_norms.append(float(step_info["grad_norm"]))
            draw_scaled_grad_norms.append(float(step_info["scaled_grad_norm"]))
            draw_noise_norms.append(float(step_info["noise_norm"]))
            draw_localization_norms.append(float(step_info["localization_norm"]))
            draw_weight_decay_norms.append(float(step_info["weight_decay_norm"]))
            draw_prior_norms.append(float(step_info["prior_norm"]))
            draw_step_distances.append(float(step_info["distance"]))
            draw_step_losses.append(float(step_info["loss"]))
            draw_snrs.append(
                float(step_info["scaled_grad_norm"])
                / (float(step_info["noise_norm"]) + 1e-12)
            )

            is_draw_step = (
                not is_burnin
                and steps_since_draw >= sgld_cfg.num_steps_bw_draws
            )
            is_burnin_draw_step = (
                is_burnin
                and steps_since_draw >= sgld_cfg.num_steps_bw_draws
            )

            if is_burnin_draw_step:
                steps_since_draw = 0
                model.eval()
                pool_seq, pool_tok = pool_obs.compute_loss(model)
                query_seq, query_tok = query_obs.compute_loss(model)

                if torch.isnan(pool_seq).any() or torch.isnan(query_seq).any():
                    logger.warning(
                        "NaN detected at burn-in draw %d (step %d, chain %d). "
                        "SGLD diverged — stopping this chain early.",
                        burnin_draw_count, step, chain_id if single_chain_mode else rank,
                    )
                    break

                if rank == 0:
                    obs_data = {
                        f"4_1_bif_loss/chain{chain_id}/pool_loss_mean": float(pool_seq.mean()),
                        f"4_1_bif_loss/chain{chain_id}/query_loss_mean": float(query_seq.mean()),
                        f"4_1_bif_loss/chain{chain_id}/is_burnin": 1,
                    }
                    obs_data.update(_sgld_draw_summary(chain_id, step))
                    swan_log(obs_data, step=burnin_draw_count)
                burnin_pool_means.append(float(pool_seq.mean()))
                burnin_query_means.append(float(query_seq.mean()))
                burnin_draw_count += 1

                _append_draw_aggregates(
                    param_dist=float(step_info["distance"]),
                    is_burnin_draw=1,
                )
                _clear_draw_buffers()

            if is_draw_step:
                steps_since_draw = 0
                model.eval()
                pool_seq, pool_tok = pool_obs.compute_loss(model)
                query_seq, query_tok = query_obs.compute_loss(model)

                if torch.isnan(pool_seq).any() or torch.isnan(query_seq).any():
                    logger.warning(
                        "NaN detected at draw %d (step %d, chain %d). "
                        "SGLD diverged — stopping this chain early.",
                        draw_count, step, chain_id if single_chain_mode else rank,
                    )
                    break

                pool_seq_losses.append(pool_seq)
                pool_token_losses.append(pool_tok)
                query_seq_losses.append(query_seq)
                query_token_losses.append(query_tok)

                draw_count += 1
                pbar.set_postfix(
                    pool_mean=f"{pool_seq.mean():.4f}",
                    query_mean=f"{query_seq.mean():.4f}",
                    draw=draw_count,
                )

                if rank == 0:
                    draw_idx = num_burnin_draws + draw_count - 1
                    with torch.no_grad():
                        param_dist_sq = sum(
                            (p.data - anchor_params[n]).float().norm().item() ** 2
                            for n, p in sampler.params
                        )
                        param_dist = param_dist_sq ** 0.5
                    obs_data = {
                        f"4_1_bif_loss/chain{chain_id}/pool_loss_mean": float(pool_seq.mean()),
                        f"4_1_bif_loss/chain{chain_id}/query_loss_mean": float(query_seq.mean()),
                        f"4_1_bif_loss/chain{chain_id}/param_dist_from_anchor": param_dist,
                        f"4_1_bif_loss/chain{chain_id}/is_burnin": 0,
                    }
                    obs_data.update(_sgld_draw_summary(chain_id, step))
                    swan_log(obs_data, step=draw_idx)
                else:
                    with torch.no_grad():
                        param_dist_sq = sum(
                            (p.data - anchor_params[n]).float().norm().item() ** 2
                            for n, p in sampler.params
                        )
                        param_dist = param_dist_sq ** 0.5

                _append_draw_aggregates(param_dist=param_dist, is_burnin_draw=0)
                _clear_draw_buffers()

            if rank == 0:
                swan_log(
                    {
                        f"4_1_bif_sgld_step/chain{chain_id}/loss": step_info["loss"],
                        f"4_1_bif_sgld_step/chain{chain_id}/grad_norm": step_info["grad_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/scaled_grad_norm": step_info["scaled_grad_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/unscaled_grad_norm": step_info["unscaled_grad_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/noise_norm": step_info["noise_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/localization_norm": step_info["localization_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/weight_decay_norm": step_info["weight_decay_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/prior_norm": step_info["prior_norm"],
                        f"4_1_bif_sgld_step/chain{chain_id}/distance": step_info["distance"],
                        f"4_1_bif_sgld_step/chain{chain_id}/dot_grad_prior": step_info["dot_grad_prior"],
                        f"4_1_bif_sgld_step/chain{chain_id}/dot_grad_noise": step_info["dot_grad_noise"],
                        f"4_1_bif_sgld_step/chain{chain_id}/dot_prior_noise": step_info["dot_prior_noise"],
                        f"4_1_bif_sgld_step/chain{chain_id}/signal_noise_ratio": step_info["scaled_grad_norm"] / (step_info["noise_norm"] + 1e-12),
                    },
                    step=step,
                )

        _save_traces_npz(chain_dir, chain_id, pool_seq_losses, pool_token_losses, pool_obs)

        if burnin_pool_means:
            np.savez_compressed(
                os.path.join(chain_dir, "burnin_loss_trace.npz"),
                pool_loss_mean=np.array(burnin_pool_means),
                query_loss_mean=np.array(burnin_query_means),
            )

        if all_draw_pool_means:
            np.savez_compressed(
                os.path.join(chain_dir, "draw_metrics.npz"),
                pool_loss_mean=np.array(all_draw_pool_means),
                query_loss_mean=np.array(all_draw_query_means),
                grad_norm_mean=np.array(all_draw_grad_norm),
                scaled_grad_norm_mean=np.array(all_draw_scaled_grad_norm),
                noise_norm_mean=np.array(all_draw_noise_norm),
                localization_norm_mean=np.array(all_draw_localization_norm),
                weight_decay_norm_mean=np.array(all_draw_weight_decay_norm),
                prior_norm_mean=np.array(all_draw_prior_norm),
                step_loss_mean=np.array(all_draw_step_loss),
                step_distance_mean=np.array(all_draw_step_distance),
                snr_mean=np.array(all_draw_snr),
                param_dist=np.array(all_draw_param_dist),
                is_burnin=np.array(all_draw_is_burnin),
            )

        query_meta = {
            "chain_id": chain_id,
            "num_draws": len(query_seq_losses),
            "n_samples": query_obs.n_samples,
            "context_length": query_obs.context_length,
            "sample_ids": [str(s) for s in query_obs.sample_ids],
            "source_types": [str(s) if s is not None else None for s in query_obs.source_types],
            "subtypes": [str(s) if s is not None else None for s in query_obs.subtypes],
            "task_types": [str(s) if s is not None else None for s in query_obs.task_types],
        }
        query_seq_arr = torch.stack(query_seq_losses, dim=0).float().numpy()
        query_token_arr = torch.stack(query_token_losses, dim=0).float().numpy()
        np.savez_compressed(
            os.path.join(chain_dir, "query_loss_trace.npz"),
            seq_loss=query_seq_arr,
            token_loss=query_token_arr,
        )
        save_json(os.path.join(chain_dir, "query_meta.json"), query_meta)

        _save_traces_legacy_jsonl(
            chain_dir, chain_id, pool_seq_losses,
            pool_obs.source_types, pool_obs.subtypes, pool_obs.task_types,
            pool_obs.sample_ids, "pool",
        )
        _save_traces_legacy_jsonl(
            chain_dir, chain_id, query_seq_losses,
            query_obs.source_types, query_obs.subtypes, query_obs.task_types,
            query_obs.sample_ids, "query",
        )

        save_json(
            f"{chain_dir}/chain_config.json",
            {
                "chain_id": chain_id,
                "draws_written": draw_count,
                "sgld_config": asdict(sgld_cfg),
            },
        )

    if single_chain_mode:
        save_json(
            f"{out_dir}/manifest_chain{chain_id:03d}.json",
            {"chain_id": chain_id},
        )
    else:
        save_json(
            f"{out_dir}/manifest_rank{rank:03d}.json",
            {"rank": rank, "assigned_chains": assigned_chains},
        )
    logger.info("All chains completed on rank %d", rank)
    if not single_chain_mode:
        _barrier()

    if rank == 0 and len(assigned_chains) > 1:
        num_burnin_draws = sgld_cfg.num_burnin_steps // max(1, sgld_cfg.num_steps_bw_draws)
        _log_all_chains_overlay(out_dir, assigned_chains, num_burnin_draws)
        _log_training_summary_charts(out_dir, assigned_chains)

    if manage_tracking and rank == 0:
        swan_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIF trace collection.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help=(
            "Path to tokenizer files. HF Trainer does not copy tokenizer files "
            "into intermediate checkpoint dirs, so pass the base model path here "
            "when running --run_all_checkpoints."
        ),
    )
    parser.add_argument("--run_all_checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated checkpoint names to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints whose output directory already has complete traces.",
    )
    parser.add_argument("--pool_jsonl", required=True)
    parser.add_argument("--query_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--pool_eval_subset", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--nbeta_mode", type=str, default="devinterp", choices=["devinterp", "dataset"])
    parser.add_argument("--nbeta", type=float, default=0.0)
    parser.add_argument("--noise_level", type=float, default=1.0)
    parser.add_argument("--num_chains", type=int, default=4)
    parser.add_argument("--draws_per_chain", type=int, default=60)
    parser.add_argument("--num_burnin_steps", type=int, default=0)
    parser.add_argument("--num_steps_bw_draws", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--sampler_type",
        default="sgld",
        choices=["sgld", "rmsprop_sgld"],
    )
    parser.add_argument("--rmsprop_alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop_eps", type=float, default=1e-1)
    parser.add_argument("--batches_per_draw", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--chain_id",
        type=int,
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    if args.run_all_checkpoints and not args.model_root:
        raise ValueError("--model_root is required with --run_all_checkpoints")
    if not args.run_all_checkpoints and not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required")

    rank, world_size, _ = _get_distributed_context()
    single_chain_mode = args.chain_id is not None
    if not single_chain_mode and world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    cfg = SGLDConfig(
        lr=args.lr,
        gamma=args.gamma,
        beta=args.beta,
        nbeta_mode=args.nbeta_mode,
        nbeta=args.nbeta,
        noise_level=args.noise_level,
        num_chains=args.num_chains,
        draws_per_chain=args.draws_per_chain,
        num_burnin_steps=args.num_burnin_steps,
        num_steps_bw_draws=args.num_steps_bw_draws,
        seed=args.seed,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        sampler_type=args.sampler_type,
        rmsprop_alpha=args.rmsprop_alpha,
        rmsprop_eps=args.rmsprop_eps,
        batches_per_draw=args.batches_per_draw,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    try:
        if args.run_all_checkpoints:
            rank, world_size, _ = _get_distributed_context()
            ckpt_filter = None
            if args.checkpoints:
                ckpt_filter = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
            if single_chain_mode:
                plan = _discover_checkpoint_plan(
                    args.model_root,
                    args.base_model_path,
                    resume_out_dir=args.out_dir if args.resume else None,
                    world_size=1,
                    draws_per_chain=cfg.draws_per_chain,
                    num_chains=cfg.num_chains,
                    checkpoint_filter=ckpt_filter,
                )
            else:
                if rank == 0:
                    plan = _discover_checkpoint_plan(
                        args.model_root,
                        args.base_model_path,
                        resume_out_dir=args.out_dir if args.resume else None,
                        world_size=world_size,
                        draws_per_chain=cfg.draws_per_chain,
                        num_chains=cfg.num_chains,
                        checkpoint_filter=ckpt_filter,
                    )
                else:
                    plan = []
                plan = _broadcast_plan(plan, rank, world_size)
            if not plan:
                logger.info("Nothing left to run.")
                return

            tokenizer_path = args.tokenizer_path
            if tokenizer_path is None:
                final_model = os.path.join(args.model_root, "final_model")
                if os.path.isdir(final_model) and os.path.exists(
                    os.path.join(final_model, "tokenizer_config.json")
                ):
                    tokenizer_path = final_model
                elif args.base_model_path and os.path.isdir(args.base_model_path):
                    tokenizer_path = args.base_model_path

            draws_per_ckpt = cfg.num_chains * cfg.draws_per_chain
            if args.experiment_name:
                run_label = args.experiment_name
            else:
                tag = resolve_model_tag(args.model_tag, args.model_root or "")
                run_label = make_bif_pipeline_name(
                    tag, cfg.lr, cfg.gamma, cfg.draws_per_chain, cfg.num_burnin_steps,
                )
            if rank == 0:
                ckpt_names = [name for name, _ in plan]
                init_run(
                    experiment_name=run_label,
                    run_name=args.run_name,
                    config={
                        "checkpoints": ckpt_names,
                        "resume": args.resume,
                        "sgld": asdict(cfg),
                        "max_length": args.max_length,
                    },
                    tags=["bif", "pipeline"] + (["resume"] if args.resume else []),
                )
            for ckpt_idx, (ckpt_name, ckpt_path) in enumerate(plan):
                logger.info("Checkpoint: %s", ckpt_name)
                run_bif(
                    model_name_or_path=ckpt_path,
                    tokenizer_path=tokenizer_path,
                    pool_jsonl=args.pool_jsonl,
                    query_jsonl=args.query_jsonl,
                    out_dir=f"{args.out_dir}/{ckpt_name}",
                    sgld_cfg=cfg,
                    max_length=args.max_length,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    pool_eval_subset=args.pool_eval_subset,
                    device=args.device,
                    dtype=args.dtype,
                    manage_tracking=False,
                    chain_id=args.chain_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if rank == 0:
                swan_finish()
        else:
            run_bif(
                model_name_or_path=args.model_name_or_path,
                tokenizer_path=args.tokenizer_path,
                pool_jsonl=args.pool_jsonl,
                query_jsonl=args.query_jsonl,
                out_dir=args.out_dir,
                sgld_cfg=cfg,
                max_length=args.max_length,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                pool_eval_subset=args.pool_eval_subset,
                device=args.device,
                dtype=args.dtype,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                manage_tracking=True,
                chain_id=args.chain_id,
                model_tag=args.model_tag,
            )
    finally:
        if not single_chain_mode and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
