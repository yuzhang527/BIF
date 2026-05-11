"""Analyze BIF traces: compute influence scores and generate plots.

Aligned with devinterp's compute_bif() approach:
- BIF = pairwise Pearson correlation between loss traces across sequences
  within the same observable (pool×pool), computed over the chain_draw axis
- Both token-level and sequence-level BIF are supported
- Chain reduction: stack (recommended) or mean across chains

Supports single-process and multi-GPU (torchrun) execution.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from bif.io import ensure_dir, read_jsonl, save_json
from bif.utils.naming import guess_model_tag, make_analyze_name
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import (
    init_run,
    log_bar,
    log_boxplot,
    log_heatmap,
    log_line,
    log_scatter,
    log_table,
)
from bif.utils.tracker import log as swan_log


@dataclass
class AnalyzeConfig:
    """All configurable knobs for BIF analysis.

    Visualization limit fields accept ``None`` to mean "auto-adapt from data".
    When set to ``None``, ``_auto_adapt_config`` will compute a sensible value
    after loading the traces.  Explicit values are always respected.
    """

    score_col: str = "bif_mean"
    top_k: int | None = None
    negate_scores: bool = False
    save_full_query_matrix: bool = False
    enable_aux_query_plots: bool = False

    hist_bins: int | None = None
    scatter_max_points: int | None = None
    heatmap_max_pool: int | None = None
    heatmap_max_query: int | None = None
    rhat_max_samples: int | None = None
    eigenvalue_max_pool: int | None = None
    eigenvalue_max_ev: int | None = None
    boxplot_max_sources: int | None = None
    boxplot_min_per_source: int | None = None
    rhat_min_draws: int | None = None
    chain_scatter_min_draws: int | None = None
    trajectory_top_n: int | None = None
    source_label_max_len: int | None = None
    heatmap_topk_max: int | None = None
    convergence_checkpoints: list[int] | None = None
    convergence_min_draws: int | None = None


def _auto_adapt_config(
    acfg: AnalyzeConfig,
    pool_size: int,
    query_size: int,
    num_draws: int,
    num_chains: int,
    num_sources: int = 0,
) -> AnalyzeConfig:
    """Fill ``None`` fields in *acfg* based on actual data dimensions.

    Returns the same object (mutated in-place) for convenience.
    """
    dpc = num_draws // max(num_chains, 1)

    if acfg.top_k is None:
        acfg.top_k = min(500, pool_size)

    if acfg.hist_bins is None:
        acfg.hist_bins = min(60, max(20, pool_size // 10))

    if acfg.scatter_max_points is None:
        acfg.scatter_max_points = min(500, pool_size)

    if acfg.heatmap_max_pool is None:
        acfg.heatmap_max_pool = min(50, pool_size)

    if acfg.heatmap_max_query is None:
        acfg.heatmap_max_query = min(20, query_size)

    if acfg.rhat_max_samples is None:
        acfg.rhat_max_samples = min(100, pool_size)

    if acfg.eigenvalue_max_pool is None:
        acfg.eigenvalue_max_pool = 1500

    if acfg.eigenvalue_max_ev is None:
        acfg.eigenvalue_max_ev = min(30, pool_size)

    if acfg.boxplot_max_sources is None:
        acfg.boxplot_max_sources = max(num_sources, 30)

    if acfg.boxplot_min_per_source is None:
        acfg.boxplot_min_per_source = max(3, pool_size // 200)

    if acfg.rhat_min_draws is None:
        acfg.rhat_min_draws = min(10, max(3, dpc // 3))

    if acfg.chain_scatter_min_draws is None:
        acfg.chain_scatter_min_draws = max(3, dpc // 5)

    if acfg.trajectory_top_n is None:
        acfg.trajectory_top_n = min(20, pool_size)

    if acfg.source_label_max_len is None:
        acfg.source_label_max_len = 25

    if acfg.heatmap_topk_max is None:
        acfg.heatmap_topk_max = min(50, pool_size)

    if acfg.convergence_checkpoints is None:
        base = [3, 5, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500]
        acfg.convergence_checkpoints = [c for c in base if c <= num_draws]

    if acfg.convergence_min_draws is None:
        acfg.convergence_min_draws = max(3, num_draws // 20)

    return acfg


def _get_dist_context() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _init_dist_if_needed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _checkpoint_sort_key(name: str) -> tuple[int, str]:
    if name == "base_model":
        return (-1, name)
    if name == "final_model":
        return (10**9, name)
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (10**8, name)


def discover_checkpoint_dirs(root: str) -> list[tuple[str, str]]:
    entries = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and (
            name in ("base_model", "final_model")
            or re.fullmatch(r"checkpoint-\d+", name)
        ):
            entries.append((name, full))
    if not entries:
        has_chains = any(
            os.path.isdir(os.path.join(root, d))
            and re.fullmatch(r"chain_\d+", d)
            for d in os.listdir(root)
        )
        if has_chains:
            entries = [("final_model", root)]
        else:
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if not os.path.isdir(full):
                    continue
                has_chains_sub = any(
                    os.path.isdir(os.path.join(full, d))
                    and re.fullmatch(r"chain_\d+", d)
                    for d in os.listdir(full)
                )
                if has_chains_sub:
                    entries.append((name, full))
    entries.sort(key=lambda x: _checkpoint_sort_key(x[0]))
    if not entries:
        raise ValueError(
            f"No checkpoint dirs under {root}\n"
            f"Expected: dirs named base_model/final_model/checkpoint-N, "
            f"or dirs containing chain_*/ subdirs, or chain_*/ at root level."
        )
    return entries


def _discover_chain_dirs(checkpoint_dir: str) -> list[str]:
    out = []
    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"chain_\d+", name):
            out.append(full)
    out.sort()
    if not out:
        raise ValueError(f"No chain dirs under {checkpoint_dir}")
    return out


def load_checkpoint_traces(checkpoint_dir: str) -> dict[str, Any]:
    """Load loss traces from a checkpoint directory.

    Supports both new (.npz) and legacy (.jsonl) formats.

    Returns dict with:
        pool_ids, pool_seq_loss, pool_token_loss, pool_meta,
        query_ids, query_seq_loss, query_token_loss, query_meta,
        num_draws
    """
    chain_dirs = _discover_chain_dirs(checkpoint_dir)

    npz_path = os.path.join(chain_dirs[0], "observable_loss_trace.npz")
    if os.path.isfile(npz_path):
        return _load_traces_npz(chain_dirs)

    return _load_traces_legacy(chain_dirs)


def _load_traces_npz(chain_dirs: list[str]) -> dict[str, Any]:
    """Load from new .npz format."""
    pool_seq_parts = []
    pool_token_parts = []
    query_seq_parts = []
    query_token_parts = []
    pool_meta = None
    query_meta = None

    for cdir in chain_dirs:
        pool_npz = np.load(os.path.join(cdir, "observable_loss_trace.npz"))
        pool_seq_parts.append(pool_npz["seq_loss"])
        pool_token_parts.append(pool_npz["token_loss"])

        query_npz = np.load(os.path.join(cdir, "query_loss_trace.npz"))
        query_seq_parts.append(query_npz["seq_loss"])
        query_token_parts.append(query_npz["token_loss"])

        if pool_meta is None:
            import json as _json
            with open(os.path.join(cdir, "observable_meta.json")) as f:
                pool_meta = _json.load(f)
            with open(os.path.join(cdir, "query_meta.json")) as f:
                query_meta = _json.load(f)

    pool_seq = np.concatenate(pool_seq_parts, axis=0)
    pool_token = np.concatenate(pool_token_parts, axis=0)
    query_seq = np.concatenate(query_seq_parts, axis=0)
    query_token = np.concatenate(query_token_parts, axis=0)

    pool_nan_frac = float(np.isnan(pool_seq).mean())
    query_nan_frac = float(np.isnan(query_seq).mean())
    if pool_nan_frac > 0.5 or query_nan_frac > 0.5:
        raise ValueError(
            f"Trace data is mostly NaN (pool={pool_nan_frac:.1%}, query={query_nan_frac:.1%}). "
            f"SGLD likely diverged — decrease lr and/or increase gamma."
        )

    num_chains = len(chain_dirs)
    draws_per_chain = pool_seq.shape[0] // num_chains

    for meta in (pool_meta, query_meta):
        if "source_type" not in meta and "source_types" in meta:
            meta["source_type"] = meta["source_types"]
        if "task_type" not in meta and "task_types" in meta:
            meta["task_type"] = meta["task_types"]

    return {
        "pool_ids": pool_meta["sample_ids"],
        "pool_seq_loss": pool_seq,
        "pool_token_loss": pool_token,
        "pool_meta": pool_meta,
        "query_ids": query_meta["sample_ids"],
        "query_seq_loss": query_seq,
        "query_token_loss": query_token,
        "query_meta": query_meta,
        "num_draws": pool_seq.shape[0],
        "num_chains": num_chains,
        "draws_per_chain": draws_per_chain,
    }


def _load_traces_legacy(chain_dirs: list[str]) -> dict[str, Any]:
    """Load from legacy .jsonl format."""
    pool_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []

    for cdir in chain_dirs:
        pool_path = os.path.join(cdir, "pool_loss_trace.jsonl")
        query_path = os.path.join(cdir, "query_loss_trace.jsonl")
        if os.path.isfile(pool_path):
            pool_rows.extend(read_jsonl(pool_path))
        if os.path.isfile(query_path):
            query_rows.extend(read_jsonl(query_path))

    pool_ids, pool_mat, pool_meta = rows_to_loss_matrix(pool_rows, "pool")
    query_ids, query_mat, query_meta = rows_to_loss_matrix(query_rows, "query")
    pool_mat, query_mat = _align_by_draw_key(pool_mat, pool_meta, query_mat, query_meta)

    num_draws = pool_mat.shape[0]

    draw_meta = pool_meta.get("draw_meta", [])
    if draw_meta:
        num_chains = len(set(d["chain_id"] for d in draw_meta))
        draws_per_chain = num_draws // num_chains if num_chains > 0 else num_draws
    else:
        num_chains = 1
        draws_per_chain = num_draws

    pool_token_loss = pool_mat[:, :, np.newaxis]
    query_token_loss = query_mat[:, :, np.newaxis]

    return {
        "pool_ids": pool_ids,
        "pool_seq_loss": pool_mat,
        "pool_token_loss": pool_token_loss,
        "pool_meta": pool_meta,
        "query_ids": query_ids,
        "query_seq_loss": query_mat,
        "query_token_loss": query_token_loss,
        "query_meta": query_meta,
        "num_draws": num_draws,
        "num_chains": num_chains,
        "draws_per_chain": draws_per_chain,
    }


def rows_to_loss_matrix(
    rows: list[dict[str, Any]], dataset_name: str
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    rows = [r for r in rows if r.get("dataset") == dataset_name]
    if not rows:
        raise ValueError(f"No rows for dataset={dataset_name}")
    rows.sort(
        key=lambda r: (
            int(r["chain_id"]),
            int(r["draw_in_chain"]),
        )
    )

    template = None
    for r in rows:
        ids = r.get("sample_ids", [])
        losses = r.get("losses", [])
        if (
            isinstance(ids, list)
            and isinstance(losses, list)
            and ids
            and len(ids) == len(losses)
        ):
            template = r
            break
    if template is None:
        raise ValueError(f"No valid rows for dataset={dataset_name}")

    sample_ids = list(template["sample_ids"])
    n = len(sample_ids)
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    template_id_set = set(sample_ids)

    first_src = template.get("source_types", [None] * n)
    first_sub = template.get("subtypes", [None] * n)
    first_task = template.get("task_types", [None] * n)

    valid_rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for r in rows:
        ids = r.get("sample_ids", [])
        losses = r.get("losses", [])
        if not isinstance(ids, list) or not isinstance(losses, list):
            dropped.append({"reason": "not_list", "chain_id": r.get("chain_id")})
            continue
        if not ids or not losses:
            dropped.append({"reason": "empty", "chain_id": r.get("chain_id")})
            continue
        if len(ids) != len(losses):
            dropped.append({"reason": "length_mismatch", "chain_id": r.get("chain_id")})
            continue
        if set(ids) != template_id_set:
            dropped.append({"reason": "id_set_mismatch", "chain_id": r.get("chain_id")})
            continue
        if len(set(ids)) != len(ids):
            dropped.append({"reason": "duplicate_ids", "chain_id": r.get("chain_id")})
            continue
        valid_rows.append(r)

    if not valid_rows:
        raise ValueError(f"All rows dropped for dataset={dataset_name}")

    mat = np.full((len(valid_rows), n), np.nan, dtype=np.float64)
    draw_meta = []

    for draw_idx, r in enumerate(valid_rows):
        for sid, loss in zip(r["sample_ids"], r["losses"]):
            mat[draw_idx, id_to_idx[sid]] = float(loss)
        draw_meta.append(
            {
                "chain_id": int(r["chain_id"]),
                "draw_in_chain": int(r["draw_in_chain"]),
                "global_draw": int(r["global_draw"]),
            }
        )

    good_mask = ~np.isnan(mat).any(axis=1)
    if not np.all(good_mask):
        mat = mat[good_mask]
        draw_meta = [dm for dm, g in zip(draw_meta, good_mask) if g]

    if mat.shape[0] == 0:
        raise ValueError(f"All rows invalid for dataset={dataset_name}")

    meta: dict[str, Any] = {
        "source_type": list(first_src),
        "subtype": list(first_sub),
        "task_type": list(first_task),
        "draw_meta": draw_meta,
        "num_rows_valid": int(mat.shape[0]),
        "num_rows_dropped": len(dropped),
        "dropped_rows": dropped[:200],
    }
    return sample_ids, mat, meta


def _align_by_draw_key(
    pool_mat: np.ndarray,
    pool_meta: dict[str, Any],
    query_mat: np.ndarray,
    query_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pool_keys = [(d["chain_id"], d["draw_in_chain"]) for d in pool_meta["draw_meta"]]
    query_keys = [(d["chain_id"], d["draw_in_chain"]) for d in query_meta["draw_meta"]]
    common = sorted(set(pool_keys) & set(query_keys))
    if not common:
        raise ValueError("No common draws between pool and query")
    pool_idx = {k: i for i, k in enumerate(pool_keys)}
    query_idx = {k: i for i, k in enumerate(query_keys)}
    pi = [pool_idx[k] for k in common]
    qi = [query_idx[k] for k in common]
    return pool_mat[pi], query_mat[qi]


def _offdiag_values(mat: np.ndarray) -> np.ndarray:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Expected a square matrix")
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


# ─── BIF Computation (aligned with devinterp) ──────────────────────────────


def compute_bif_pairwise(
    seq_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
) -> np.ndarray:
    """Compute pairwise BIF correlation matrix (aligned with devinterp compute_bif).

    This computes the Pearson correlation between loss traces for all pairs
    of samples within the same observable, across the chain_draw axis.

    This is the correct BIF definition from the paper: BIF(i,j) = corr(L_i, L_j)
    where L_i and L_j are the loss traces of samples i and j across SGLD draws.

    Args:
        seq_loss: Shape (num_draws, n_samples) — sequence-level loss per draw.
            num_draws = num_chains × draws_per_chain when chains are stacked.
        num_chains: Number of chains. Used to reshape for chain reduction.
        reduce_chains: "stack" (recommended) or "mean".

    Returns:
        BIF correlation matrix of shape (n_samples, n_samples).
    """
    if reduce_chains == "stack":
        loss = seq_loss.T  # (n_samples, num_draws)
    elif reduce_chains == "mean":
        draws_per_chain = seq_loss.shape[0] // num_chains
        reshaped = seq_loss.reshape(num_chains, draws_per_chain, -1)
        loss = reshaped.mean(axis=0).T  # (n_samples, draws_per_chain)
    else:
        raise ValueError(f"Unknown reduce_chains: {reduce_chains}")

    loss_t = torch.as_tensor(loss, dtype=torch.float32)
    if torch.cuda.is_available():
        loss_t = loss_t.cuda()

    corr = torch.corrcoef(loss_t)
    return corr.cpu().numpy()


def compute_bif_tokenwise(
    token_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Compute token-level BIF (aligned with devinterp _tokenwise_bif).

    Args:
        token_loss: Shape (num_draws, n_samples, context_length).
        num_chains: Number of chains.
        reduce_chains: "stack" or "mean".
        batch_size: Batch size for block processing.
        device: Torch device.

    Returns:
        Token-level BIF of shape (n_samples, n_samples, context_length, context_length).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if reduce_chains == "stack":
        loss = token_loss.transpose(1, 0, 2)
    elif reduce_chains == "mean":
        draws_per_chain = token_loss.shape[0] // num_chains
        reshaped = token_loss.reshape(num_chains, draws_per_chain, -1, token_loss.shape[2])
        loss = reshaped.mean(axis=0).transpose(1, 0, 2)
    else:
        raise ValueError(f"Unknown reduce_chains: {reduce_chains}")

    n_samples = loss.shape[0]
    n_tokens = loss.shape[2]
    result = np.empty((n_samples, n_tokens, n_samples, n_tokens), dtype=np.float32)

    for i in range(0, n_samples, batch_size):
        for j in range(0, n_samples, batch_size):
            bi = min(batch_size, n_samples - i)
            bj = min(batch_size, n_samples - j)
            block_i = torch.as_tensor(loss[i:i+bi], device=device)
            block_j = torch.as_tensor(loss[j:j+bj], device=device)
            block_corr = _batch_corrcoef_tokenwise(block_i, block_j)
            result[i:i+bi, :, j:j+bj, :] = block_corr.cpu().numpy()
            del block_i, block_j, block_corr

    return result.transpose(0, 2, 1, 3)


def _batch_corrcoef_tokenwise(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Batched token-wise correlation.

    Args:
        a: shape (n_a, series_a, observations) — (batch, tokens, draws)
        b: shape (n_b, series_b, observations)

    Returns:
        shape (n_a, n_b, series_a, series_b) cross-correlation block.
    """
    n_a, series_a, n_obs = a.shape
    n_b, series_b, _ = b.shape

    a_centered = a - a.mean(dim=2, keepdim=True)
    b_centered = b - b.mean(dim=2, keepdim=True)

    a_broadcast = a_centered[:, None, :, :].expand(n_a, n_b, series_a, n_obs)
    b_broadcast = b_centered[None, :, :, :].expand(n_a, n_b, series_b, n_obs)
    combined = torch.cat([a_broadcast, b_broadcast], dim=2)

    cov = combined @ combined.transpose(-1, -2) / (n_obs - 1)

    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    std = torch.sqrt(diag)
    cov /= std.unsqueeze(-1) * std.unsqueeze(-2)

    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    cov *= 1 - eye
    cov += eye

    return cov[:, :, :series_a, series_a:]


def compute_bif_scores(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
    negate_scores: bool = False,
) -> dict[str, np.ndarray]:
    """Compute BIF influence scores.

    Aligned with devinterp's compute_bif() approach:
    - pool_bif_matrix: pairwise BIF within pool (N_pool × N_pool)
    - query_bif_matrix: pairwise BIF within query (N_query × N_query)
    - cross_corr: pool × query cross-correlation (for backward compat)

    The primary score is the mean BIF correlation with other samples,
    which measures how "influential" a sample is in the loss landscape.
    """
    pool_bif_matrix = compute_bif_pairwise(pool_seq_loss, num_chains, reduce_chains)
    query_bif_matrix = compute_bif_pairwise(query_seq_loss, num_chains, reduce_chains)

    np.fill_diagonal(pool_bif_matrix, 0.0)
    np.fill_diagonal(query_bif_matrix, 0.0)

    pool_bif_mean = pool_bif_matrix.mean(axis=1)
    pool_bif_abs_mean = np.abs(pool_bif_matrix).mean(axis=1)

    pool_centered = pool_seq_loss - pool_seq_loss.mean(axis=0, keepdims=True)
    query_centered = query_seq_loss - query_seq_loss.mean(axis=0, keepdims=True)
    cross_cov = (pool_centered.T @ query_centered) / pool_centered.shape[0]

    pool_z = _safe_zscore_cols(pool_seq_loss)
    query_z = _safe_zscore_cols(query_seq_loss)
    cross_corr = (pool_z.T @ query_z) / pool_z.shape[0]

    sign = -1.0 if negate_scores else 1.0

    mean_loss = pool_seq_loss.mean(axis=0)
    self_variance = pool_seq_loss.var(axis=0)

    draw_idx = np.arange(pool_seq_loss.shape[0], dtype=np.float64)
    draw_idx = (draw_idx - draw_idx.mean()) / (draw_idx.std() + 1e-12)
    draw_trend = ((pool_z.T @ draw_idx) / len(draw_idx)).reshape(-1)

    return {
        "bif_mean": sign * pool_bif_mean,
        "bif_abs_mean": pool_bif_abs_mean,
        "bif_matrix": pool_bif_matrix,
        "query_bif_matrix": query_bif_matrix,
        "cross_corr_mean_over_queries": sign * cross_corr.mean(axis=1),
        "cross_corr_absmean_over_queries": sign * np.abs(cross_corr).mean(axis=1),
        "cross_corr_matrix": cross_corr,
        "cross_cov_avg_over_queries": sign * cross_cov.mean(axis=1),
        "mean_loss": mean_loss,
        "self_variance": self_variance,
        "draw_trend": draw_trend,
    }


def _safe_zscore_cols(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (mat - mu) / sd


def average_rank(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    order = np.argsort(-scores if descending else scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks


def spearman_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    ra = average_rank(a, descending=True)
    rb = average_rank(b, descending=True)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    a_top = set(np.argsort(-a)[:k].tolist())
    b_top = set(np.argsort(-b)[:k].tolist())
    return len(a_top & b_top) / float(k)


def build_pool_score_df(
    pool_ids: list[Any],
    pool_meta: dict[str, Any],
    score_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    src_list = pool_meta.get("source_type", [None] * len(pool_ids))
    sub_list = pool_meta.get("subtype", [None] * len(pool_ids))
    task_list = pool_meta.get("task_type", [None] * len(pool_ids))

    if isinstance(src_list, (list, tuple)) and len(src_list) == len(pool_ids):
        pass
    else:
        src_list = [None] * len(pool_ids)

    if isinstance(sub_list, (list, tuple)) and len(sub_list) == len(pool_ids):
        pass
    else:
        sub_list = [None] * len(pool_ids)

    if isinstance(task_list, (list, tuple)) and len(task_list) == len(pool_ids):
        pass
    else:
        task_list = [None] * len(pool_ids)

    df = pd.DataFrame(
        {
            "sample_id": pool_ids,
            "source": src_list,
            "subtype": sub_list,
            "task_type": task_list,
        }
    )
    for k, v in score_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(pool_ids):
            df[k] = v
    return df


def make_global_trajectory_df(
    per_ckpt_df: dict[str, pd.DataFrame], score_col: str
) -> pd.DataFrame:
    names = list(per_ckpt_df.keys())
    base_ids = per_ckpt_df[names[0]]["sample_id"].tolist()
    merged = pd.DataFrame({"sample_id": base_ids})
    source_map = per_ckpt_df[names[0]][
        ["sample_id", "source", "subtype", "task_type"]
    ].copy()
    merged = merged.merge(source_map, on="sample_id", how="left")

    for ck in names:
        cur = per_ckpt_df[ck][["sample_id", score_col]].copy()
        cur = cur.rename(columns={score_col: f"score__{ck}"})
        merged = merged.merge(cur, on="sample_id", how="left")

    score_cols = [f"score__{ck}" for ck in names]
    merged["traj_mean"] = merged[score_cols].mean(axis=1)
    merged["traj_std"] = merged[score_cols].std(axis=1)
    merged["traj_min"] = merged[score_cols].min(axis=1)
    merged["traj_max"] = merged[score_cols].max(axis=1)
    merged["emergence_last_minus_first"] = (
        merged[score_cols[-1]] - merged[score_cols[0]]
    )
    arr = merged[score_cols].to_numpy(dtype=np.float64)
    merged["num_positive_deltas"] = (np.diff(arr, axis=1) > 0).sum(axis=1)
    return merged


def _score_histogram_bars(
    scores: np.ndarray, bins: int = 40
) -> tuple[list[str], list[int]]:
    counts, edges = np.histogram(scores, bins=bins)
    labels = [f"{edges[i]:.3f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()


def _source_shift_series(
    names: list[str], top_dfs: dict[str, pd.DataFrame], source_col: str
) -> dict[str, list[float]]:
    all_sources = sorted(
        {
            s
            for ck in names
            for s in top_dfs[ck][source_col].fillna("unknown").astype(str).unique()
        }
    )
    series: dict[str, list[float]] = {}
    for src in all_sources:
        series[src] = [
            float(
                top_dfs[ck][source_col]
                .fillna("unknown")
                .astype(str)
                .value_counts(normalize=True)
                .get(src, 0.0)
            )
            for ck in names
        ]
    return series


def _safe_standardize(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return []
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-12:
        return [0.0 for _ in arr]
    return ((arr - mu) / sd).tolist()


def _build_source_recommendations(
    names: list[str],
    per_ckpt_df: dict[str, pd.DataFrame],
    per_ckpt_top: dict[str, pd.DataFrame],
    top_k: int,
    score_col: str,
) -> pd.DataFrame:
    sources = sorted(
        {
            str(src)
            for ck in names
            for src in per_ckpt_df[ck]["source"].fillna("unknown").astype(str).unique()
        }
    )
    if not sources:
        return pd.DataFrame()

    rows = []
    last_ck = names[-1]
    first_ck = names[0]
    for src in sources:
        mean_scores = []
        mean_query_overlap = []
        mean_self_variance = []
        top_fracs = []
        for ck in names:
            cur = per_ckpt_df[ck]
            src_df = cur[cur["source"].fillna("unknown").astype(str) == src]
            if src_df.empty:
                mean_scores.append(0.0)
                mean_query_overlap.append(0.0)
                mean_self_variance.append(0.0)
            else:
                mean_scores.append(float(src_df[score_col].mean()))
                mean_query_overlap.append(float(src_df.get("cross_corr_mean_over_queries", pd.Series([0.0])).mean()))
                mean_self_variance.append(float(src_df.get("self_variance", pd.Series([0.0])).mean()))

            top_df = per_ckpt_top[ck]
            top_src = top_df["source"].fillna("unknown").astype(str)
            top_fracs.append(float((top_src == src).mean()) if len(top_src) else 0.0)

        rows.append(
            {
                "source": src,
                "late_bif_mean": mean_scores[-1],
                "late_query_overlap": mean_query_overlap[-1],
                "late_self_variance": mean_self_variance[-1],
                "late_topk_frac": top_fracs[-1],
                "source_shift": top_fracs[-1] - top_fracs[0],
                "score_traj_mean": float(np.mean(mean_scores)),
                "query_traj_mean": float(np.mean(mean_query_overlap)),
                "score_traj_std": float(np.std(mean_scores)),
                "topk_traj_mean": float(np.mean(top_fracs)),
                "topk_first": top_fracs[0],
                "topk_last": top_fracs[-1],
                "n_checkpoints_present": int(sum(v != 0.0 for v in mean_scores)),
                "first_checkpoint": first_ck,
                "last_checkpoint": last_ck,
            }
        )

    rec_df = pd.DataFrame(rows)
    for col in (
        "late_bif_mean",
        "late_query_overlap",
        "late_topk_frac",
        "source_shift",
        "score_traj_mean",
    ):
        rec_df[f"z_{col}"] = _safe_standardize(rec_df[col].tolist())
    rec_df["z_late_self_variance"] = _safe_standardize(rec_df["late_self_variance"].tolist())

    rec_df["bif_training_score"] = (
        rec_df["z_late_bif_mean"]
        + 0.75 * rec_df["z_late_topk_frac"]
        + 0.5 * rec_df["z_source_shift"]
        + 0.5 * rec_df["z_score_traj_mean"]
        - 0.5 * rec_df["z_late_self_variance"]
    )
    rec_df["query_target_score"] = (
        rec_df["z_late_query_overlap"]
        + 0.75 * rec_df["z_late_bif_mean"]
        + 0.5 * rec_df["z_source_shift"]
        - 0.5 * rec_df["z_late_self_variance"]
    )
    rec_df = rec_df.sort_values(
        ["query_target_score", "bif_training_score"],
        ascending=False,
    ).reset_index(drop=True)
    rec_df["recommend_rank"] = np.arange(1, len(rec_df) + 1)
    return rec_df


def _trajectory_stats_series(
    traj_df: pd.DataFrame, names: list[str], sort_by: str, top_n: int
) -> dict[str, list[float]]:
    score_cols = [f"score__{ck}" for ck in names]
    sub = traj_df.sort_values(sort_by, ascending=False).head(top_n)
    arr = sub[score_cols].to_numpy(dtype=np.float64)
    series: dict[str, list[float]] = {
        "p75": [
            round(float(np.percentile(arr[:, j], 75)), 6) for j in range(len(names))
        ],
        "median": [
            round(float(np.percentile(arr[:, j], 50)), 6) for j in range(len(names))
        ],
        "p25": [
            round(float(np.percentile(arr[:, j], 25)), 6) for j in range(len(names))
        ],
        "mean": [round(float(arr[:, j].mean()), 6) for j in range(len(names))],
    }
    return series


def _checkpoint_sort_index(name: str, all_names: list[str]) -> int:
    sorted_names = sorted(all_names, key=_checkpoint_sort_key)
    return sorted_names.index(name) if name in sorted_names else 0


def _log_sample_table(
    traj_df: pd.DataFrame,
    names: list[str],
    score_col: str,
    top_k: int,
    pool_df: pd.DataFrame | None = None,
) -> None:
    n_preview = min(50, len(traj_df))

    text_map = {}
    if pool_df is not None:
        id_col = "id" if "id" in pool_df.columns else "sample_id"
        if id_col in pool_df.columns and "text" in pool_df.columns:
            text_map = dict(zip(pool_df[id_col].astype(str), pool_df["text"]))

    def _fmt_text(t: str, max_len: int = 200) -> str:
        s = str(t).strip().replace("\n", " ").replace("\r", " ")
        return s[:max_len] + "..." if len(s) > max_len else s

    def _build_rows(sub_df: pd.DataFrame) -> list[list[Any]]:
        score_cols = [
            f"score__{ck}" for ck in names if f"score__{ck}" in sub_df.columns
        ]
        rows = []
        for rank_i, (_, row) in enumerate(sub_df.head(n_preview).iterrows(), 1):
            r = [rank_i, str(row.get("source", ""))]
            for sc in score_cols:
                v = row.get(sc)
                r.append(f"{v:.4f}" if pd.notna(v) else "")
            r.append(f"{row.get('traj_mean', 0):.4f}")
            r.append(f"{row.get('emergence_last_minus_first', 0):.4f}")
            sid = str(row.get("sample_id", ""))
            r.append(_fmt_text(text_map.get(sid, "")))
            rows.append(r)
        return rows

    ck_short = [
        ck.replace("checkpoint-", "ck").replace("final_model", "final") for ck in names
    ]
    headers = ["rank", "source"] + ck_short + ["traj_mean", "emergence", "text"]

    top_mean = traj_df.head(n_preview)
    log_table(
        "4_2_influence/samples/top",
        headers=headers,
        rows=_build_rows(top_mean),
    )


def _log_checkpoint_sample_table(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    ck_name: str,
    pool_df: pd.DataFrame | None = None,
    include_aux_query_corr: bool = False,
) -> None:
    n_preview = min(50, len(df))

    text_map: dict[str, str] = {}
    if pool_df is not None:
        id_col = "id" if "id" in pool_df.columns else "sample_id"
        if id_col in pool_df.columns and "text" in pool_df.columns:
            text_map = dict(zip(pool_df[id_col].astype(str), pool_df["text"]))

    def _fmt(t: str) -> str:
        return str(t).strip().replace("\n", " ").replace("\r", " ")

    headers = ["rank", "sample_id", "source", score_col, "bif_abs_mean", "self_variance", "mean_loss"]
    if include_aux_query_corr:
        headers.append("aux_query_corr")
    headers.append("text")
    rows = []
    for rank_i, (_, row) in enumerate(df.head(n_preview).iterrows(), 1):
        sid = str(row.get("sample_id", ""))
        cur = [
            rank_i,
            sid,
            str(row.get("source", "")),
            f"{row.get(score_col, 0):.4f}",
            f"{row.get('bif_abs_mean', 0):.4f}",
            f"{row.get('self_variance', 0):.4f}",
            f"{row.get('mean_loss', 0):.4f}",
        ]
        if include_aux_query_corr:
            cur.append(f"{row.get('cross_corr_mean_over_queries', 0):.4f}")
        cur.append(_fmt(text_map.get(sid, "")))
        rows.append(cur)

    log_table(
        f"4_2_influence/samples/top/{ck_name}",
        headers=headers,
        rows=rows,
    )

    bottom = df.tail(n_preview).iloc[::-1]
    rows_bot = []
    for rank_i, (_, row) in enumerate(bottom.iterrows(), 1):
        sid = str(row.get("sample_id", ""))
        cur = [
            rank_i,
            sid,
            str(row.get("source", "")),
            f"{row.get(score_col, 0):.4f}",
            f"{row.get('bif_abs_mean', 0):.4f}",
            f"{row.get('self_variance', 0):.4f}",
            f"{row.get('mean_loss', 0):.4f}",
        ]
        if include_aux_query_corr:
            cur.append(f"{row.get('cross_corr_mean_over_queries', 0):.4f}")
        cur.append(_fmt(text_map.get(sid, "")))
        rows_bot.append(cur)

    log_table(
        f"4_2_influence/samples/bottom/{ck_name}",
        headers=headers,
        rows=rows_bot,
    )


def _process_one_checkpoint(
    ck_name: str,
    ck_dir: str,
    out_dir: str,
    acfg: AnalyzeConfig,
    ck_step: int = 0,
    pool_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    t0 = __import__("time").monotonic()
    try:
        loaded = load_checkpoint_traces(ck_dir)
    except ValueError as exc:
        print(f"[analyze] Skipping {ck_name}: {exc}")
        return {"checkpoint": ck_name, "error": str(exc)}

    pool_seq = loaded["pool_seq_loss"]
    query_seq = loaded["query_seq_loss"]
    if np.isnan(pool_seq).any() or np.isnan(query_seq).any():
        pool_nan = float(np.isnan(pool_seq).mean())
        query_nan = float(np.isnan(query_seq).mean())
        msg = (
            f"Trace contains NaN (pool={pool_nan:.1%}, query={query_nan:.1%}). "
            f"SGLD diverged — skipping analysis."
        )
        print(f"[analyze] Skipping {ck_name}: {msg}")
        return {"checkpoint": ck_name, "error": msg}

    num_chains = loaded.get("num_chains", 1)
    scores = compute_bif_scores(
        loaded["pool_seq_loss"],
        loaded["query_seq_loss"],
        num_chains=num_chains,
        reduce_chains="stack",
        negate_scores=acfg.negate_scores,
    )
    df = build_pool_score_df(loaded["pool_ids"], loaded["pool_meta"], scores)
    if acfg.score_col not in df.columns:
        raise ValueError(
            f"score_col={acfg.score_col!r} not in {df.columns.tolist()}"
        )
    df = df.sort_values(acfg.score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    ck_out = f"{out_dir}/{ck_name}"
    ensure_dir(ck_out)
    df.to_csv(f"{ck_out}/pool_scores.csv", index=False)
    df.head(acfg.top_k).to_csv(f"{ck_out}/top_{acfg.top_k}.csv", index=False)

    bif_matrix_path = f"{ck_out}/bif_matrix.npy"
    np.save(bif_matrix_path, scores["bif_matrix"])

    if acfg.save_full_query_matrix:
        np.save(
            f"{ck_out}/query_pair_corr_matrix.npy",
            scores["cross_corr_matrix"],
        )

    scores_arr = df[acfg.score_col].to_numpy()
    bif_mat = scores["bif_matrix"]
    pool_mat = loaded["pool_seq_loss"]
    query_mat = loaded["query_seq_loss"]

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        core_summary = _log_core_bif_summary(
            bif_mat,
            pool_mat,
            query_mat,
            num_chains,
            ck_name,
            rhat_max_samples=acfg.rhat_max_samples,
            rhat_min_draws=acfg.rhat_min_draws,
        )
        save_json(
            f"{ck_out}/ckpt_meta.json",
            {
                "checkpoint": ck_name,
                "num_draws": int(loaded["num_draws"]),
                "pool_size": int(loaded["pool_seq_loss"].shape[1]),
                "query_size": int(loaded["query_seq_loss"].shape[1]),
                "num_chains": num_chains,
                **core_summary,
            },
        )
        burnin_offset = _read_num_burnin_draws(ck_dir)
        _log_loss_traces(
            pool_mat, query_mat, num_chains, ck_name,
            draw_offset=burnin_offset,
        )
        _log_score_histogram(scores_arr, ck_name, bins=acfg.hist_bins)
        _log_corr_distribution(
            bif_mat, ck_name,
            bins=acfg.hist_bins,
        )
        _log_score_vs_selfvar_scatter(
            scores_arr,
            pool_mat, ck_name,
            max_points=acfg.scatter_max_points,
        )
        _log_bif_heatmap_topk(
            bif_mat, df, loaded["pool_ids"], acfg.top_k, ck_name,
            score_col=acfg.score_col,
            heatmap_topk_max=acfg.heatmap_topk_max,
        )
        _log_score_by_source(
            df, acfg.score_col, acfg.top_k, ck_name,
            max_sources=acfg.boxplot_max_sources,
            min_per_source=acfg.boxplot_min_per_source,
            label_max_len=acfg.source_label_max_len,
        )
        _log_eigenvalue_spectrum(
            bif_mat, ck_name,
            max_pool=acfg.eigenvalue_max_pool,
            max_ev=acfg.eigenvalue_max_ev,
        )
        _log_convergence(
            pool_mat, num_chains, ck_name,
            checkpoints=acfg.convergence_checkpoints,
            min_draws=acfg.convergence_min_draws,
        )
        if num_chains > 1:
            _log_rhat(
                pool_mat, num_chains, ck_name,
                max_samples=acfg.rhat_max_samples,
                min_draws=acfg.rhat_min_draws,
            )
            _log_chain_scatter(
                pool_mat, num_chains, scores_arr, ck_name,
                max_points=acfg.scatter_max_points,
                min_draws=acfg.chain_scatter_min_draws,
            )

        if acfg.enable_aux_query_plots:
            _log_aux_query_corr_distribution(
                scores["cross_corr_matrix"],
                ck_name,
                bins=acfg.hist_bins,
            )
            _log_score_vs_selfvar_scatter(
                scores["cross_cov_avg_over_queries"],
                pool_mat,
                ck_name,
                max_points=acfg.scatter_max_points,
                chart_key="5_aux_query/cross_cov_vs_selfvar",
                yaxis_name="aux_query_cross_cov",
            )
            _log_cross_cov_heatmap(
                scores["cross_corr_matrix"], loaded["pool_ids"],
                loaded["query_ids"], ck_name,
                pool_sources=df["source"].fillna("unknown").tolist() if "source" in df.columns else None,
                max_pool=acfg.heatmap_max_pool,
                max_query=acfg.heatmap_max_query,
                query_sources=loaded.get("query_meta", {}).get("source_types"),
                query_task_types=loaded.get("query_meta", {}).get("task_types"),
            )

        _log_checkpoint_sample_table(
            df,
            acfg.score_col,
            acfg.top_k,
            ck_name,
            pool_df=pool_df,
            include_aux_query_corr=acfg.enable_aux_query_plots,
        )

    elapsed = __import__("time").monotonic() - t0
    return {
        "checkpoint": ck_name,
        "num_draws": int(loaded["num_draws"]),
        "pool_size": len(loaded["pool_ids"]),
        "query_size": len(loaded.get("query_ids", [])),
        "score_mean": float(df[acfg.score_col].mean()),
        "score_std": float(df[acfg.score_col].std()),
        "analysis_seconds": round(elapsed, 2),
    }


def _read_num_burnin_draws(checkpoint_dir: str) -> int:
    """Read num_burnin_draws from chain_config.json (saved by bif_runner)."""
    for name in os.listdir(checkpoint_dir):
        chain_dir = os.path.join(checkpoint_dir, name)
        if not (os.path.isdir(chain_dir) and re.fullmatch(r"chain_\d+", name)):
            continue
        cfg_path = os.path.join(chain_dir, "chain_config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            import json as _json
            with open(cfg_path) as f:
                cfg = _json.load(f)
            sgld_cfg = cfg.get("sgld_config", {})
            burnin_steps = int(sgld_cfg.get("num_burnin_steps", 0))
            steps_bw = max(1, int(sgld_cfg.get("num_steps_bw_draws", 1)))
            return burnin_steps // steps_bw
        except Exception:
            continue
    return 0


def _log_loss_traces(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
    draw_offset: int = 0,
) -> None:
    """Loss trace per draw — native swanlab.log time-series with zoom/align.

    Args:
        draw_offset: Number of burnin draws to offset the x-axis by,
            so post-burnin draw indices match the runner's x-axis.
    """
    total = pool_seq_loss.shape[0]
    pool_mean = pool_seq_loss.mean(axis=1)
    pool_std = pool_seq_loss.std(axis=1)
    query_mean = query_seq_loss.mean(axis=1)
    query_std = query_seq_loss.std(axis=1)

    if num_chains > 1:
        dpc = total // num_chains
        for draw_idx in range(dpc):
            data = {}
            pool_cvals = []
            query_cvals = []
            for c in range(num_chains):
                offset = c * dpc
                pv = float(pool_seq_loss[offset + draw_idx].mean())
                qv = float(query_seq_loss[offset + draw_idx].mean())
                data[f"1_diag/{ck_name}/pool_loss/chain_{c}"] = pv
                data[f"1_diag/{ck_name}/query_loss/chain_{c}"] = qv
                pool_cvals.append(pv)
                query_cvals.append(qv)
            data[f"1_diag/{ck_name}/pool_loss/chains_mean"] = float(np.mean(pool_cvals))
            data[f"1_diag/{ck_name}/query_loss/chains_mean"] = float(np.mean(query_cvals))
            swan_log(data, step=draw_offset + draw_idx)
    else:
        for draw_idx in range(total):
            swan_log({
                f"1_diag/{ck_name}/pool_loss/mean": float(pool_mean[draw_idx]),
                f"1_diag/{ck_name}/pool_loss/std": float(pool_std[draw_idx]),
                f"1_diag/{ck_name}/query_loss/mean": float(query_mean[draw_idx]),
                f"1_diag/{ck_name}/query_loss/std": float(query_std[draw_idx]),
            }, step=draw_offset + draw_idx)


def _log_score_histogram(
    scores_arr: np.ndarray, ck_name: str, *, bins: int = 40,
) -> None:
    labels, counts = _score_histogram_bars(scores_arr, bins=bins)
    log_bar(f"2_scores/distribution/{ck_name}", xaxis=labels, series={"count": counts})


def _log_corr_distribution(
    bif_matrix: np.ndarray,
    ck_name: str,
    *,
    bins: int = 40,
) -> None:
    n_pool = bif_matrix.shape[0]
    triu_idx = np.triu_indices(n_pool, k=1)
    pool_corr_vals = bif_matrix[triu_idx]

    pool_labels, pool_counts = _score_histogram_bars(pool_corr_vals, bins=bins)
    log_bar(
        f"3_influence/pool_corr_distribution/{ck_name}",
        xaxis=pool_labels,
        series={"count": pool_counts},
    )


def _log_aux_query_corr_distribution(
    cross_corr_matrix: np.ndarray,
    ck_name: str,
    *,
    bins: int = 40,
) -> None:
    cross_corr_avg = cross_corr_matrix.mean(axis=1)

    cross_labels, cross_counts = _score_histogram_bars(cross_corr_avg, bins=bins)
    log_bar(
        f"5_aux_query/cross_corr_distribution/{ck_name}",
        xaxis=cross_labels,
        series={"count": cross_counts},
    )


def _log_score_vs_selfvar_scatter(
    score_arr: np.ndarray,
    pool_seq_loss: np.ndarray,
    ck_name: str,
    *,
    max_points: int = 300,
    chart_key: str = "2_scores/score_vs_selfvar",
    yaxis_name: str = "score",
) -> None:
    """Score vs self-variance: distinguish structure from simple volatility."""
    pool_var = pool_seq_loss.var(axis=0)
    n = len(score_arr)
    max_pts = min(max_points, n)
    if n > max_pts:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n, max_pts, replace=False))
    else:
        idx = np.arange(n)

    log_scatter(
        f"{chart_key}/{ck_name}",
        xaxis_name="pool_self_variance",
        yaxis_name=yaxis_name,
        series={
            "samples": [(float(pool_var[i]), float(score_arr[i])) for i in idx],
        },
    )


def _log_cross_cov_heatmap(
    cross_corr_matrix: np.ndarray,
    pool_ids: list,
    query_ids: list,
    ck_name: str,
    pool_sources: list | None = None,
    *,
    max_pool: int = 50,
    max_query: int = 20,
    query_bar_limit: int = 30,
    query_sources: list | None = None,
    query_task_types: list | None = None,
) -> None:
    """Query sensitivity: which queries are most/least influenced by pool data.

    Always shows:
    1. Histogram of query sensitivity distribution (scales to any count)
    2. Top-K / Bottom-K bar chart with meaningful query IDs
    3. Per-source-group aggregation (if query metadata available)
    """
    n_pool = cross_corr_matrix.shape[0]
    n_query = cross_corr_matrix.shape[1]

    pool_mean_per_query = cross_corr_matrix[:, :n_query].mean(axis=0)
    pool_std_per_query = cross_corr_matrix[:, :n_query].std(axis=0)

    def _make_query_label(idx: int) -> str:
        if query_ids and idx < len(query_ids):
            sid = str(query_ids[idx])
            if len(sid) > 18:
                sid = sid[:8] + ".." + sid[-6:]
            return sid
        return f"q{idx}"

    sorted_idx = np.argsort(-pool_mean_per_query)
    n_head = min(10, n_query)
    n_tail = min(10, n_query)
    head_idx = sorted_idx[:n_head]
    tail_idx = sorted_idx[-n_tail:]

    if n_query <= n_head + n_tail + 3:
        all_labels = [_make_query_label(i) for i in sorted_idx]
        all_means = [round(float(pool_mean_per_query[i]), 4) for i in sorted_idx]
        all_stds = [round(float(pool_std_per_query[i]), 4) for i in sorted_idx]
    else:
        head_labels = [_make_query_label(i) + f"(#{r+1})" for r, i in enumerate(head_idx)]
        tail_labels = [_make_query_label(i) + f"(#{n_query-n_tail+r+1})" for r, i in enumerate(tail_idx)]
        all_labels = head_labels + ["..."] + tail_labels
        all_means = [round(float(pool_mean_per_query[i]), 4) for i in head_idx] + [None] + [round(float(pool_mean_per_query[i]), 4) for i in tail_idx]
        all_stds = [round(float(pool_std_per_query[i]), 4) for i in head_idx] + [None] + [round(float(pool_std_per_query[i]), 4) for i in tail_idx]

    log_bar(
        f"3_influence/query_sensitivity/{ck_name}",
        xaxis=all_labels,
        series={
            "mean_cross_corr": all_means,
            "std_cross_corr": all_stds,
        },
    )

    labels_hist, counts_hist = _score_histogram_bars(
        pool_mean_per_query, bins=min(40, max(10, n_query // 5))
    )
    log_bar(
        f"3_influence/query_sensitivity_distribution/{ck_name}",
        xaxis=labels_hist,
        series={"count": counts_hist},
    )

    if query_sources is not None and len(query_sources) == n_query:
        sources = sorted(set(query_sources))
        if len(sources) > 1:
            src_means = []
            src_stds = []
            src_labels = []
            for src in sources:
                mask = [i for i, s in enumerate(query_sources) if s == src]
                vals = pool_mean_per_query[mask]
                src_labels.append(str(src)[:20])
                src_means.append(round(float(vals.mean()), 4))
                src_stds.append(round(float(vals.std()), 4))
            log_bar(
                f"3_influence/query_sensitivity_by_source/{ck_name}",
                xaxis=src_labels,
                series={
                    "mean_cross_corr": src_means,
                    "std_cross_corr": src_stds,
                },
            )

    if query_task_types is not None and len(query_task_types) == n_query:
        tasks = sorted(set(t for t in query_task_types if t is not None))
        if len(tasks) > 1:
            task_means = []
            task_stds = []
            task_labels = []
            for t in tasks:
                mask = [i for i, tt in enumerate(query_task_types) if tt == t]
                vals = pool_mean_per_query[mask]
                task_labels.append(str(t)[:20])
                task_means.append(round(float(vals.mean()), 4))
                task_stds.append(round(float(vals.std()), 4))
            log_bar(
                f"3_influence/query_sensitivity_by_task/{ck_name}",
                xaxis=task_labels,
                series={
                    "mean_cross_corr": task_means,
                    "std_cross_corr": task_stds,
                },
            )

    if pool_sources is not None and len(pool_sources) == n_pool:
        sources = sorted(set(pool_sources))
        source_to_indices = {src: [] for src in sources}
        for i in range(n_pool):
            source_to_indices[pool_sources[i]].append(i)

        src_query_mat = np.array([
            cross_corr_matrix[source_to_indices[src], :n_query].mean(axis=0)
            for src in sources
        ])
        inter_source_std = src_query_mat.std(axis=0)

        labels_hist2, counts_hist2 = _score_histogram_bars(
            inter_source_std, bins=min(30, max(10, n_query // 10))
        )
        log_bar(
            f"3_influence/query_sensitivity_source_spread/{ck_name}",
            xaxis=labels_hist2,
            series={"count": counts_hist2},
        )
    else:
        max_p = min(max_pool, n_pool)
        max_q = min(max_query, n_query)
        pool_labels = [f"p{i}" for i in range(max_p)]
        query_labels = [_make_query_label(j) for j in range(max_q)]
        log_heatmap(
            f"3_influence/pool_x_query_heatmap/{ck_name}",
            xaxis=query_labels,
            yaxis=pool_labels,
            matrix=cross_corr_matrix[:max_p, :max_q],
            value_label="cross_corr",
        )


def _log_bif_heatmap_topk(
    bif_mat: np.ndarray,
    df: pd.DataFrame,
    pool_ids: list,
    top_k: int,
    ck_name: str,
    score_col: str = "bif_mean",
    *,
    heatmap_topk_max: int = 50,
) -> None:
    """Replace giant heatmap with interpretable summary charts.

    Old approach: 50×50 (or 500×500) heatmap — unreadable and huge.
    New approach:
      1. topk_pairwise_corr_distribution — histogram of pairwise BIF correlations
         among top-K samples (shows whether top samples form a cluster or are diverse).
      2. intra_vs_inter_source_corr — bar chart comparing within-source vs
         cross-source mean BIF correlation per source (shows source block structure
         or its absence).
      3. topk_source_overlap — small heatmap showing which sources contribute to
         the top-K and how much they overlap with each other.
    """
    n_pool = bif_mat.shape[0]
    k = min(heatmap_topk_max, top_k, n_pool)
    mat = bif_mat[:k, :k].copy()
    np.fill_diagonal(mat, np.nan)

    valid = mat[~np.isnan(mat)]
    if len(valid) == 0:
        return

    labels, counts = _score_histogram_bars(valid, bins=min(40, max(10, k // 2)))
    log_bar(
        f"3_influence/topk_pairwise_corr_distribution/{ck_name}",
        xaxis=labels,
        series={"count": counts},
    )

    if "source" in df.columns:
        sources = sorted(df["source"].fillna("unknown").unique().tolist())
        if len(sources) >= 2:
            source_score = {}
            for src in sources:
                mask = (df["source"].fillna("unknown") == src).to_numpy()
                source_score[src] = float(df.loc[mask, score_col].mean()) if score_col in df.columns else 0.0

            sources_sorted = sorted(sources, key=lambda s: -source_score.get(s, 0))
            src_labels = [str(s)[:20] for s in sources_sorted]

            intra_vals = []
            inter_vals = []
            for src in sources_sorted:
                mask_i = (df["source"].fillna("unknown") == src).to_numpy()
                idx_i = np.where(mask_i)[0]
                if len(idx_i) > 1:
                    sub = bif_mat[np.ix_(idx_i, idx_i)]
                    intra_vals.append(float(sub[~np.eye(len(idx_i), dtype=bool)].mean()))
                else:
                    intra_vals.append(0.0)

                other_idx = np.where(~mask_i)[0]
                if len(other_idx) > 0 and len(idx_i) > 0:
                    inter_vals.append(float(bif_mat[np.ix_(idx_i, other_idx)].mean()))
                else:
                    inter_vals.append(0.0)

            log_bar(
                f"3_influence/intra_vs_inter_source_corr/{ck_name}",
                xaxis=src_labels,
                series={
                    "intra_source_mean": [round(v, 4) for v in intra_vals],
                    "inter_source_mean": [round(v, 4) for v in inter_vals],
                },
            )

            n_src = len(sources_sorted)
            overlap_mat = np.zeros((n_src, n_src))
            for i, si in enumerate(sources_sorted):
                for j, sj in enumerate(sources_sorted):
                    if i == j:
                        overlap_mat[i, j] = float((df.head(k)["source"].fillna("unknown") == si).sum())
                    else:
                        top_mask = (df.head(k)["source"].fillna("unknown") == si).values
                        si_idx = np.where(top_mask)[0]
                        sj_all = np.where((df["source"].fillna("unknown") == sj).values)[0]
                        if len(si_idx) > 0 and len(sj_all) > 0:
                            overlap_mat[i, j] = float(bif_mat[np.ix_(si_idx, sj_all)].mean())
                        else:
                            overlap_mat[i, j] = 0.0
            log_heatmap(
                f"3_influence/topk_source_overlap/{ck_name}",
                xaxis=src_labels,
                yaxis=src_labels,
                matrix=overlap_mat,
                value_label="mean_BIF",
            )


def _log_score_by_source(
    df: pd.DataFrame,
    score_col: str,
    top_k: int,
    ck_name: str,
    *,
    max_sources: int = 20,
    min_per_source: int = 5,
    label_max_len: int = 20,
) -> None:
    """Score distribution by source + enrichment metrics."""
    if "source" not in df.columns or score_col not in df.columns:
        return

    sources = sorted(df["source"].fillna("unknown").unique().tolist())

    topk_src_frac = df.head(top_k)["source"].fillna("unknown").value_counts(normalize=True)
    bottomk_src_frac = df.tail(top_k)["source"].fillna("unknown").value_counts(normalize=True)
    pool_src_frac = df["source"].fillna("unknown").value_counts(normalize=True)
    all_sources_sorted = sorted(pool_src_frac.index.tolist())
    src_labels = [str(s)[:label_max_len] for s in all_sources_sorted]

    log_bar(
        f"3_influence/source_distribution/{ck_name}",
        xaxis=src_labels,
        series={
            "top_k": [round(float(topk_src_frac.get(s, 0.0)), 4) for s in all_sources_sorted],
            "bottom_k": [round(float(bottomk_src_frac.get(s, 0.0)), 4) for s in all_sources_sorted],
            "pool": [round(float(pool_src_frac.get(s, 0.0)), 4) for s in all_sources_sorted],
        },
        stack=False,
    )

    if len(sources) >= 2 and len(sources) <= max_sources:
        box_data = []
        labels = []
        for src in sources:
            vals = df[df["source"].fillna("unknown") == src][score_col].dropna().values
            if len(vals) < min_per_source:
                continue
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            lower = max(float(vals.min()), q1 - 1.5 * iqr)
            upper = min(float(vals.max()), q3 + 1.5 * iqr)
            box_data.append([round(lower, 6), round(q1, 6), round(median, 6), round(q3, 6), round(upper, 6)])
            labels.append(str(src)[:label_max_len])

        if len(box_data) >= 2:
            log_boxplot(
                f"3_influence/score_by_source/{ck_name}",
                xaxis=labels,
                series={score_col: box_data},
            )

        enrichment = [
            round(
                (float(topk_src_frac.get(src, 0)) + 1e-9)
                / (float(pool_src_frac.get(src, 0)) + 1e-9),
                4,
            )
            for src in all_sources_sorted
        ]
        log_bar(
            f"3_influence/enrichment/{ck_name}",
            xaxis=src_labels,
            series={"enrichment_ratio": enrichment},
        )


def _log_eigenvalue_spectrum(
    bif_mat: np.ndarray,
    ck_name: str,
    *,
    max_pool: int = 800,
    max_ev: int = 20,
) -> None:
    """Eigenvalue spectrum of BIF matrix: effective dimensionality of influence."""
    n = bif_mat.shape[0]
    if n > max_pool:
        return
    symmetric = (bif_mat + bif_mat.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
        eigenvalues = np.sort(eigenvalues)[::-1]
        n_ev = min(max_ev, len(eigenvalues))
        ev_labels = [f"ev{i}" for i in range(n_ev)]
        total_var = float(eigenvalues.sum()) + 1e-12
        cum_frac = [round(float(eigenvalues[:k].sum()) / total_var, 4) for k in range(1, n_ev + 1)]
        log_bar(
            f"2_scores/eigenvalue_spectrum/{ck_name}",
            xaxis=ev_labels,
            series={"eigenvalue": [round(v, 6) for v in eigenvalues[:n_ev].tolist()], "cumulative_frac": cum_frac},
        )
    except Exception:
        pass


def _log_convergence(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
    *,
    checkpoints: list[int] | None = None,
    min_draws: int = 3,
) -> None:
    """BIF score convergence: native swanlab.log time-series vs n_draws."""
    total_draws = pool_seq_loss.shape[0]
    if checkpoints is None:
        checkpoints = [5, 10, 20, 30, 50, 80, 100, 150, 200]
    checkpoints = sorted(set(checkpoints + [total_draws]))
    checkpoints = [c for c in checkpoints if c <= total_draws and c >= min_draws]
    if len(checkpoints) < 2:
        return

    for n_draws in checkpoints:
        sub_loss = pool_seq_loss[:n_draws]
        try:
            sub_scores = compute_bif_pairwise(sub_loss, num_chains=num_chains, reduce_chains="stack")
            np.fill_diagonal(sub_scores, 0.0)
            bif_mean = sub_scores.mean(axis=1)
            swan_log({
                f"1_diag/{ck_name}/convergence/bif_mean_avg": float(bif_mean.mean()),
                f"1_diag/{ck_name}/convergence/bif_mean_std": float(bif_mean.std()),
            }, step=n_draws)
        except Exception:
            pass


def _log_rhat(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
    *,
    max_samples: int = 50,
    min_draws: int = 5,
) -> None:
    """Gelman-Rubin R-hat: are chains converged? R-hat < 1.1 means OK."""
    if num_chains < 2:
        return
    draws_per_chain = pool_seq_loss.shape[0] // num_chains
    if draws_per_chain < min_draws:
        return

    n_samples = pool_seq_loss.shape[1]
    max_s = min(max_samples, n_samples)

    rhat_values = []
    for s in range(max_s):
        chains_data = []
        for c in range(num_chains):
            chain_loss = pool_seq_loss[c * draws_per_chain:(c + 1) * draws_per_chain, s]
            chains_data.append(chain_loss)
        chains_arr = np.array(chains_data)

        chain_means = chains_arr.mean(axis=1)
        chain_vars = chains_arr.var(axis=1, ddof=1)

        grand_mean = chain_means.mean()
        B = draws_per_chain / (num_chains - 1) * np.sum((chain_means - grand_mean) ** 2)
        W = chain_vars.mean()

        if W < 1e-12:
            continue
        var_hat = (1 - 1.0 / draws_per_chain) * W + B / draws_per_chain
        rhat = float(np.sqrt(var_hat / W))
        rhat_values.append(rhat)

    if not rhat_values:
        return

    rhat_arr = np.array(rhat_values)
    log_bar(
        f"1_diag/rhat/{ck_name}",
        xaxis=["mean", "max", "frac<1.1", "frac<1.2"],
        series={
            "rhat": [
                round(float(rhat_arr.mean()), 4),
                round(float(rhat_arr.max()), 4),
                round(float((rhat_arr < 1.1).mean()), 4),
                round(float((rhat_arr < 1.2).mean()), 4),
            ],
        },
    )


def _compute_rhat_summary(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    *,
    max_samples: int = 50,
    min_draws: int = 5,
) -> dict[str, float]:
    if num_chains < 2:
        return {}
    draws_per_chain = pool_seq_loss.shape[0] // num_chains
    if draws_per_chain < min_draws:
        return {}

    n_samples = pool_seq_loss.shape[1]
    max_s = min(max_samples, n_samples)
    rhat_values = []
    for s in range(max_s):
        chains_data = []
        for c in range(num_chains):
            chain_loss = pool_seq_loss[c * draws_per_chain:(c + 1) * draws_per_chain, s]
            chains_data.append(chain_loss)
        chains_arr = np.array(chains_data)
        chain_means = chains_arr.mean(axis=1)
        chain_vars = chains_arr.var(axis=1, ddof=1)
        grand_mean = chain_means.mean()
        B = draws_per_chain / (num_chains - 1) * np.sum((chain_means - grand_mean) ** 2)
        W = chain_vars.mean()
        if W < 1e-12:
            continue
        var_hat = (1 - 1.0 / draws_per_chain) * W + B / draws_per_chain
        rhat_values.append(float(np.sqrt(var_hat / W)))

    if not rhat_values:
        return {}

    rhat_arr = np.array(rhat_values, dtype=np.float64)
    return {
        "rhat_mean": float(rhat_arr.mean()),
        "rhat_max": float(rhat_arr.max()),
        "rhat_frac_lt_1p1": float((rhat_arr < 1.1).mean()),
    }


def _log_chain_scatter(
    pool_seq_loss: np.ndarray,
    num_chains: int,
    scores_arr: np.ndarray,
    ck_name: str,
    *,
    max_points: int = 300,
    min_draws: int = 3,
) -> None:
    """Each chain's score vs mean-of-other-chains — scales to N chains."""
    if num_chains < 2:
        return
    draws_per_chain = pool_seq_loss.shape[0] // num_chains
    if draws_per_chain < min_draws:
        return

    chain_scores = []
    for c in range(num_chains):
        chain_loss = pool_seq_loss[c * draws_per_chain:(c + 1) * draws_per_chain]
        try:
            chain_bif = compute_bif_pairwise(chain_loss, num_chains=1, reduce_chains="stack")
            np.fill_diagonal(chain_bif, 0.0)
            chain_scores.append(chain_bif.mean(axis=1))
        except Exception:
            return

    chain_scores_arr = np.array(chain_scores)
    n = chain_scores_arr.shape[1]
    max_pts = min(max_points, n)
    if n > max_pts:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n, max_pts, replace=False))
    else:
        idx = np.arange(n)

    for c in range(num_chains):
        others = [j for j in range(num_chains) if j != c]
        others_mean = chain_scores_arr[others].mean(axis=0)
        log_scatter(
            f"1_diag/chain_vs_rest/chain_{c}/{ck_name}",
            xaxis_name=f"chain_{c}_score",
            yaxis_name="mean_other_chains",
            series={
                "samples": [(float(chain_scores_arr[c][k]), float(others_mean[k])) for k in idx]
            },
        )


def _log_core_bif_summary(
    bif_mat: np.ndarray,
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int,
    ck_name: str,
    *,
    rhat_max_samples: int,
    rhat_min_draws: int,
) -> dict[str, float]:
    offdiag = _offdiag_values(bif_mat)
    pool_loss_trace = pool_seq_loss.mean(axis=1)
    query_loss_trace = query_seq_loss.mean(axis=1)

    summary = {
        "bif_offdiag_mean": float(offdiag.mean()) if len(offdiag) else 0.0,
        "bif_offdiag_std": float(offdiag.std()) if len(offdiag) else 0.0,
        "bif_positive_frac": float((offdiag > 0).mean()) if len(offdiag) else 0.0,
        "bif_abs_mean": float(np.abs(offdiag).mean()) if len(offdiag) else 0.0,
        "pool_loss_mean": float(pool_loss_trace.mean()),
        "query_loss_mean": float(query_loss_trace.mean()),
    }
    summary.update(
        _compute_rhat_summary(
            pool_seq_loss,
            num_chains,
            max_samples=rhat_max_samples,
            min_draws=rhat_min_draws,
        )
    )

    rows = [[k, round(v, 6)] for k, v in summary.items()]
    log_table(
        f"1_diag/{ck_name}/core_summary",
        headers=["metric", "value"],
        rows=rows,
    )
    return summary


def _log_rank_stability(
    names: list[str], score_vecs: dict[str, np.ndarray],
) -> None:
    """Spearman correlation heatmap: rank stability across checkpoints."""
    n = len(names)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = spearman_from_scores(score_vecs[a], score_vecs[b])
    log_heatmap(
        "4_2_influence/rank_stability_spearman",
        xaxis=names,
        yaxis=names,
        matrix=mat,
        value_label="Spearman ρ",
    )


def _log_topk_overlap(
    names: list[str], score_vecs: dict[str, np.ndarray], top_k: int,
) -> None:
    """Top-K overlap heatmap across checkpoints."""
    n = len(names)
    k = min(top_k, len(next(iter(score_vecs.values()))))
    mat = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = topk_overlap(score_vecs[a], score_vecs[b], k=k)
    log_heatmap(
        f"4_2_influence/top{k}_overlap",
        xaxis=names,
        yaxis=names,
        matrix=mat,
        value_label="overlap_ratio",
    )


def _log_source_query_overlap_vs_checkpoint(
    names: list[str],
    per_ckpt_df: dict[str, pd.DataFrame],
) -> None:
    sources = sorted(
        {
            str(src)
            for ck in names
            for src in per_ckpt_df[ck]["source"].fillna("unknown").astype(str).unique()
        }
    )
    if not sources or len(names) < 1:
        return

    mat = np.zeros((len(sources), len(names)), dtype=np.float64)
    for ck_idx, ck in enumerate(names):
        cur = per_ckpt_df[ck]
        for src_idx, src in enumerate(sources):
            src_df = cur[cur["source"].fillna("unknown").astype(str) == src]
            if not src_df.empty and "cross_corr_mean_over_queries" in src_df.columns:
                mat[src_idx, ck_idx] = float(src_df["cross_corr_mean_over_queries"].mean())

    log_heatmap(
        "4_2_influence/source/query_overlap_vs_checkpoint",
        xaxis=names,
        yaxis=sources,
        matrix=mat,
        value_label="mean_query_overlap",
    )


def _log_source_recommendations(rec_df: pd.DataFrame) -> None:
    if rec_df.empty:
        return

    preview = rec_df.head(min(20, len(rec_df)))
    rows = []
    for _, row in preview.iterrows():
        rows.append([
            int(row["recommend_rank"]),
            str(row["source"]),
            f"{float(row['query_target_score']):.4f}",
            f"{float(row['bif_training_score']):.4f}",
            f"{float(row['late_bif_mean']):.4f}",
            f"{float(row['late_query_overlap']):.4f}",
            f"{float(row['source_shift']):.4f}",
            f"{float(row['late_topk_frac']):.4f}",
            f"{float(row['late_self_variance']):.4f}",
        ])
    log_table(
        "4_2_influence/source/recommendations",
        headers=[
            "rank",
            "source",
            "query_target",
            "bif_training",
            "late_bif",
            "late_query_overlap",
            "source_shift",
            "late_topk_frac",
            "late_self_var",
        ],
        rows=rows,
    )

    xaxis = preview["source"].astype(str).tolist()
    log_bar(
        "4_2_influence/source/recommendation_scores",
        xaxis=xaxis,
        series={
            "query_target_score": [round(float(v), 4) for v in preview["query_target_score"].tolist()],
            "bif_training_score": [round(float(v), 4) for v in preview["bif_training_score"].tolist()],
            "source_shift": [round(float(v), 4) for v in preview["source_shift"].tolist()],
        },
        stack=False,
    )


def _global_analysis(
    out_dir: str,
    names: list[str],
    acfg: AnalyzeConfig,
    summary_rows: list[dict[str, Any]],
    pool_df: pd.DataFrame | None = None,
) -> None:
    pd.DataFrame(summary_rows).to_csv(f"{out_dir}/checkpoint_summary.csv", index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        headers = [
            "checkpoint",
            "score_mean",
            "score_std",
            "bif_offdiag_mean",
            "bif_offdiag_std",
            "bif_positive_frac",
            "rhat_mean",
            "rhat_max",
            "num_draws",
        ]
        rows = []
        for _, row in summary_df.iterrows():
            rows.append([
                str(row.get("checkpoint", "")),
                f"{float(row.get('score_mean', 0.0)):.4f}",
                f"{float(row.get('score_std', 0.0)):.4f}",
                f"{float(row.get('bif_offdiag_mean', 0.0)):.4f}",
                f"{float(row.get('bif_offdiag_std', 0.0)):.4f}",
                f"{float(row.get('bif_positive_frac', 0.0)):.4f}",
                f"{float(row.get('rhat_mean', float('nan'))):.4f}",
                f"{float(row.get('rhat_max', float('nan'))):.4f}",
                str(int(row.get("num_draws", 0))),
            ])
        log_table("1_diag_global/checkpoint_summary", headers=headers, rows=rows)

        xaxis = summary_df["checkpoint"].astype(str).tolist()
        global_series = {
            "bif_offdiag_mean": summary_df["bif_offdiag_mean"].astype(float).round(6).tolist(),
            "bif_positive_frac": summary_df["bif_positive_frac"].astype(float).round(6).tolist(),
            "score_mean": summary_df["score_mean"].astype(float).round(6).tolist(),
        }
        log_line("1_diag_global/core_metrics", xaxis=xaxis, series=global_series, smooth=False)

        if "rhat_mean" in summary_df.columns:
            rhat_vals = []
            for v in summary_df["rhat_mean"].tolist():
                try:
                    fv = float(v)
                except Exception:
                    fv = float("nan")
                rhat_vals.append(None if np.isnan(fv) else round(fv, 6))
            if any(v is not None for v in rhat_vals):
                log_line(
                    "1_diag_global/rhat",
                    xaxis=xaxis,
                    series={"rhat_mean": rhat_vals},
                    smooth=False,
                )

    per_ckpt_df: dict[str, pd.DataFrame] = {}
    score_vecs: dict[str, np.ndarray] = {}
    per_ckpt_top: dict[str, pd.DataFrame] = {}
    for ck_name in names:
        csv_path = f"{out_dir}/{ck_name}/pool_scores.csv"
        df = pd.read_csv(csv_path)
        per_ckpt_df[ck_name] = df
        score_vecs[ck_name] = df[acfg.score_col].to_numpy()
        per_ckpt_top[ck_name] = df.head(acfg.top_k).copy()

    if len(names) > 1:
        pair_rows = []
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j <= i:
                    continue
                k = min(acfg.top_k, len(score_vecs[a]))
                pair_rows.append({
                    "checkpoint_a": a,
                    "checkpoint_b": b,
                    "spearman": spearman_from_scores(score_vecs[a], score_vecs[b]),
                    f"top{k}_overlap": topk_overlap(score_vecs[a], score_vecs[b], k=k),
                })
        pd.DataFrame(pair_rows).to_csv(
            f"{out_dir}/pairwise_stability.csv", index=False
        )

    traj_df = make_global_trajectory_df(per_ckpt_df, acfg.score_col)
    traj_df = traj_df.sort_values("traj_mean", ascending=False).reset_index(drop=True)
    traj_df.to_csv(f"{out_dir}/trajectory_scores.csv", index=False)

    _log_sample_table(traj_df, names, acfg.score_col, acfg.top_k, pool_df=pool_df)

    for ck_idx, ck_name in enumerate(names):
        ck_col = f"score__{ck_name}"
        if ck_col in traj_df.columns:
            swan_log(
                {
                    "4_2_influence/trajectory/topk_mean": float(
                        traj_df.head(acfg.top_k)[ck_col].mean()
                    ),
                    "4_2_influence/trajectory/topk_std": float(
                        traj_df.head(acfg.top_k)[ck_col].std()
                    ),
                },
                step=ck_idx,
            )

    if "source" in traj_df.columns:
        sources = sorted(traj_df["source"].dropna().unique().tolist())
        ck_short = [
            n.replace("checkpoint-", "ck").replace("final_model", "final")
            for n in names
        ]
        if sources and len(names) > 1:
            src_ck_mat = np.zeros((len(sources), len(names)))
            for ck_idx, ck_name in enumerate(names):
                ck_col = f"score__{ck_name}"
                if ck_col not in traj_df.columns:
                    continue
                for src_idx, src in enumerate(sources):
                    src_vals = traj_df[traj_df["source"] == src][ck_col]
                    if not src_vals.empty:
                        src_ck_mat[src_idx, ck_idx] = float(src_vals.mean())
            log_heatmap(
                "4_2_influence/source/score_vs_checkpoint",
                xaxis=ck_short,
                yaxis=sources,
                matrix=src_ck_mat,
                value_label="mean_score",
            )

        if sources and len(names) > 1:
            count_mat = np.zeros((len(sources), len(names)))
            for ck_idx, ck_name in enumerate(names):
                top_ck = per_ckpt_top.get(ck_name)
                if top_ck is None or "source" not in top_ck.columns:
                    continue
                vc = top_ck["source"].fillna("unknown").value_counts()
                for src_idx, src in enumerate(sources):
                    count_mat[src_idx, ck_idx] = int(vc.get(src, 0))
            log_heatmap(
                "4_2_influence/source/topk_count_vs_checkpoint",
                xaxis=ck_short,
                yaxis=sources,
                matrix=count_mat,
                value_label="count_in_topK",
            )

    if len(names) > 1:
        _log_rank_stability(names, score_vecs)
        _log_topk_overlap(names, score_vecs, acfg.top_k)

        top_n = acfg.trajectory_top_n
        top_series = _trajectory_stats_series(
            traj_df, names, sort_by="traj_mean", top_n=top_n
        )
        log_line(
            "4_2_influence/trajectory/top_by_mean",
            xaxis=names,
            series=top_series,
            smooth=True,
        )

        emergent_series = _trajectory_stats_series(
            traj_df, names, sort_by="emergence_last_minus_first", top_n=top_n
        )
        log_line(
            "4_2_influence/trajectory/top_emergent",
            xaxis=names,
            series=emergent_series,
            smooth=True,
        )

    if "source" in traj_df.columns:
        shift_series = _source_shift_series(names, per_ckpt_top, "source")
        log_bar(
            "4_2_influence/source/shift_topk",
            xaxis=names,
            series=shift_series,
            stack=True,
        )
        _log_source_query_overlap_vs_checkpoint(names, per_ckpt_df)

    source_rows = []
    for ck in names:
        cur = per_ckpt_df[ck]
        top = cur.head(acfg.top_k)
        top_counts = top["source"].fillna("unknown").value_counts(normalize=True)
        all_counts = cur["source"].fillna("unknown").value_counts(normalize=True)
        all_sources = sorted(
            set(top_counts.index.tolist()) | set(all_counts.index.tolist())
        )
        for src in all_sources:
            source_rows.append(
                {
                    "checkpoint": ck,
                    "source": src,
                    "top_fraction": float(top_counts.get(src, 0.0)),
                    "all_fraction": float(all_counts.get(src, 0.0)),
                    "enrichment_ratio": float(
                        (top_counts.get(src, 0.0) + 1e-12)
                        / (all_counts.get(src, 0.0) + 1e-12)
                    ),
                }
            )
    pd.DataFrame(source_rows).to_csv(
        f"{out_dir}/source_enrichment_topk.csv", index=False
    )

    rec_df = _build_source_recommendations(
        names,
        per_ckpt_df,
        per_ckpt_top,
        acfg.top_k,
        acfg.score_col,
    )
    if not rec_df.empty:
        rec_df.to_csv(f"{out_dir}/source_recommendations.csv", index=False)
        _log_source_recommendations(rec_df)


def _auto_adapt_from_first_checkpoint(
    acfg: AnalyzeConfig,
    all_ckpts: list[tuple[str, str]],
    pool_df: pd.DataFrame | None,
) -> None:
    """Peek at the first checkpoint's trace dims to auto-adapt config."""
    for ck_name, ck_dir in all_ckpts[:1]:
        try:
            loaded = load_checkpoint_traces(ck_dir)
        except Exception:
            continue
        pool_size = loaded["pool_seq_loss"].shape[1]
        query_size = loaded["query_seq_loss"].shape[1]
        num_draws = loaded["pool_seq_loss"].shape[0]
        num_chains = loaded.get("num_chains", 1)
        num_sources = 0
        if pool_df is not None and "source" in pool_df.columns:
            num_sources = pool_df["source"].nunique()
        elif "source_type" in loaded.get("pool_meta", {}):
            num_sources = len(set(loaded["pool_meta"]["source_type"]) - {None})
        _auto_adapt_config(
            acfg,
            pool_size=pool_size,
            query_size=query_size,
            num_draws=num_draws,
            num_chains=num_chains,
            num_sources=num_sources,
        )
        break


def analyze_bif_results(
    bif_root: str,
    out_dir: str,
    acfg: AnalyzeConfig | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    manage_tracking: bool = True,
) -> None:
    if acfg is None:
        acfg = AnalyzeConfig()
    rank, world_size = _get_dist_context()
    ensure_dir(out_dir)

    all_ckpts = discover_checkpoint_dirs(bif_root)
    names = [x[0] for x in all_ckpts]

    pool_df: pd.DataFrame | None = None
    for pool_name in ("pt_pool.jsonl", "pool_10k_rebalanced.jsonl"):
        pool_path = os.path.join(bif_root, "..", "pool", pool_name)
        if os.path.exists(pool_path):
            pool_df = pd.DataFrame(read_jsonl(pool_path))
            break

    if pool_df is None:
        search_dirs = [bif_root]
        parent = os.path.dirname(bif_root)
        for _ in range(3):
            if parent and parent != "/":
                search_dirs.append(parent)
                parent = os.path.dirname(parent)
        import json as _json
        for search_dir in search_dirs:
            run_cfg_path = os.path.join(search_dir, "run_config.json")
            if os.path.exists(run_cfg_path):
                with open(run_cfg_path) as _f:
                    _run_cfg = _json.load(_f)
                _pool_jsonl = _run_cfg.get("pool_jsonl", "")
                if _pool_jsonl and os.path.exists(_pool_jsonl):
                    pool_df = pd.DataFrame(read_jsonl(_pool_jsonl))
                    break

    _auto_adapt_from_first_checkpoint(acfg, all_ckpts, pool_df)

    if rank == 0 and manage_tracking:
        auto_name = make_analyze_name(
            guess_model_tag(bif_root), acfg.score_col, acfg.top_k or 500,
        )
        init_run(
            experiment_name=experiment_name or auto_name,
            run_name=run_name,
            config={"bif_root": bif_root, **{k: v for k, v in acfg.__dict__.items() if v is not None}},
            tags=["analysis"],
        )

    assigned = all_ckpts[rank::world_size]
    summary_rows_local: list[dict[str, Any]] = []
    for ck_name, ck_dir in assigned:
        ck_step = names.index(ck_name) if ck_name in names else 0
        row = _process_one_checkpoint(
            ck_name,
            ck_dir,
            out_dir,
            acfg,
            ck_step=ck_step,
            pool_df=pool_df,
        )
        summary_rows_local.append(row)

    _barrier()

    if rank == 0:
        summary_rows: list[dict[str, Any]] = []
        valid_names: list[str] = []
        for ck_name, _ in all_ckpts:
            csv_path = f"{out_dir}/{ck_name}/pool_scores.csv"
            if not os.path.exists(csv_path):
                print(f"[analyze] Skipping {ck_name} in global analysis (no pool_scores.csv)")
                continue
            meta_path = f"{out_dir}/{ck_name}/ckpt_meta.json"
            df = pd.read_csv(csv_path)
            meta: dict[str, Any] = {}
            if os.path.exists(meta_path):
                import json as _json

                with open(meta_path, encoding="utf-8") as _f:
                    meta = _json.load(_f)
            summary_rows.append(
                {
                    "checkpoint": ck_name,
                    "score_mean": float(df[acfg.score_col].mean()),
                    "score_std": float(df[acfg.score_col].std()),
                    "num_draws": int(meta.get("num_draws", 0)),
                    "pool_size": len(df),
                    "bif_offdiag_mean": float(meta.get("bif_offdiag_mean", 0.0)),
                    "bif_offdiag_std": float(meta.get("bif_offdiag_std", 0.0)),
                    "bif_positive_frac": float(meta.get("bif_positive_frac", 0.0)),
                    "bif_abs_mean": float(meta.get("bif_abs_mean", 0.0)),
                    "rhat_mean": float(meta.get("rhat_mean", float("nan"))),
                    "rhat_max": float(meta.get("rhat_max", float("nan"))),
                }
            )
            valid_names.append(ck_name)

        _global_analysis(
            out_dir, valid_names, acfg, summary_rows, pool_df=pool_df
        )

        save_json(
            f"{out_dir}/analysis_config.json",
            {
                "bif_root": bif_root,
                **{k: v for k, v in acfg.__dict__.items()},
                "checkpoint_names": names,
                "world_size": world_size,
            },
        )

    _barrier()

    if rank == 0 and manage_tracking:
        swan_finish()

    _barrier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze BIF results (single or multi-GPU via torchrun).",
    )
    parser.add_argument("--bif_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_col", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--save_full_query_matrix", action="store_true")
    parser.add_argument("--enable_aux_query_plots", action="store_true")
    parser.add_argument("--negate_scores", action="store_true")
    parser.add_argument("--hist_bins", type=int, default=None)
    parser.add_argument("--scatter_max_points", type=int, default=None)
    parser.add_argument("--trajectory_top_n", type=int, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    acfg = AnalyzeConfig()
    if args.score_col is not None:
        acfg.score_col = args.score_col
    if args.top_k is not None:
        acfg.top_k = args.top_k
    if args.save_full_query_matrix:
        acfg.save_full_query_matrix = True
    if args.enable_aux_query_plots:
        acfg.enable_aux_query_plots = True
    if args.negate_scores:
        acfg.negate_scores = True
    if args.hist_bins is not None:
        acfg.hist_bins = args.hist_bins
    if args.scatter_max_points is not None:
        acfg.scatter_max_points = args.scatter_max_points
    if args.trajectory_top_n is not None:
        acfg.trajectory_top_n = args.trajectory_top_n

    _init_dist_if_needed()
    try:
        analyze_bif_results(
            bif_root=args.bif_root,
            out_dir=args.out_dir,
            acfg=acfg,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
        rank, _ = _get_dist_context()
        if rank == 0:
            print("Analysis complete.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
