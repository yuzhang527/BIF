"""Analyze sequence-only BIF traces.

This analyzer is compatible with the sequence-only runner that saves:
- chain_*/observable_loss_trace.npz with key ``seq_loss``
- chain_*/query_loss_trace.npz with key ``seq_loss``

It intentionally removes all token_loss loading/computation. The primary score is:

    bif_mean(pool_i) = mean_j corr(loss_trace(pool_i), loss_trace(query_j))

where correlations are computed across retained post-burn-in SGLD draws. The sign is
left as correlation by default; pass --negate_scores if you want the raw BIF sign
convention (-Cov / -Corr).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch.distributed as dist

from bif.io import ensure_dir, read_jsonl, save_json
from bif.utils.naming import guess_model_tag, make_analyze_name

try:
    from bif.utils.tracker import finish as swan_finish
    from bif.utils.tracker import init_run, log_bar, log_heatmap, log_line, log_table
except Exception:  # pragma: no cover - lets the analyzer run without tracking.
    def swan_finish() -> None:
        return None

    def init_run(*args, **kwargs) -> None:
        return None

    def log_bar(*args, **kwargs) -> None:
        return None

    def log_heatmap(*args, **kwargs) -> None:
        return None

    def log_line(*args, **kwargs) -> None:
        return None

    def log_table(*args, **kwargs) -> None:
        return None


@dataclass
class AnalyzeConfig:
    """Config for sequence-only BIF analysis."""

    score_col: str = "bif_mean"
    top_k: int | None = None
    bottom_k: int | None = None
    negate_scores: bool = False
    save_full_query_matrix: bool = True
    hist_bins: int | None = None
    scatter_max_points: int | None = None
    heatmap_max_pool: int | None = None
    heatmap_max_query: int | None = None
    source_label_max_len: int | None = None


def _auto_adapt_config(
    acfg: AnalyzeConfig,
    pool_size: int,
    query_size: int,
    num_draws: int,
) -> AnalyzeConfig:
    del num_draws
    if acfg.top_k is None:
        acfg.top_k = min(500, pool_size)
    if acfg.bottom_k is None:
        acfg.bottom_k = acfg.top_k
    if acfg.hist_bins is None:
        acfg.hist_bins = min(60, max(20, pool_size // 10))
    if acfg.scatter_max_points is None:
        acfg.scatter_max_points = min(500, pool_size)
    if acfg.heatmap_max_pool is None:
        acfg.heatmap_max_pool = min(50, pool_size)
    if acfg.heatmap_max_query is None:
        acfg.heatmap_max_query = min(20, query_size)
    if acfg.source_label_max_len is None:
        acfg.source_label_max_len = 25
    return acfg


def _get_dist_context() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _init_dist_if_needed() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if _cuda_available() else "gloo"
        dist.init_process_group(backend=backend)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


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
    """Discover checkpoint output dirs that contain chain_* subdirs."""
    entries: list[tuple[str, str]] = []

    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full) and (
            name in ("base_model", "final_model") or re.fullmatch(r"checkpoint-\d+", name)
        ):
            entries.append((name, full))

    if not entries:
        has_chains_at_root = any(
            os.path.isdir(os.path.join(root, d)) and re.fullmatch(r"chain_\d+", d)
            for d in os.listdir(root)
        )
        if has_chains_at_root:
            entries = [("final_model", root)]
        else:
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if not os.path.isdir(full):
                    continue
                has_chains_sub = any(
                    os.path.isdir(os.path.join(full, d)) and re.fullmatch(r"chain_\d+", d)
                    for d in os.listdir(full)
                )
                if has_chains_sub:
                    entries.append((name, full))

    entries.sort(key=lambda x: _checkpoint_sort_key(x[0]))
    if not entries:
        raise ValueError(
            f"No checkpoint dirs under {root}. Expected chain_*/ at root or "
            "checkpoint/base/final dirs containing chain_*/."
        )
    return entries


def _discover_chain_dirs(checkpoint_dir: str) -> list[str]:
    out: list[str] = []
    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if os.path.isdir(full) and re.fullmatch(r"chain_\d+", name):
            out.append(full)
    out.sort()
    if not out:
        raise ValueError(f"No chain dirs under {checkpoint_dir}")
    return out


def load_checkpoint_traces(checkpoint_dir: str) -> dict[str, Any]:
    """Load sequence-level loss traces from one checkpoint output dir.

    New format: chain_*/observable_loss_trace.npz and query_loss_trace.npz.
    Legacy format: chain_*/pool_loss_trace.jsonl and query_loss_trace.jsonl.
    """
    chain_dirs = _discover_chain_dirs(checkpoint_dir)
    npz_path = os.path.join(chain_dirs[0], "observable_loss_trace.npz")
    if os.path.isfile(npz_path):
        return _load_traces_npz(chain_dirs)
    return _load_traces_legacy(chain_dirs)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalise_meta(meta: dict[str, Any]) -> dict[str, Any]:
    if "source_type" not in meta and "source_types" in meta:
        meta["source_type"] = meta["source_types"]
    if "subtype" not in meta and "subtypes" in meta:
        meta["subtype"] = meta["subtypes"]
    if "task_type" not in meta and "task_types" in meta:
        meta["task_type"] = meta["task_types"]
    return meta


def _load_traces_npz(chain_dirs: list[str]) -> dict[str, Any]:
    pool_seq_parts: list[np.ndarray] = []
    query_seq_parts: list[np.ndarray] = []
    pool_meta: dict[str, Any] | None = None
    query_meta: dict[str, Any] | None = None
    draws_per_chain_parts: list[int] = []

    for cdir in chain_dirs:
        pool_path = os.path.join(cdir, "observable_loss_trace.npz")
        query_path = os.path.join(cdir, "query_loss_trace.npz")
        if not os.path.isfile(pool_path):
            raise ValueError(f"Missing {pool_path}")
        if not os.path.isfile(query_path):
            raise ValueError(f"Missing {query_path}")

        pool_npz = np.load(pool_path)
        query_npz = np.load(query_path)
        if "seq_loss" in pool_npz.files:
            pool_seq = pool_npz["seq_loss"]
        elif "pool_seq_loss" in pool_npz.files:
            pool_seq = pool_npz["pool_seq_loss"]
        else:
            raise ValueError(f"No seq_loss key in {pool_path}")

        if "seq_loss" in query_npz.files:
            query_seq = query_npz["seq_loss"]
        elif "query_seq_loss" in query_npz.files:
            query_seq = query_npz["query_seq_loss"]
        else:
            raise ValueError(f"No seq_loss key in {query_path}")

        if pool_seq.shape[0] != query_seq.shape[0]:
            raise ValueError(
                f"Draw mismatch in {cdir}: pool has {pool_seq.shape[0]}, "
                f"query has {query_seq.shape[0]}"
            )
        pool_seq_parts.append(np.asarray(pool_seq, dtype=np.float64))
        query_seq_parts.append(np.asarray(query_seq, dtype=np.float64))
        draws_per_chain_parts.append(int(pool_seq.shape[0]))

        if pool_meta is None:
            pool_meta = _normalise_meta(_load_json(os.path.join(cdir, "observable_meta.json")))
        if query_meta is None:
            query_meta = _normalise_meta(_load_json(os.path.join(cdir, "query_meta.json")))

    assert pool_meta is not None and query_meta is not None

    # Keep chain order: chain_000 draws, then chain_001 draws, etc.
    pool_seq_all = np.concatenate(pool_seq_parts, axis=0)
    query_seq_all = np.concatenate(query_seq_parts, axis=0)

    pool_nan_frac = float(np.isnan(pool_seq_all).mean())
    query_nan_frac = float(np.isnan(query_seq_all).mean())
    if pool_nan_frac > 0.5 or query_nan_frac > 0.5:
        raise ValueError(
            f"Trace data is mostly NaN (pool={pool_nan_frac:.1%}, "
            f"query={query_nan_frac:.1%}). SGLD likely diverged."
        )

    if len(set(draws_per_chain_parts)) == 1:
        draws_per_chain = draws_per_chain_parts[0]
    else:
        # Keep going, but chain-reduction by mean will not be valid.
        draws_per_chain = min(draws_per_chain_parts)

    return {
        "pool_ids": pool_meta["sample_ids"],
        "pool_seq_loss": pool_seq_all,
        "pool_meta": pool_meta,
        "query_ids": query_meta["sample_ids"],
        "query_seq_loss": query_seq_all,
        "query_meta": query_meta,
        "num_draws": int(pool_seq_all.shape[0]),
        "num_chains": int(len(chain_dirs)),
        "draws_per_chain": int(draws_per_chain),
        "draws_per_chain_parts": draws_per_chain_parts,
    }


def _load_traces_legacy(chain_dirs: list[str]) -> dict[str, Any]:
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

    return {
        "pool_ids": pool_ids,
        "pool_seq_loss": pool_mat,
        "pool_meta": _normalise_meta(pool_meta),
        "query_ids": query_ids,
        "query_seq_loss": query_mat,
        "query_meta": _normalise_meta(query_meta),
        "num_draws": int(num_draws),
        "num_chains": int(num_chains),
        "draws_per_chain": int(draws_per_chain),
    }


def rows_to_loss_matrix(
    rows: list[dict[str, Any]],
    dataset_name: str,
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    rows = [r for r in rows if r.get("dataset") == dataset_name]
    if not rows:
        raise ValueError(f"No rows for dataset={dataset_name}")

    rows.sort(key=lambda r: (int(r["chain_id"]), int(r["draw_in_chain"])))

    template = None
    for r in rows:
        ids = r.get("sample_ids", [])
        losses = r.get("losses", [])
        if isinstance(ids, list) and isinstance(losses, list) and ids and len(ids) == len(losses):
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
    draw_meta: list[dict[str, int]] = []
    for draw_idx, r in enumerate(valid_rows):
        for sid, loss in zip(r["sample_ids"], r["losses"]):
            mat[draw_idx, id_to_idx[sid]] = float(loss)
        draw_meta.append(
            {
                "chain_id": int(r["chain_id"]),
                "draw_in_chain": int(r["draw_in_chain"]),
                "global_draw": int(r.get("global_draw", draw_idx)),
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


def _safe_zscore_cols(mat: np.ndarray) -> np.ndarray:
    mu = np.nanmean(mat, axis=0, keepdims=True)
    sd = np.nanstd(mat, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (mat - mu) / sd


def compute_bif_scores(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int = 1,
    reduce_chains: str = "stack",
    negate_scores: bool = False,
) -> dict[str, np.ndarray]:
    """Compute sequence-level pool x query BIF scores."""
    if pool_seq_loss.shape[0] != query_seq_loss.shape[0]:
        raise ValueError(
            "pool_seq_loss and query_seq_loss must have the same number of draws; "
            f"got {pool_seq_loss.shape[0]} and {query_seq_loss.shape[0]}"
        )

    if reduce_chains == "stack":
        pool_loss = pool_seq_loss
        query_loss = query_seq_loss
    elif reduce_chains == "mean":
        if num_chains <= 0:
            raise ValueError("num_chains must be positive")
        if pool_seq_loss.shape[0] % num_chains != 0:
            raise ValueError(
                f"num_draws={pool_seq_loss.shape[0]} is not divisible by num_chains={num_chains}"
            )
        draws_per_chain = pool_seq_loss.shape[0] // num_chains
        pool_loss = pool_seq_loss.reshape(num_chains, draws_per_chain, -1).mean(axis=0)
        query_loss = query_seq_loss.reshape(num_chains, draws_per_chain, -1).mean(axis=0)
    else:
        raise ValueError(f"Unknown reduce_chains: {reduce_chains}")

    pool_z = _safe_zscore_cols(pool_loss)
    query_z = _safe_zscore_cols(query_loss)

    n_obs = max(pool_z.shape[0], 1)
    cross_corr = (pool_z.T @ query_z) / n_obs

    pool_centered = pool_loss - pool_loss.mean(axis=0, keepdims=True)
    query_centered = query_loss - query_loss.mean(axis=0, keepdims=True)
    cross_cov = (pool_centered.T @ query_centered) / n_obs

    sign = -1.0 if negate_scores else 1.0
    cross_corr_mean = sign * cross_corr.mean(axis=1)
    cross_corr_absmean = np.abs(cross_corr).mean(axis=1)
    cross_cov_mean = sign * cross_cov.mean(axis=1)

    mean_loss = pool_loss.mean(axis=0)
    self_variance = pool_loss.var(axis=0)

    draw_idx = np.arange(pool_loss.shape[0], dtype=np.float64)
    draw_idx = (draw_idx - draw_idx.mean()) / (draw_idx.std() + 1e-12)
    draw_trend = ((pool_z.T @ draw_idx) / max(len(draw_idx), 1)).reshape(-1)

    return {
        "bif_mean": cross_corr_mean,
        "cross_corr_mean_over_queries": cross_corr_mean,
        "cross_corr_absmean_over_queries": cross_corr_absmean,
        "cross_corr_matrix": cross_corr,
        "cross_cov_avg_over_queries": cross_cov_mean,
        "mean_loss": mean_loss,
        "self_variance": self_variance,
        "draw_trend": draw_trend,
    }


def build_pool_score_df(
    pool_ids: list[Any],
    pool_meta: dict[str, Any],
    score_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    src_list = pool_meta.get("source_type", [None] * len(pool_ids))
    sub_list = pool_meta.get("subtype", [None] * len(pool_ids))
    task_list = pool_meta.get("task_type", [None] * len(pool_ids))

    if not isinstance(src_list, (list, tuple)) or len(src_list) != len(pool_ids):
        src_list = [None] * len(pool_ids)
    if not isinstance(sub_list, (list, tuple)) or len(sub_list) != len(pool_ids):
        sub_list = [None] * len(pool_ids)
    if not isinstance(task_list, (list, tuple)) or len(task_list) != len(pool_ids):
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


def _score_histogram_bars(scores: np.ndarray, bins: int = 40) -> tuple[list[str], list[int]]:
    counts, edges = np.histogram(scores, bins=bins)
    labels = [f"{edges[i]:.3f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()


def _load_pool_texts_near_root(bif_root: str) -> pd.DataFrame | None:
    # Best-effort preview text lookup, matching the original analyzer behavior.
    candidates = [
        os.path.join(bif_root, "..", "pool", "pt_pool.jsonl"),
        os.path.join(bif_root, "..", "pool", "pool_10k_rebalanced.jsonl"),
    ]
    parent = os.path.dirname(bif_root)
    for _ in range(4):
        if not parent or parent == "/":
            break
        run_cfg = os.path.join(parent, "run_config.json")
        if os.path.exists(run_cfg):
            try:
                with open(run_cfg, encoding="utf-8") as f:
                    cfg = json.load(f)
                p = cfg.get("pool_jsonl")
                if p:
                    candidates.append(p)
            except Exception:
                pass
        parent = os.path.dirname(parent)

    for path in candidates:
        if path and os.path.exists(path):
            try:
                return pd.DataFrame(read_jsonl(path))
            except Exception:
                pass
    return None


def _make_text_map(pool_df: pd.DataFrame | None) -> dict[str, str]:
    if pool_df is None:
        return {}
    id_col = "id" if "id" in pool_df.columns else "sample_id"
    if id_col in pool_df.columns and "text" in pool_df.columns:
        return dict(zip(pool_df[id_col].astype(str), pool_df["text"].astype(str)))
    return {}


def _fmt_preview(text: str, max_len: int = 240) -> str:
    s = str(text).strip().replace("\n", " ").replace("\r", " ")
    return s[:max_len] + "..." if len(s) > max_len else s


def _log_basic_charts(
    ck_name: str,
    df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    pool_seq: np.ndarray,
    query_seq: np.ndarray,
    query_ids: list[Any],
    acfg: AnalyzeConfig,
) -> None:
    score_arr = df[acfg.score_col].to_numpy(dtype=np.float64)
    labels, counts = _score_histogram_bars(score_arr, bins=int(acfg.hist_bins or 40))
    log_bar(f"2_scores/distribution/{ck_name}", xaxis=labels, series={"count": counts})

    # Loss traces as line chart.
    n = pool_seq.shape[0]
    xaxis = [str(i) for i in range(n)]
    log_line(
        f"1_diag/loss_trace/{ck_name}",
        xaxis=xaxis,
        series={
            "pool_mean": [float(v) for v in pool_seq.mean(axis=1)],
            "query_mean": [float(v) for v in query_seq.mean(axis=1)],
        },
        smooth=True,
    )

    # Small pool x query heatmap.
    mat = scores["cross_corr_matrix"]
    max_p = min(int(acfg.heatmap_max_pool or 50), mat.shape[0])
    max_q = min(int(acfg.heatmap_max_query or 20), mat.shape[1])
    query_labels = [str(query_ids[j])[:18] for j in range(max_q)]
    pool_labels = [f"p{i}" for i in range(max_p)]
    log_heatmap(
        f"3_influence/pool_x_query_heatmap/{ck_name}",
        xaxis=query_labels,
        yaxis=pool_labels,
        matrix=mat[:max_p, :max_q],
        value_label="corr",
    )

    # Top table.
    rows = []
    for rank_i, (_, row) in enumerate(df.head(min(50, len(df))).iterrows(), 1):
        rows.append(
            [
                rank_i,
                str(row.get("sample_id", "")),
                str(row.get("source", "")),
                f"{float(row.get(acfg.score_col, 0.0)):.6f}",
                f"{float(row.get('cross_corr_absmean_over_queries', 0.0)):.6f}",
                f"{float(row.get('self_variance', 0.0)):.6f}",
                f"{float(row.get('mean_loss', 0.0)):.6f}",
            ]
        )
    log_table(
        f"4_2_influence/samples/top/{ck_name}",
        headers=["rank", "sample_id", "source", acfg.score_col, "absmean", "self_var", "mean_loss"],
        rows=rows,
    )


def _process_one_checkpoint(
    ck_name: str,
    ck_dir: str,
    out_dir: str,
    acfg: AnalyzeConfig,
    pool_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    loaded = load_checkpoint_traces(ck_dir)
    pool_seq = loaded["pool_seq_loss"]
    query_seq = loaded["query_seq_loss"]

    if np.isnan(pool_seq).any() or np.isnan(query_seq).any():
        pool_nan = float(np.isnan(pool_seq).mean())
        query_nan = float(np.isnan(query_seq).mean())
        raise ValueError(
            f"Trace contains NaN (pool={pool_nan:.1%}, query={query_nan:.1%}). "
            "SGLD diverged."
        )

    num_chains = int(loaded.get("num_chains", 1))
    scores = compute_bif_scores(
        pool_seq,
        query_seq,
        num_chains=num_chains,
        reduce_chains="stack",
        negate_scores=acfg.negate_scores,
    )
    df = build_pool_score_df(loaded["pool_ids"], loaded["pool_meta"], scores)

    if acfg.score_col not in df.columns:
        raise ValueError(
            f"score_col={acfg.score_col!r} is not present. Available columns: {df.columns.tolist()}"
        )

    df = df.sort_values(acfg.score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    text_map = _make_text_map(pool_df)
    if text_map:
        df["text_preview"] = df["sample_id"].astype(str).map(lambda sid: _fmt_preview(text_map.get(sid, "")))

    ck_out = os.path.join(out_dir, ck_name)
    ensure_dir(ck_out)
    df.to_csv(os.path.join(ck_out, "pool_scores.csv"), index=False)

    top_k = min(int(acfg.top_k or len(df)), len(df))
    bottom_k = min(int(acfg.bottom_k or top_k), len(df))
    df.head(top_k).to_csv(os.path.join(ck_out, f"top_{top_k}.csv"), index=False)
    df.tail(bottom_k).iloc[::-1].to_csv(os.path.join(ck_out, f"bottom_{bottom_k}.csv"), index=False)

    if acfg.save_full_query_matrix:
        np.save(os.path.join(ck_out, "pool_query_corr_matrix.npy"), scores["cross_corr_matrix"])

    cross_vals = scores["cross_corr_matrix"].reshape(-1)
    meta = {
        "checkpoint": ck_name,
        "score_definition": "bif_mean = mean_j corr(pool_i_loss_trace, query_j_loss_trace)",
        "negate_scores": acfg.negate_scores,
        "num_draws": int(loaded["num_draws"]),
        "num_chains": int(num_chains),
        "draws_per_chain": int(loaded.get("draws_per_chain", 0)),
        "pool_size": int(pool_seq.shape[1]),
        "query_size": int(query_seq.shape[1]),
        "score_mean": float(df[acfg.score_col].mean()),
        "score_std": float(df[acfg.score_col].std()),
        "score_min": float(df[acfg.score_col].min()),
        "score_max": float(df[acfg.score_col].max()),
        "cross_corr_mean": float(np.nanmean(cross_vals)),
        "cross_corr_std": float(np.nanstd(cross_vals)),
        "cross_corr_min": float(np.nanmin(cross_vals)),
        "cross_corr_max": float(np.nanmax(cross_vals)),
        "matrix_saved": bool(acfg.save_full_query_matrix),
    }
    save_json(os.path.join(ck_out, "ckpt_meta.json"), meta)

    _log_basic_charts(
        ck_name=ck_name,
        df=df,
        scores=scores,
        pool_seq=pool_seq,
        query_seq=query_seq,
        query_ids=loaded["query_ids"],
        acfg=acfg,
    )

    return meta


def _auto_adapt_from_first_checkpoint(
    acfg: AnalyzeConfig,
    all_ckpts: list[tuple[str, str]],
) -> None:
    for _, ck_dir in all_ckpts[:1]:
        loaded = load_checkpoint_traces(ck_dir)
        _auto_adapt_config(
            acfg,
            pool_size=loaded["pool_seq_loss"].shape[1],
            query_size=loaded["query_seq_loss"].shape[1],
            num_draws=loaded["pool_seq_loss"].shape[0],
        )
        break


def _make_global_trajectory(
    out_dir: str,
    valid_names: list[str],
    score_col: str,
) -> pd.DataFrame | None:
    if not valid_names:
        return None

    base = pd.read_csv(os.path.join(out_dir, valid_names[0], "pool_scores.csv"))
    merged = base[["sample_id", "source", "subtype", "task_type"]].copy()
    for ck in valid_names:
        cur = pd.read_csv(os.path.join(out_dir, ck, "pool_scores.csv"))
        cur = cur[["sample_id", score_col]].rename(columns={score_col: f"score__{ck}"})
        merged = merged.merge(cur, on="sample_id", how="left")

    score_cols = [f"score__{ck}" for ck in valid_names]
    merged["traj_mean"] = merged[score_cols].mean(axis=1)
    merged["traj_std"] = merged[score_cols].std(axis=1)
    merged["traj_min"] = merged[score_cols].min(axis=1)
    merged["traj_max"] = merged[score_cols].max(axis=1)
    if len(score_cols) >= 2:
        merged["emergence_last_minus_first"] = merged[score_cols[-1]] - merged[score_cols[0]]
    else:
        merged["emergence_last_minus_first"] = 0.0

    merged = merged.sort_values("traj_mean", ascending=False).reset_index(drop=True)
    merged.to_csv(os.path.join(out_dir, "trajectory_scores.csv"), index=False)
    return merged


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
    names = [name for name, _ in all_ckpts]
    _auto_adapt_from_first_checkpoint(acfg, all_ckpts)

    pool_df = _load_pool_texts_near_root(bif_root)

    if rank == 0 and manage_tracking:
        auto_name = make_analyze_name(
            guess_model_tag(bif_root),
            acfg.score_col,
            int(acfg.top_k or 500),
        )
        init_run(
            experiment_name=experiment_name or auto_name,
            run_name=run_name,
            config={"bif_root": bif_root, **acfg.__dict__},
            tags=["analysis", "sequence_only"],
        )

    assigned = all_ckpts[rank::world_size]
    local_errors: list[dict[str, Any]] = []
    for ck_name, ck_dir in assigned:
        try:
            _process_one_checkpoint(ck_name, ck_dir, out_dir, acfg, pool_df=pool_df)
        except Exception as exc:
            print(f"[analyze] Skipping {ck_name}: {exc}")
            local_errors.append({"checkpoint": ck_name, "error": str(exc)})

    if local_errors:
        err_path = os.path.join(out_dir, f"analysis_errors_rank{rank:03d}.json")
        save_json(err_path, {"errors": local_errors})

    _barrier()

    if rank == 0:
        summary_rows: list[dict[str, Any]] = []
        valid_names: list[str] = []
        for ck_name in names:
            meta_path = os.path.join(out_dir, ck_name, "ckpt_meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            summary_rows.append(meta)
            valid_names.append(ck_name)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(out_dir, "checkpoint_summary.csv"), index=False)

        if summary_rows:
            rows = []
            for row in summary_rows:
                rows.append(
                    [
                        str(row.get("checkpoint", "")),
                        f"{float(row.get('score_mean', 0.0)):.6f}",
                        f"{float(row.get('score_std', 0.0)):.6f}",
                        str(int(row.get("num_draws", 0))),
                        str(int(row.get("pool_size", 0))),
                        str(int(row.get("query_size", 0))),
                    ]
                )
            log_table(
                "1_diag_global/checkpoint_summary",
                headers=["checkpoint", "score_mean", "score_std", "num_draws", "pool_size", "query_size"],
                rows=rows,
            )
            log_line(
                "1_diag_global/score_mean",
                xaxis=[str(r["checkpoint"]) for r in summary_rows],
                series={"score_mean": [float(r["score_mean"]) for r in summary_rows]},
                smooth=False,
            )

        traj_df = _make_global_trajectory(out_dir, valid_names, acfg.score_col)
        if traj_df is not None and len(valid_names) > 1:
            preview = traj_df.head(min(int(acfg.top_k or 50), 50))
            rows = []
            score_cols = [f"score__{ck}" for ck in valid_names]
            for i, (_, row) in enumerate(preview.iterrows(), 1):
                rows.append(
                    [i, str(row.get("sample_id", "")), str(row.get("source", ""))]
                    + [f"{float(row[c]):.6f}" if pd.notna(row[c]) else "" for c in score_cols]
                    + [f"{float(row.get('traj_mean', 0.0)):.6f}"]
                )
            log_table(
                "4_2_influence/trajectory/top",
                headers=["rank", "sample_id", "source"] + valid_names + ["traj_mean"],
                rows=rows,
            )

        save_json(
            os.path.join(out_dir, "analysis_config.json"),
            {
                "bif_root": bif_root,
                **acfg.__dict__,
                "checkpoint_names": names,
                "valid_checkpoint_names": valid_names,
                "world_size": world_size,
                "sequence_only": True,
            },
        )

    _barrier()
    if rank == 0 and manage_tracking:
        swan_finish()
    _barrier()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sequence-only BIF results (single or multi-GPU via torchrun)."
    )
    parser.add_argument("--bif_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--score_col", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--bottom_k", type=int, default=None)
    parser.add_argument("--save_full_query_matrix", action="store_true")
    parser.add_argument("--no_save_full_query_matrix", action="store_true")
    parser.add_argument("--negate_scores", action="store_true")
    parser.add_argument("--hist_bins", type=int, default=None)
    parser.add_argument("--scatter_max_points", type=int, default=None)
    parser.add_argument("--heatmap_max_pool", type=int, default=None)
    parser.add_argument("--heatmap_max_query", type=int, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    acfg = AnalyzeConfig()
    if args.score_col is not None:
        acfg.score_col = args.score_col
    if args.top_k is not None:
        acfg.top_k = args.top_k
    if args.bottom_k is not None:
        acfg.bottom_k = args.bottom_k
    if args.save_full_query_matrix:
        acfg.save_full_query_matrix = True
    if args.no_save_full_query_matrix:
        acfg.save_full_query_matrix = False
    if args.negate_scores:
        acfg.negate_scores = True
    if args.hist_bins is not None:
        acfg.hist_bins = args.hist_bins
    if args.scatter_max_points is not None:
        acfg.scatter_max_points = args.scatter_max_points
    if args.heatmap_max_pool is not None:
        acfg.heatmap_max_pool = args.heatmap_max_pool
    if args.heatmap_max_query is not None:
        acfg.heatmap_max_query = args.heatmap_max_query

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
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
