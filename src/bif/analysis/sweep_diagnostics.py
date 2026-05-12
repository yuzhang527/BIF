"""Diagnostics for BIF sampler hyperparameter sweeps.

This module intentionally reuses the existing BIF trace loader and scorer from
``bif.analysis.bif_analyzer``.  It does not run SGLD and it does not duplicate
SwanLab loss-trace visualization; it focuses on stability diagnostics that are
hard to read from trace plots alone.  When a SwanLab run has already been
initialized by the surrounding pipeline, the diagnostic summaries, overlap
curves, correlation curves, and tables are also logged for visualization.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bif.analysis.bif_analyzer import (
    compute_bif_scores,
    discover_checkpoint_dirs,
    load_checkpoint_traces,
)
from bif.io import ensure_dir, save_json
from bif.utils.tracker import log as swan_log
from bif.utils.tracker import log_bar, log_line, log_table


@dataclass
class SplitStabilityConfig:
    """Config for random draw half-split stability checks."""

    enabled: bool = True
    num_splits: int = 20
    split_fraction: float = 0.5
    top_k: list[int] = field(default_factory=lambda: [50, 100, 500])
    score_col: str = "bif_mean"
    pass_threshold: float = 0.9
    seed: int = 42
    min_draws: int = 8


@dataclass
class ChainStabilityConfig:
    """Config for comparing rankings obtained from individual chains."""

    enabled: bool = True
    top_k: list[int] = field(default_factory=lambda: [50, 100, 500])
    score_col: str = "bif_mean"
    min_draws_per_chain: int = 4


@dataclass
class DiagnosticConfig:
    """Diagnostics computed after trace collection.

    ``reduce_chains`` is forced to stack for split diagnostics, because random
    draw splits destroy contiguous chain structure.  It is kept as a config
    field for final full-draw scoring and for consistency with analyzer knobs.
    """

    split_stability: SplitStabilityConfig = field(default_factory=SplitStabilityConfig)
    chain_stability: ChainStabilityConfig = field(default_factory=ChainStabilityConfig)
    reduce_chains: str = "stack"
    negate_scores: bool = False
    checkpoint: str | None = None


def diagnostic_config_from_dict(raw: dict[str, Any] | None) -> DiagnosticConfig:
    """Build ``DiagnosticConfig`` from a plain YAML dict."""

    raw = dict(raw or {})
    split_raw = raw.pop("split_stability", None) or {}
    chain_raw = raw.pop("chain_stability", None) or {}

    split = SplitStabilityConfig(**_filter_dataclass_kwargs(SplitStabilityConfig, split_raw))
    chain = ChainStabilityConfig(**_filter_dataclass_kwargs(ChainStabilityConfig, chain_raw))
    cfg = DiagnosticConfig(
        split_stability=split,
        chain_stability=chain,
        **_filter_dataclass_kwargs(DiagnosticConfig, raw),
    )
    return cfg


def _filter_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    fields = set(getattr(cls, "__dataclass_fields__", {}).keys())
    return {k: v for k, v in dict(values or {}).items() if k in fields}


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.array([], dtype=np.int64)
    idx = np.where(finite)[0]
    k_eff = min(int(k), len(idx))
    if k_eff <= 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-scores[idx], kind="mergesort")[:k_eff]
    return idx[order]


def topk_overlap_metrics(scores_a: np.ndarray, scores_b: np.ndarray, top_k: list[int]) -> dict[str, float]:
    """Return overlap-recall and Jaccard metrics for several K values."""

    out: dict[str, float] = {}
    for k in top_k:
        a = set(map(int, _top_indices(scores_a, k)))
        b = set(map(int, _top_indices(scores_b, k)))
        inter = len(a & b)
        denom = min(int(k), len(a), len(b))
        union = len(a | b)
        out[f"top{k}_overlap_recall"] = float(inter / denom) if denom > 0 else float("nan")
        out[f"top{k}_jaccard"] = float(inter / union) if union > 0 else float("nan")
    return out


def _corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    if method == "spearman":
        xs = pd.Series(xs).rank(method="average").to_numpy()
        ys = pd.Series(ys).rank(method="average").to_numpy()
    if np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _score_from_traces(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    score_col: str,
    num_chains: int = 1,
    reduce_chains: str = "stack",
    negate_scores: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scores = compute_bif_scores(
        pool_seq_loss=pool_seq_loss,
        query_seq_loss=query_seq_loss,
        num_chains=num_chains,
        reduce_chains=reduce_chains,
        negate_scores=negate_scores,
    )
    if score_col not in scores:
        available = ", ".join(sorted(scores.keys()))
        raise KeyError(f"score_col={score_col!r} not found in BIF scores. Available: {available}")
    return np.asarray(scores[score_col]), scores


def compute_split_stability(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    cfg: SplitStabilityConfig,
    negate_scores: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Repeatedly split chain-draw observations and compare top-K rankings."""

    num_draws = int(pool_seq_loss.shape[0])
    if num_draws != int(query_seq_loss.shape[0]):
        raise ValueError("pool and query traces must have the same number of draws")
    if num_draws < cfg.min_draws:
        empty = pd.DataFrame()
        summary = {
            "split_enabled": float(cfg.enabled),
            "split_num_draws": float(num_draws),
            "split_skipped_min_draws": 1.0,
        }
        return empty, summary

    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    split_size = int(round(num_draws * float(cfg.split_fraction)))
    split_size = max(2, min(num_draws - 2, split_size))

    for split_id in range(int(cfg.num_splits)):
        perm = rng.permutation(num_draws)
        idx_a = np.sort(perm[:split_size])
        idx_b = np.sort(perm[split_size:])

        score_a, _ = _score_from_traces(
            pool_seq_loss[idx_a],
            query_seq_loss[idx_a],
            cfg.score_col,
            num_chains=1,
            reduce_chains="stack",
            negate_scores=negate_scores,
        )
        score_b, _ = _score_from_traces(
            pool_seq_loss[idx_b],
            query_seq_loss[idx_b],
            cfg.score_col,
            num_chains=1,
            reduce_chains="stack",
            negate_scores=negate_scores,
        )
        row: dict[str, Any] = {
            "split_id": split_id,
            "n_draws_a": int(len(idx_a)),
            "n_draws_b": int(len(idx_b)),
            "score_pearson": _corr(score_a, score_b, "pearson"),
            "score_spearman": _corr(score_a, score_b, "spearman"),
        }
        row.update(topk_overlap_metrics(score_a, score_b, cfg.top_k))
        rows.append(row)

    df = pd.DataFrame(rows)
    summary: dict[str, float] = {
        "split_enabled": 1.0,
        "split_num_draws": float(num_draws),
        "split_num_splits": float(cfg.num_splits),
    }
    for col in df.columns:
        if col in {"split_id", "n_draws_a", "n_draws_b"}:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        summary[f"split_{col}_mean"] = _safe_float(vals.mean())
        summary[f"split_{col}_std"] = _safe_float(vals.std(ddof=0))
        summary[f"split_{col}_min"] = _safe_float(vals.min())
    if cfg.top_k:
        primary = f"split_top{int(cfg.top_k[0])}_overlap_recall_mean"
        if primary in summary:
            summary["split_pass"] = float(summary[primary] >= cfg.pass_threshold)
    return df, summary


def compute_chain_stability(
    pool_seq_loss: np.ndarray,
    query_seq_loss: np.ndarray,
    num_chains: int,
    draws_per_chain: int,
    cfg: ChainStabilityConfig,
    negate_scores: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare BIF rankings produced by individual chains."""

    if num_chains <= 1:
        return pd.DataFrame(), {"chain_enabled": float(cfg.enabled), "chain_skipped_single_chain": 1.0}
    if draws_per_chain < cfg.min_draws_per_chain:
        return pd.DataFrame(), {
            "chain_enabled": float(cfg.enabled),
            "chain_skipped_min_draws": 1.0,
            "chain_draws_per_chain": float(draws_per_chain),
        }

    chain_scores: list[np.ndarray] = []
    for c in range(num_chains):
        start = c * draws_per_chain
        end = start + draws_per_chain
        score, _ = _score_from_traces(
            pool_seq_loss[start:end],
            query_seq_loss[start:end],
            cfg.score_col,
            num_chains=1,
            reduce_chains="stack",
            negate_scores=negate_scores,
        )
        chain_scores.append(score)

    rows: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(num_chains), 2):
        row: dict[str, Any] = {
            "chain_i": i,
            "chain_j": j,
            "score_pearson": _corr(chain_scores[i], chain_scores[j], "pearson"),
            "score_spearman": _corr(chain_scores[i], chain_scores[j], "spearman"),
        }
        row.update(topk_overlap_metrics(chain_scores[i], chain_scores[j], cfg.top_k))
        rows.append(row)

    df = pd.DataFrame(rows)
    summary: dict[str, float] = {
        "chain_enabled": 1.0,
        "chain_num_chains": float(num_chains),
        "chain_draws_per_chain": float(draws_per_chain),
    }
    for col in df.columns:
        if col in {"chain_i", "chain_j"}:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        summary[f"chain_{col}_mean"] = _safe_float(vals.mean())
        summary[f"chain_{col}_std"] = _safe_float(vals.std(ddof=0))
        summary[f"chain_{col}_min"] = _safe_float(vals.min())
    return df, summary


def _make_score_dataframe(pool_ids: list[Any], scores: dict[str, np.ndarray]) -> pd.DataFrame:
    data: dict[str, Any] = {"sample_id": list(pool_ids)}
    n = len(pool_ids)
    for key, value in scores.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == n:
            data[key] = arr.astype(float)
    return pd.DataFrame(data)


def _trace_summary(pool_seq_loss: np.ndarray, query_seq_loss: np.ndarray) -> dict[str, float]:
    return {
        "num_draws": float(pool_seq_loss.shape[0]),
        "pool_size": float(pool_seq_loss.shape[1]),
        "query_size": float(query_seq_loss.shape[1]),
        "pool_loss_mean": _safe_float(np.nanmean(pool_seq_loss)),
        "pool_loss_std": _safe_float(np.nanstd(pool_seq_loss)),
        "query_loss_mean": _safe_float(np.nanmean(query_seq_loss)),
        "query_loss_std": _safe_float(np.nanstd(query_seq_loss)),
    }



def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _finite_or_none(x: Any) -> float | None:
    try:
        value = float(x)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _metric_list(df: pd.DataFrame, col: str) -> list[float | None]:
    return [_finite_or_none(x) for x in df[col].tolist()]


def _safe_str_list(values: list[Any]) -> list[str]:
    return [str(v) for v in values]


def _line_metric_columns(df: pd.DataFrame) -> list[str]:
    """Only plot top-K overlap metrics.

    This keeps SwanLab clean: one compact line chart per diagnostic, with
    different K values as different series.
    """

    if df.empty:
        return []

    return [
        col
        for col in df.columns
        if col.endswith("_overlap_recall")
    ]



def _summary_metric_columns(df: pd.DataFrame) -> list[str]:
    """Overview columns worth plotting across sweep checkpoints."""

    if df.empty:
        return []
    metric_cols: list[str] = []
    for col in df.columns:
        if col in {"checkpoint", "checkpoint_dir"}:
            continue
        if (
            col.endswith("_pass")
            or col.endswith("_mean")
            or col.endswith("_min")
            or col.endswith("_std")
            or col in {"num_chains", "draws_per_chain", "num_draws", "pool_size", "query_size"}
        ):
            metric_cols.append(col)
    return metric_cols


def _records_for_table(df: pd.DataFrame, max_rows: int = 80) -> tuple[list[str], list[list[Any]]]:
    if df.empty:
        return [], []
    shown = df.head(max_rows).copy()
    shown = shown.replace([np.inf, -np.inf], np.nan)
    shown = shown.where(pd.notnull(shown), "")
    headers = list(shown.columns)
    rows = shown.astype(str).values.tolist()
    return headers, rows


def _log_detail_curves(
    checkpoint_name: str,
    diagnostic_name: str,
    df: pd.DataFrame,
) -> None:
    """Log all split/chain overlap values in one compact chart.

    For split_stability:
      x-axis = split_id
      series = topK_overlap_recall

    For chain_stability:
      x-axis = chain pair, e.g. 0-1
      series = topK_overlap_recall
    """

    if not _is_rank0() or df.empty:
        return

    if diagnostic_name == "split_stability" and "split_id" in df.columns:
        xaxis = _safe_str_list(df["split_id"].tolist())
        chart_name = "split_overlap_recall"
    elif {"chain_i", "chain_j"}.issubset(df.columns):
        xaxis = [f"{int(i)}-{int(j)}" for i, j in zip(df["chain_i"], df["chain_j"])]
        chart_name = "chain_overlap_recall"
    else:
        xaxis = [str(i) for i in range(len(df))]
        chart_name = "overlap_recall"

    series: dict[str, list[float | None]] = {}
    for col in _line_metric_columns(df):
        series[col] = _metric_list(df, col)

    if series:
        log_line(
            f"6_sweep_diagnostics/{checkpoint_name}/{diagnostic_name}/{chart_name}",
            xaxis=xaxis,
            series=series,
            smooth=False,
        )



def _log_checkpoint_summary(
    checkpoint_name: str,
    summary: dict[str, Any],
    split_df: pd.DataFrame,
    chain_df: pd.DataFrame,
) -> None:
    """Log only split/chain top-K overlap mean diagnostics to SwanLab.

    This intentionally avoids logging:
      - per-split curves
      - per-chain-pair curves
      - tables
      - jaccard metrics
      - pearson/spearman metrics
      - std/min/pass diagnostics

    Expected chart keys:
      6_sweep_diagnostics/{checkpoint}/summary/split_top150_overlap_recall_mean
      6_sweep_diagnostics/{checkpoint}/summary/chain_top150_overlap_recall_mean
    """

    if not _is_rank0():
        return

    wanted_keys = [
        key
        for key in summary.keys()
        if (
            key.startswith("split_top")
            or key.startswith("chain_top")
        )
        and key.endswith("_overlap_recall_mean")
    ]

    def _sort_key(key: str) -> tuple[int, int]:
        # split first, chain second
        kind_order = 0 if key.startswith("split_") else 1

        # Extract K from keys like:
        #   split_top150_overlap_recall_mean
        #   chain_top150_overlap_recall_mean
        try:
            top_part = key.split("_top", 1)[1]
            k = int(top_part.split("_", 1)[0])
        except Exception:
            k = 10**9

        return kind_order, k

    wanted_keys = sorted(wanted_keys, key=_sort_key)

    for key in wanted_keys:
        finite = _finite_or_none(summary.get(key))
        if finite is None:
            continue

        log_bar(
            f"6_sweep_diagnostics/{checkpoint_name}/summary/{key}",
            xaxis=[checkpoint_name],
            series={key: [finite]},
        )




def _log_diagnostics_overview(df: pd.DataFrame) -> None:
    """Do not log sweep-wide diagnostics overview to SwanLab.

    The CSV summary is still written to disk. We skip SwanLab overview charts
    to avoid clutter.
    """
    return



def run_diagnostics_for_checkpoint(
    checkpoint_name: str,
    checkpoint_dir: str,
    out_dir: str,
    cfg: DiagnosticConfig,
) -> dict[str, Any]:
    """Run diagnostics for one checkpoint trace directory.

    Besides writing CSV/JSON artifacts, this function logs diagnostic scalars,
    curves, and tables to SwanLab when the surrounding process has initialized a
    SwanLab run.  Logging is intentionally best-effort via bif.utils.tracker.
    """

    ensure_dir(out_dir)
    traces = load_checkpoint_traces(checkpoint_dir)
    pool_seq = np.asarray(traces["pool_seq_loss"], dtype=np.float64)
    query_seq = np.asarray(traces["query_seq_loss"], dtype=np.float64)
    num_chains = int(traces.get("num_chains", 1))
    draws_per_chain = int(traces.get("draws_per_chain", pool_seq.shape[0] // max(num_chains, 1)))

    summary: dict[str, Any] = {
        "checkpoint": checkpoint_name,
        "checkpoint_dir": checkpoint_dir,
        "num_chains": num_chains,
        "draws_per_chain": draws_per_chain,
    }
    summary.update(_trace_summary(pool_seq, query_seq))

    score_col = cfg.split_stability.score_col
    full_score, full_scores = _score_from_traces(
        pool_seq,
        query_seq,
        score_col=score_col,
        num_chains=num_chains,
        reduce_chains=cfg.reduce_chains,
        negate_scores=cfg.negate_scores,
    )
    score_df = _make_score_dataframe(traces["pool_ids"], full_scores)
    score_df.to_csv(os.path.join(out_dir, "pool_scores_diagnostic.csv"), index=False)
    summary[f"{score_col}_mean"] = _safe_float(np.nanmean(full_score))
    summary[f"{score_col}_std"] = _safe_float(np.nanstd(full_score))

    split_df = pd.DataFrame()
    chain_df = pd.DataFrame()

    if cfg.split_stability.enabled:
        split_df, split_summary = compute_split_stability(
            pool_seq,
            query_seq,
            cfg.split_stability,
            negate_scores=cfg.negate_scores,
        )
        if not split_df.empty:
            split_df.to_csv(os.path.join(out_dir, "split_stability.csv"), index=False)
        summary.update(split_summary)
    else:
        summary["split_enabled"] = 0.0

    if cfg.chain_stability.enabled:
        chain_df, chain_summary = compute_chain_stability(
            pool_seq,
            query_seq,
            num_chains,
            draws_per_chain,
            cfg.chain_stability,
            negate_scores=cfg.negate_scores,
        )
        if not chain_df.empty:
            chain_df.to_csv(os.path.join(out_dir, "chain_stability.csv"), index=False)
        summary.update(chain_summary)
    else:
        summary["chain_enabled"] = 0.0

    save_json(os.path.join(out_dir, "diagnostics_summary.json"), summary)

    _log_checkpoint_summary(
        checkpoint_name=checkpoint_name,
        summary=summary,
        split_df=split_df,
        chain_df=chain_df,
    )

    return summary


def run_diagnostics_for_bif_root(
    bif_root: str,
    out_dir: str,
    cfg: DiagnosticConfig | dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run diagnostics for all checkpoint dirs under a BIF trace root."""

    if not isinstance(cfg, DiagnosticConfig):
        cfg = diagnostic_config_from_dict(cfg if isinstance(cfg, dict) else None)
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, "diagnostics_config.json"), asdict(cfg))

    rows: list[dict[str, Any]] = []
    for checkpoint_name, checkpoint_dir in discover_checkpoint_dirs(bif_root):
        if cfg.checkpoint is not None and checkpoint_name != cfg.checkpoint:
            continue
        ckpt_out = os.path.join(out_dir, checkpoint_name)
        rows.append(run_diagnostics_for_checkpoint(checkpoint_name, checkpoint_dir, ckpt_out, cfg))

    if not rows:
        raise ValueError(f"No diagnostics were run under {bif_root}; checkpoint filter={cfg.checkpoint!r}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "diagnostics_summary.csv"), index=False)

    _log_diagnostics_overview(df)

    return df


def compare_score_files(
    candidate_csv: str,
    baseline_csv: str,
    score_col: str,
    top_k: list[int],
) -> dict[str, float]:
    """Compare one run's diagnostic scores with a matched nbeta=0 baseline."""

    cand = pd.read_csv(candidate_csv)
    base = pd.read_csv(baseline_csv)
    if "sample_id" not in cand.columns or "sample_id" not in base.columns:
        raise KeyError("Both score files must contain sample_id")
    if score_col not in cand.columns or score_col not in base.columns:
        raise KeyError(f"Both score files must contain score_col={score_col!r}")
    merged = cand[["sample_id", score_col]].merge(
        base[["sample_id", score_col]],
        on="sample_id",
        suffixes=("_candidate", "_baseline"),
    )
    x = merged[f"{score_col}_candidate"].to_numpy(dtype=float)
    y = merged[f"{score_col}_baseline"].to_numpy(dtype=float)
    out = {
        "baseline_n_matched": float(len(merged)),
        "baseline_score_pearson": _corr(x, y, "pearson"),
        "baseline_score_spearman": _corr(x, y, "spearman"),
        "baseline_mean_abs_diff": _safe_float(np.nanmean(np.abs(x - y))),
    }
    overlaps = topk_overlap_metrics(x, y, top_k)
    out.update({f"baseline_{k}": v for k, v in overlaps.items()})
    return out


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run BIF sweep diagnostics for an existing BIF trace root.")
    parser.add_argument("--bif_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default=None, help="Optional diagnostics YAML config")
    args = parser.parse_args()

    raw_cfg = None
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
    run_diagnostics_for_bif_root(args.bif_root, args.out_dir, raw_cfg)
