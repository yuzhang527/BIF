"""Grid sweep runner for BIF sampler diagnostics.

The runner builds a grid over lr, gamma, and nbeta, launches existing
``run-bif`` and ``analyze-bif`` commands for each grid point, then runs the
new split/chain stability diagnostics from ``sweep_diagnostics``.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bif.analysis.sweep_diagnostics import (
    DiagnosticConfig,
    compare_score_files,
    diagnostic_config_from_dict,
    run_diagnostics_for_bif_root,
)
from bif.io import ensure_dir, save_json


RUN_BIF_VALUE_FLAGS = {
    "model_name_or_path",
    "model_root",
    "base_model_path",
    "tokenizer_path",
    "checkpoints",
    "pool_jsonl",
    "query_jsonl",
    "out_dir",
    "num_chains",
    "draws_per_chain",
    "max_length",
    "train_batch_size",
    "eval_batch_size",
    "pool_eval_subset",
    "lr",
    "gamma",
    "beta",
    "nbeta_mode",
    "nbeta",
    "noise_level",
    "num_burnin_steps",
    "num_steps_bw_draws",
    "seed",
    "grad_clip",
    "weight_decay",
    "sampler_type",
    "rmsprop_alpha",
    "rmsprop_eps",
    "batches_per_draw",
    "gradient_accumulation_steps",
    "chain_id",
    "device",
    "dtype",
    "experiment_name",
    "model_tag",
    "run_name",
}

RUN_BIF_BOOL_FLAGS = {
    "run_all_checkpoints",
    "resume",
}

ANALYZE_VALUE_FLAGS = {
    "bif_root",
    "out_dir",
    "score_col",
    "top_k",
    "experiment_name",
    "run_name",
    "hist_bins",
    "scatter_max_points",
    "heatmap_max_pool",
    "heatmap_max_query",
    "rhat_max_samples",
    "eigenvalue_max_pool",
    "eigenvalue_max_ev",
    "boxplot_max_sources",
    "boxplot_min_per_source",
    "rhat_min_draws",
    "chain_scatter_min_draws",
    "trajectory_top_n",
    "source_label_max_len",
    "heatmap_topk_max",
    "convergence_min_draws",
}

ANALYZE_LIST_FLAGS = {"convergence_checkpoints"}
ANALYZE_BOOL_FLAGS = {"enable_aux_query_plots", "negate_scores"}


@dataclass(frozen=True)
class SweepPoint:
    run_id: str
    lr: float
    gamma: float
    nbeta: float
    is_nbeta_zero: bool = False


@dataclass
class ExecutionConfig:
    run_bif: bool = True
    analyze_bif: bool = True
    diagnostics: bool = True
    skip_existing: bool = True
    continue_on_error: bool = True
    dry_run: bool = False
    python_executable: str = sys.executable
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class BaselineConfig:
    run_nbeta_zero: bool = True
    compare_to_nbeta_zero: bool = True
    include_if_in_grid: bool = True


@dataclass
class SweepConfig:
    base_run_config: str
    output_dir: str
    sweep: dict[str, Any]
    run_overrides: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


def _read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _resolve_path(path: str, base_dir: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in dict(patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _values_from_spec(spec: Any, name: str) -> list[float]:
    """Parse a sweep axis specification.

    Supported forms:
      * [1e-6, 3e-6]
      * {values: [...]}
      * {min: 1e-6, max: 1e-5, num: 3, scale: log|linear}
    """

    if isinstance(spec, (list, tuple)):
        values = list(spec)
    elif isinstance(spec, dict) and "values" in spec:
        values = list(spec["values"])
    elif isinstance(spec, dict) and {"min", "max", "num"}.issubset(spec):
        lo = float(spec["min"])
        hi = float(spec["max"])
        num = int(spec["num"])
        scale = str(spec.get("scale", "linear")).lower()
        if num <= 0:
            raise ValueError(f"sweep.{name}.num must be > 0")
        if scale == "log":
            if lo <= 0 or hi <= 0:
                raise ValueError(f"log sweep for {name} requires positive min/max")
            values = np.logspace(np.log10(lo), np.log10(hi), num=num).tolist()
        elif scale == "linear":
            values = np.linspace(lo, hi, num=num).tolist()
        else:
            raise ValueError(f"Unknown sweep.{name}.scale={scale!r}")
    else:
        raise ValueError(f"Invalid sweep axis for {name}: {spec!r}")
    if not values:
        raise ValueError(f"sweep.{name} must contain at least one value")
    return [float(v) for v in values]


def _float_token(value: float) -> str:
    text = f"{float(value):.10g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _run_id(prefix: str, index: int, lr: float, gamma: float, nbeta: float) -> str:
    return (
        f"{prefix}_{index:04d}"
        f"_lr{_float_token(lr)}"
        f"_gamma{_float_token(gamma)}"
        f"_nbeta{_float_token(nbeta)}"
    )


def _build_plan(raw_cfg: SweepConfig) -> list[SweepPoint]:
    sweep = raw_cfg.sweep
    for axis in ("lr", "gamma", "nbeta"):
        if axis not in sweep:
            raise ValueError(f"Missing sweep.{axis}")
    lrs = _values_from_spec(sweep["lr"], "lr")
    gammas = _values_from_spec(sweep["gamma"], "gamma")
    nbetas = _values_from_spec(sweep["nbeta"], "nbeta")

    points: list[SweepPoint] = []
    seen: set[tuple[float, float, float]] = set()
    idx = 0
    for lr, gamma, nbeta in itertools.product(lrs, gammas, nbetas):
        key = (float(lr), float(gamma), float(nbeta))
        if key in seen:
            continue
        seen.add(key)
        points.append(
            SweepPoint(
                run_id=_run_id("grid", idx, lr, gamma, nbeta),
                lr=lr,
                gamma=gamma,
                nbeta=nbeta,
                is_nbeta_zero=(float(nbeta) == 0.0),
            )
        )
        idx += 1

    if raw_cfg.baseline.run_nbeta_zero:
        for lr, gamma in itertools.product(lrs, gammas):
            key = (float(lr), float(gamma), 0.0)
            if key in seen:
                if raw_cfg.baseline.include_if_in_grid:
                    continue
                continue
            seen.add(key)
            points.append(
                SweepPoint(
                    run_id=_run_id("nbeta0", idx, lr, gamma, 0.0),
                    lr=lr,
                    gamma=gamma,
                    nbeta=0.0,
                    is_nbeta_zero=True,
                )
            )
            idx += 1
    return points


def _payload_to_cli_args(payload: dict[str, Any], value_flags: set[str], bool_flags: set[str], list_flags: set[str] | None = None) -> list[str]:
    list_flags = list_flags or set()
    args: list[str] = []
    for key in sorted(value_flags):
        value = payload.get(key)
        if value is None:
            continue
        args.extend([f"--{key}", str(value)])
    for key in sorted(bool_flags):
        if bool(payload.get(key, False)):
            args.append(f"--{key}")
    for key in sorted(list_flags):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            value = [value]
        args.append(f"--{key}")
        args.extend(str(v) for v in value)
    return args


def _run_subprocess(cmd: list[str], cwd: str | None, env: dict[str, str], log_path: str, dry_run: bool = False) -> int:
    ensure_dir(str(Path(log_path).parent))
    printable = " ".join(cmd)
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n$ {printable}\n")
        log_f.flush()
        if dry_run:
            log_f.write("[dry-run] command not executed\n")
            return 0
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_f.write(f"[exit_code] {proc.returncode}\n")
        return int(proc.returncode)


def _is_done(run_dir: str) -> bool:
    manifest = os.path.join(run_dir, "sweep_run_manifest.json")
    if not os.path.isfile(manifest):
        return False
    try:
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == "done"
    except Exception:
        return False


def _analysis_payload(base: dict[str, Any], point: SweepPoint, run_dir: str, run_name: str) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    payload["bif_root"] = os.path.join(run_dir, "traces")
    payload["out_dir"] = os.path.join(run_dir, "analysis")
    payload.setdefault("run_name", run_name)
    return payload


def _run_payload(base: dict[str, Any], point: SweepPoint, run_dir: str, run_name: str) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    payload.update({"lr": point.lr, "gamma": point.gamma, "nbeta": point.nbeta})
    payload["out_dir"] = os.path.join(run_dir, "traces")
    payload.setdefault("run_name", run_name)
    return payload


def _score_file(run_dir: str, checkpoint: str, score_file_name: str = "pool_scores_diagnostic.csv") -> str:
    return os.path.join(run_dir, "diagnostics", checkpoint, score_file_name)


def _pick_summary_checkpoint(row: pd.Series, diagnostics_cfg: DiagnosticConfig) -> str:
    if diagnostics_cfg.checkpoint:
        return diagnostics_cfg.checkpoint
    ckpt = row.get("checkpoint")
    if isinstance(ckpt, str) and ckpt:
        return ckpt
    return "final_model"


def _add_baseline_comparisons(
    summary: pd.DataFrame,
    output_dir: str,
    diagnostics_cfg: DiagnosticConfig,
    baseline_cfg: BaselineConfig,
) -> pd.DataFrame:
    if summary.empty or not baseline_cfg.compare_to_nbeta_zero:
        return summary
    score_col = diagnostics_cfg.split_stability.score_col
    top_k = diagnostics_cfg.split_stability.top_k
    out = summary.copy()

    baseline_rows = out[out["is_nbeta_zero"].astype(bool)].copy()
    baseline_lookup: dict[tuple[float, float], pd.Series] = {}
    for _, brow in baseline_rows.iterrows():
        baseline_lookup[(float(brow["lr"]), float(brow["gamma"]))] = brow

    compare_rows: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        key = (float(row["lr"]), float(row["gamma"]))
        brow = baseline_lookup.get(key)
        if brow is None or bool(row["is_nbeta_zero"]):
            continue
        checkpoint = _pick_summary_checkpoint(row, diagnostics_cfg)
        cand_csv = _score_file(str(row["run_dir"]), checkpoint)
        base_csv = _score_file(str(brow["run_dir"]), checkpoint)
        if not os.path.isfile(cand_csv) or not os.path.isfile(base_csv):
            continue
        try:
            metrics = compare_score_files(cand_csv, base_csv, score_col, top_k)
        except Exception as exc:
            metrics = {"baseline_compare_error": str(exc)}
        metrics.update({"run_id": row["run_id"], "baseline_run_id": brow["run_id"]})
        compare_rows.append(metrics)
        for mkey, mval in metrics.items():
            if mkey not in {"run_id", "baseline_run_id"}:
                out.loc[idx, mkey] = mval
        out.loc[idx, "baseline_run_id"] = brow["run_id"]

    if compare_rows:
        pd.DataFrame(compare_rows).to_csv(os.path.join(output_dir, "baseline_comparisons.csv"), index=False)
    return out


def _load_sweep_config(config_path: str, cli_output_dir: str | None = None) -> tuple[SweepConfig, dict[str, Any], str]:
    raw = _read_yaml(config_path)
    base_dir = str(Path(config_path).resolve().parent)
    if "base_run_config" not in raw:
        raise ValueError("sweep config must define base_run_config")
    if "sweep" not in raw:
        raise ValueError("sweep config must define sweep")
    raw["base_run_config"] = _resolve_path(str(raw["base_run_config"]), base_dir)
    if cli_output_dir is not None:
        raw["output_dir"] = cli_output_dir
    if "output_dir" not in raw:
        raise ValueError("sweep config must define output_dir or CLI must pass --out_dir")
    raw["output_dir"] = _resolve_path(str(raw["output_dir"]), base_dir)

    baseline = BaselineConfig(**raw.get("baseline", {}) or {})
    execution = ExecutionConfig(**raw.get("execution", {}) or {})
    cfg = SweepConfig(
        base_run_config=raw["base_run_config"],
        output_dir=raw["output_dir"],
        sweep=raw["sweep"],
        run_overrides=raw.get("run_overrides", {}) or {},
        analysis=raw.get("analysis", {}) or {},
        diagnostics=raw.get("diagnostics", {}) or {},
        baseline=baseline,
        execution=execution,
    )
    return cfg, raw, base_dir


def run_sweep(config_path: str, out_dir: str | None = None, dry_run: bool | None = None) -> pd.DataFrame:
    cfg, raw_cfg, _ = _load_sweep_config(config_path, out_dir)
    if dry_run is not None:
        cfg.execution.dry_run = bool(dry_run)

    ensure_dir(cfg.output_dir)
    ensure_dir(os.path.join(cfg.output_dir, "runs"))
    _write_yaml(os.path.join(cfg.output_dir, "sweep_config_resolved.yaml"), raw_cfg)

    base_run = _read_yaml(cfg.base_run_config)
    base_run = _deep_update(base_run, cfg.run_overrides)

    diagnostics_cfg = diagnostic_config_from_dict(cfg.diagnostics)
    if "score_col" in cfg.analysis and not cfg.diagnostics.get("split_stability", {}).get("score_col"):
        diagnostics_cfg.split_stability.score_col = str(cfg.analysis["score_col"])
        diagnostics_cfg.chain_stability.score_col = str(cfg.analysis["score_col"])
    if cfg.analysis.get("negate_scores") is not None:
        diagnostics_cfg.negate_scores = bool(cfg.analysis.get("negate_scores"))

    plan = _build_plan(cfg)
    plan_df = pd.DataFrame([asdict(p) for p in plan])
    plan_df.to_csv(os.path.join(cfg.output_dir, "sweep_plan.csv"), index=False)

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in cfg.execution.extra_env.items()})
    rows: list[dict[str, Any]] = []

    for idx, point in enumerate(plan):
        run_dir = os.path.join(cfg.output_dir, "runs", point.run_id)
        ensure_dir(run_dir)
        log_path = os.path.join(run_dir, "commands.log")
        manifest_path = os.path.join(run_dir, "sweep_run_manifest.json")
        started = time.time()

        if cfg.execution.skip_existing and _is_done(run_dir):
            with open(manifest_path, encoding="utf-8") as f:
                existing = json.load(f)
            rows.append(existing)
            continue

        status = "done"
        error = ""
        run_payload = _run_payload(base_run, point, run_dir, point.run_id)
        analysis_payload = _analysis_payload(cfg.analysis, point, run_dir, point.run_id)
        _write_yaml(os.path.join(run_dir, "run_config.yaml"), run_payload)
        _write_yaml(os.path.join(run_dir, "analysis_config.yaml"), analysis_payload)

        try:
            if cfg.execution.run_bif:
                cmd = [cfg.execution.python_executable, "-m", "bif.cli", "run-bif"]
                cmd.extend(_payload_to_cli_args(run_payload, RUN_BIF_VALUE_FLAGS, RUN_BIF_BOOL_FLAGS))
                code = _run_subprocess(cmd, cwd=None, env=env, log_path=log_path, dry_run=cfg.execution.dry_run)
                if code != 0:
                    raise RuntimeError(f"run-bif failed with exit code {code}")

            if cfg.execution.analyze_bif:
                cmd = [cfg.execution.python_executable, "-m", "bif.cli", "analyze-bif"]
                cmd.extend(
                    _payload_to_cli_args(
                        analysis_payload,
                        ANALYZE_VALUE_FLAGS,
                        ANALYZE_BOOL_FLAGS,
                        ANALYZE_LIST_FLAGS,
                    )
                )
                code = _run_subprocess(cmd, cwd=None, env=env, log_path=log_path, dry_run=cfg.execution.dry_run)
                if code != 0:
                    raise RuntimeError(f"analyze-bif failed with exit code {code}")

            diag_rows = pd.DataFrame()
            if cfg.execution.diagnostics and not cfg.execution.dry_run:
                diag_rows = run_diagnostics_for_bif_root(
                    bif_root=run_payload["out_dir"],
                    out_dir=os.path.join(run_dir, "diagnostics"),
                    cfg=diagnostics_cfg,
                )
            elif cfg.execution.diagnostics and cfg.execution.dry_run:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("[dry-run] diagnostics not executed\n")

        except Exception as exc:
            status = "error"
            error = str(exc)
            if not cfg.execution.continue_on_error:
                raise

        elapsed = time.time() - started
        row: dict[str, Any] = {
            "run_id": point.run_id,
            "run_dir": run_dir,
            "lr": point.lr,
            "gamma": point.gamma,
            "nbeta": point.nbeta,
            "is_nbeta_zero": point.is_nbeta_zero,
            "status": status,
            "error": error,
            "elapsed_sec": elapsed,
        }
        diag_csv = os.path.join(run_dir, "diagnostics", "diagnostics_summary.csv")
        if status == "done" and os.path.isfile(diag_csv):
            diag_df = pd.read_csv(diag_csv)
            if not diag_df.empty:
                diag_row = diag_df.iloc[0].to_dict()
                row.update({k: v for k, v in diag_row.items() if k not in row})
        rows.append(row)
        save_json(manifest_path, row)

        pd.DataFrame(rows).to_csv(os.path.join(cfg.output_dir, "sweep_summary.partial.csv"), index=False)

    summary = pd.DataFrame(rows)
    if not cfg.execution.dry_run:
        summary = _add_baseline_comparisons(summary, cfg.output_dir, diagnostics_cfg, cfg.baseline)
    summary.to_csv(os.path.join(cfg.output_dir, "sweep_summary.csv"), index=False)
    save_json(
        os.path.join(cfg.output_dir, "sweep_summary_meta.json"),
        {
            "num_runs": int(len(summary)),
            "num_done": int((summary.get("status") == "done").sum()) if "status" in summary else 0,
            "num_error": int((summary.get("status") == "error").sum()) if "status" in summary else 0,
            "diagnostics": asdict(diagnostics_cfg),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a grid sweep over BIF SGLD parameters.")
    parser.add_argument("--config", required=True, help="YAML sweep config")
    parser.add_argument("--out_dir", default=None, help="Override output_dir in config")
    parser.add_argument("--dry_run", action="store_true", help="Write the run plan without launching commands")
    args = parser.parse_args(argv)
    summary = run_sweep(args.config, out_dir=args.out_dir, dry_run=args.dry_run if args.dry_run else None)
    print(summary.tail(min(20, len(summary))).to_string(index=False))


if __name__ == "__main__":
    main()
