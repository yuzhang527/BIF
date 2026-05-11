"""Centralised SwanLab experiment naming for the BIF pipeline.

Every step (train, run-bif, analyze-bif, extract-top, schedule-compare,
schedule-analyze) uses the same ``{model_tag}-{step}-{key_params}`` pattern
so that experiments are self-describing and easy to find in the SwanLab UI.

Convention
----------
  {model_tag}-{step_abbreviation}-{key_hyperparams}

Examples
--------
  70m-train-lr2e-4-bs4-ep1
  70m-bif-ck300-lr2e-5-g0-d800-b40
  70m-analyze-corr-top500
  70m-extract-corr-top500
  70m-replay-mix-sel-rr01
  70m-sched-analysis
"""

from __future__ import annotations

import os
import re


def guess_model_tag(path: str) -> str:
    m = re.search(r"(\d{2,4}m)", path, re.IGNORECASE)
    return m.group(1).lower() if m else "model"


def fmt_lr(lr: float) -> str:
    if lr == 0:
        return "0"
    return re.sub(r"e([+-])0+(\d)", r"e\1\2", f"{lr:.0e}")


def fmt_ckpt_short(ckpt_name: str) -> str:
    if ckpt_name == "final_model":
        return "final"
    m = re.match(r"checkpoint-(\d+)", ckpt_name)
    if m:
        return f"ck{m.group(1)}"
    return ckpt_name


def make_train_name(model_tag: str, lr: float, bs: int, epochs: float) -> str:
    ep = str(epochs) if epochs != int(epochs) else str(int(epochs))
    return f"{model_tag}-train-lr{fmt_lr(lr)}-bs{bs}-ep{ep}"


def make_bif_name(
    model_tag: str, ckpt_name: str, lr: float, gamma: float, draws: int, burn_in: int,
    sampler_type: str = "sgld",
) -> str:
    prefix = "rmsgld" if sampler_type == "rmsprop_sgld" else "bif"
    return (
        f"{model_tag}-{prefix}-{fmt_ckpt_short(ckpt_name)}"
        f"-lr{fmt_lr(lr)}-g{gamma}-d{draws}-b{burn_in}"
    )


def make_bif_pipeline_name(
    model_tag: str, lr: float, gamma: float, draws: int, burn_in: int,
    sampler_type: str = "sgld",
) -> str:
    prefix = "rmsgld-pipe" if sampler_type == "rmsprop_sgld" else "bif-pipe"
    return f"{model_tag}-{prefix}-lr{fmt_lr(lr)}-g{gamma}-d{draws}-b{burn_in}"


def make_analyze_name(model_tag: str, score_col: str, top_k: int) -> str:
    sc_tag = "corr" if "corr" in score_col else ("raw" if "raw" in score_col else "score")
    return f"{model_tag}-analyze-{sc_tag}-top{top_k}"


def make_extract_name(model_tag: str, score_col: str, top_k: int) -> str:
    sc_tag = "corr" if "corr" in score_col else ("raw" if "raw" in score_col else "score")
    return f"{model_tag}-extract-{sc_tag}-top{top_k}"


def _fmt_ratio(r: float) -> str:
    if r == 0:
        return "0"
    if r == int(r):
        return str(int(r))
    return str(r).replace(".", "p")


def make_replay_name(
    model_tag: str,
    schedule: str,
    replay_mode: str,
    replay_ratio: float,
    score_type: str = "",
) -> str:
    sched = "seq" if schedule == "sequential" else "mix"
    mode = {"selected": "sel", "random": "rnd", "top_random": "trnd", "none": "none"}.get(
        replay_mode, replay_mode
    )
    name = f"{model_tag}-replay-{sched}-{mode}-rr{_fmt_ratio(replay_ratio)}"
    if score_type:
        name = f"{name}-{score_type}"
    return name


def make_replay_group_name(
    model_tag: str,
    replay_ratio: float,
    score_type: str = "",
) -> str:
    name = f"{model_tag}-replay-rr{_fmt_ratio(replay_ratio)}"
    if score_type:
        name = f"{name}-{score_type}"
    return name


def make_schedule_analysis_name(model_tag: str) -> str:
    return f"{model_tag}-sched-analysis"


def resolve_model_tag(explicit: str | None, base_model_path: str) -> str:
    if explicit:
        return explicit
    return guess_model_tag(base_model_path)
