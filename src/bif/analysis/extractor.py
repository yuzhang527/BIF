"""Extract top-influence samples from BIF analysis results."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from bif.io import ensure_dir, read_jsonl, write_jsonl
from bif.utils.naming import guess_model_tag, make_extract_name
from bif.utils.tracker import finish as swan_finish
from bif.utils.tracker import init_run, log_bar, log_heatmap, log_line, log_table


def _infer_score_col(df: pd.DataFrame, user_col: str | None = None) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(f"{user_col} not in {df.columns.tolist()}")
        return user_col
    for c in [
        "traj_mean",
        "corr_mean_over_queries",
        "corr_absmean_over_queries",
        "raw_cov_avg_over_queries",
        "emergence_last_minus_first",
    ]:
        if c in df.columns:
            return c
    numeric = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "rank"
    ]
    if not numeric:
        raise ValueError("Could not infer score column")
    return numeric[0]


def _fmt_text(text: str) -> str:
    return str(text).strip().replace("\n", " ")


def extract_top_samples(
    pool_jsonl: str,
    ranking_csv: str,
    out_dir: str,
    id_col: str = "sample_id",
    source_col: str = "source",
    text_col: str = "text",
    score_col: str | None = None,
    top_k: int = 500,
    top_n_per_source: int = 3,
    preview_chars: int = 2000,
    restrict_source_topn_to_topk: bool = False,
    ascending: bool = False,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Extract top-influence samples and per-source representatives.
    
    Args:
        ascending: If True, sort ascending (select bottom-K / most harmful).
    """
    ensure_dir(out_dir)

    auto_name = make_extract_name(
        guess_model_tag(pool_jsonl), score_col or "score", top_k,
    )
    init_run(
        experiment_name=experiment_name or auto_name,
        run_name=run_name,
        config={
            "pool_jsonl": pool_jsonl,
            "ranking_csv": ranking_csv,
            "score_col": score_col,
            "top_k": top_k,
            "top_n_per_source": top_n_per_source,
            "ascending": ascending,
        },
        tags=["extraction"],
    )

    ranking_df = pd.read_csv(ranking_csv)
    score_col = _infer_score_col(ranking_df, score_col)
    ranking_df = ranking_df.sort_values(score_col, ascending=ascending).reset_index(
        drop=True
    )

    pool_df = pd.DataFrame(read_jsonl(pool_jsonl))
    merged = ranking_df.merge(
        pool_df,
        left_on=id_col,
        right_on="id",
        how="left",
        suffixes=("", "_pool"),
    )
    # If ranking CSV already had a 'text' column the merge renames pool's copy
    # to 'text_pool'.  Always prefer the pool's text column.
    if text_col not in merged.columns and f"{text_col}_pool" in merged.columns:
        merged = merged.rename(columns={f"{text_col}_pool": text_col})

    topk_df = merged.head(top_k).copy()
    topk_ids = topk_df[id_col].astype(str).tolist()

    with open(f"{out_dir}/top_{top_k}_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(topk_ids) + "\n")
    with open(f"{out_dir}/top_{top_k}_ids.json", "w", encoding="utf-8") as f:
        json.dump(topk_ids, f, ensure_ascii=False, indent=2)

    keep_cols = [
        c
        for c in [
            id_col,
            source_col,
            score_col,
            "subtype",
            "task_type",
            text_col,
            "url",
        ]
        if c in topk_df.columns
    ]
    topk_df[keep_cols].to_csv(f"{out_dir}/top_{top_k}_full.csv", index=False)
    write_jsonl(
        f"{out_dir}/top_{top_k}_full.jsonl",
        topk_df[keep_cols].to_dict(orient="records"),
    )

    source_df = topk_df if restrict_source_topn_to_topk else merged
    source_df = source_df.sort_values(score_col, ascending=ascending)

    per_source_rows = []
    for src, g in source_df.groupby(source_col, dropna=False, sort=True):
        per_source_rows.append(g.head(top_n_per_source))
    if per_source_rows:
        per_source_df = pd.concat(per_source_rows, ignore_index=True)
    else:
        per_source_df = merged.head(top_n_per_source).copy()
    per_source_df["text_preview"] = per_source_df[text_col].apply(
        lambda x: _fmt_text(x)
    )

    keep2 = [
        c
        for c in [
            id_col,
            source_col,
            score_col,
            "subtype",
            "task_type",
            "url",
            text_col,
            "text_preview",
        ]
        if c in per_source_df.columns
    ]
    per_source_df[keep2].to_csv(f"{out_dir}/top_samples_per_source.csv", index=False)
    per_source_df[keep2].to_json(
        f"{out_dir}/top_samples_per_source.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )

    md_lines = [
        "# Top samples per source",
        "",
        f"- Score column: `{score_col}`",
        f"- Top-K: `{top_k}`",
        f"- Top-N per source: `{top_n_per_source}`",
        "",
    ]
    for src, g in per_source_df.groupby(source_col, dropna=False, sort=True):
        md_lines.append(f"## Source: {src}")
        md_lines.append("")
        for _, row in g.sort_values(score_col, ascending=ascending).iterrows():
            md_lines.append(f"- ID: `{row[id_col]}`")
            md_lines.append(f"  - Score: `{row[score_col]:.8f}`")
            if "url" in row and pd.notna(row.get("url")):
                md_lines.append(f"  - URL: `{row['url']}`")
            md_lines.append(f"  - Preview: {_fmt_text(row[text_col])}")
        md_lines.append("")

    with open(f"{out_dir}/top_samples_per_source.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ── SwanLab: source distribution of top-K, score stats ──────────────
    source_counts = (
        topk_df[source_col].fillna("unknown").value_counts()
        if source_col in topk_df.columns
        else pd.Series(dtype=int)
    )
    # ── SwanLab: source distribution — native echarts Pie + Bar ─────────
    if not source_counts.empty:
        log_bar(
            "4_3_extraction/source_distribution",
            xaxis=source_counts.index.astype(str).tolist(),
            series={"count": source_counts.values.tolist()},
        )

    # ── SwanLab: score histogram — native echarts Bar ────────────────────
    import numpy as _np

    scores_arr = topk_df[score_col].to_numpy()
    counts, edges = _np.histogram(scores_arr, bins=40)
    hist_labels = [f"{edges[i]:.4f}" for i in range(len(edges) - 1)]
    log_bar(
        "4_3_extraction/score_histogram",
        xaxis=hist_labels,
        series={"count": counts.tolist()},
    )

    # ── SwanLab: Top-K sample table with text previews ────────────────────
    n_table = min(30, len(topk_df))
    headers = ["rank", "source", "score", "subtype", "task_type", "text"]
    rows = []
    for rank_i, (_, row) in enumerate(topk_df.head(n_table).iterrows(), 1):
        rows.append(
            [
                rank_i,
                str(row.get(source_col, "")),
                f"{row.get(score_col, 0):.6f}",
                str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
                str(row.get("task_type", "")) if pd.notna(row.get("task_type")) else "",
                _fmt_text(str(row.get(text_col, ""))),            ]
        )
    log_table("4_3_extraction/samples_top_selected", headers=headers, rows=rows)

    # ── Text statistics: top-K vs bottom-K vs full pool ────────────────────
    # All features are language/domain-agnostic — no hardcoded patterns.
    bottom_k = min(top_k, len(merged) - top_k)
    bottom_df = merged.tail(bottom_k).copy() if bottom_k > 0 else merged.head(1).copy()

    def _char_entropy(t: str) -> float:
        from collections import Counter

        c = Counter(str(t))
        n = max(1, sum(c.values()))
        return float(-sum((v / n) * np.log2(v / n + 1e-12) for v in c.values()))

    def _type_token_ratio(t: str) -> float:
        words = str(t).split()
        return len(set(words)) / max(1, len(words))

    def _max_word_repeat(t: str) -> int:
        from collections import Counter

        wc = Counter(str(t).lower().split())
        return int(max(wc.values())) if wc else 0

    text_features = {
        "length_chars": lambda t: float(len(str(t))),
        "word_count": lambda t: float(len(str(t).split())),
        "type_token_ratio": _type_token_ratio,
        "char_entropy": _char_entropy,
        "max_word_repeat": lambda t: float(_max_word_repeat(t)),
        "avg_word_len": lambda t: float(
            np.mean([len(w) for w in str(t).split()]) if str(t).split() else 0
        ),
        "digit_frac": lambda t: sum(c.isdigit() for c in str(t)) / max(1, len(str(t))),
        "punct_frac": lambda t: (
            sum(not c.isalnum() and not c.isspace() for c in str(t))
            / max(1, len(str(t)))
        ),
        "non_ascii_frac": lambda t: (
            sum(ord(c) > 127 for c in str(t)) / max(1, len(str(t)))
        ),
        "whitespace_frac": lambda t: (
            sum(c.isspace() for c in str(t)) / max(1, len(str(t)))
        ),
        "line_count": lambda t: float(str(t).count("\n") + 1),
        "blank_line_frac": lambda t: (
            sum(1 for l in str(t).split("\n") if not l.strip())
            / max(1, str(t).count("\n") + 1)
        ),
    }

    groups = {"top": topk_df, "bottom": bottom_df, "full": merged}
    feature_names = list(text_features.keys())
    mat = np.zeros((len(feature_names), len(groups)))
    for g_idx, (_, g_df) in enumerate(groups.items()):
        for f_idx, feat_name in enumerate(feature_names):
            fn = text_features[feat_name]
            vals = g_df[text_col].apply(fn)
            mat[f_idx, g_idx] = float(vals.mean())

    log_heatmap(
        "4_3_extraction/text_stats_feature_comparison",
        xaxis=list(groups.keys()),
        yaxis=feature_names,
        matrix=mat,
        value_label="mean",
    )

    # ── Source × text-feature heatmap ─────────────────────────────────────
    if source_col in topk_df.columns and not source_counts.empty:
        sources = source_counts.index.tolist()
        src_feat_mat = np.zeros((len(feature_names), len(sources)))
        for src_idx, src in enumerate(sources):
            src_df = topk_df[topk_df[source_col] == src]
            for f_idx, feat_name in enumerate(feature_names):
                fn = text_features[feat_name]
                vals = src_df[text_col].apply(fn)
                src_feat_mat[f_idx, src_idx] = float(vals.mean())
        log_heatmap(
            "4_3_extraction/text_stats_source_features_topk",
            xaxis=[s[:15] for s in sources],
            yaxis=feature_names,
            matrix=src_feat_mat,
            value_label="mean",
        )

    # ── Word frequency shift: top-K vs full pool ──────────────────────────
    # Dynamically discovers what words are unusually frequent in top-K,
    # without any hardcoded patterns — works for any language / domain.
    from collections import Counter as _Counter

    def _word_freq(df: pd.DataFrame, col: str) -> dict[str, float]:
        wc: dict[str, int] = _Counter()
        for t in df[col].dropna():
            for w in str(t).lower().split():
                if len(w) >= 3:
                    wc[w] += 1
        total = max(1, sum(wc.values()))
        return {w: c / total for w, c in wc.items()}

    top_freq = _word_freq(topk_df, text_col)
    pool_freq = _word_freq(merged, text_col)
    all_words = set(top_freq.keys()) | set(pool_freq.keys())

    if all_words:
        shift = {}
        for w in all_words:
            shift[w] = top_freq.get(w, 0) - pool_freq.get(w, 0)
        top_shifted = sorted(shift.items(), key=lambda x: abs(x[1]), reverse=True)[:30]
        shift_headers = ["word", "top_freq", "pool_freq", "shift"]
        shift_rows = []
        for w, s in top_shifted:
            shift_rows.append(
                [
                    w,
                    f"{top_freq.get(w, 0):.6f}",
                    f"{pool_freq.get(w, 0):.6f}",
                    f"{s:+.6f}",
                ]
            )
        log_table(
            "4_3_extraction/text_stats_word_shift_topk_vs_pool",
            headers=shift_headers,
            rows=shift_rows,
        )

    swan_finish()

    return {"score_col": score_col, "top_k": top_k}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top-influence samples.")
    parser.add_argument("--pool_jsonl", required=True)
    parser.add_argument("--ranking_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--id_col", default="sample_id")
    parser.add_argument("--source_col", default="source")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--score_col", default=None)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--top_n_per_source", type=int, default=3)
    parser.add_argument("--preview_chars", type=int, default=2000)
    parser.add_argument("--restrict_source_topn_to_topk", action="store_true")
    parser.add_argument(
        "--experiment_name",
        default=None,
        help="SwanLab experiment name. Auto-generated from params if not set.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="SwanLab run display name within the experiment (e.g. 'extraction' in pipeline mode).",
    )
    args = parser.parse_args()

    result = extract_top_samples(
        pool_jsonl=args.pool_jsonl,
        ranking_csv=args.ranking_csv,
        out_dir=args.out_dir,
        id_col=args.id_col,
        source_col=args.source_col,
        text_col=args.text_col,
        score_col=args.score_col,
        top_k=args.top_k,
        top_n_per_source=args.top_n_per_source,
        preview_chars=args.preview_chars,
        restrict_source_topn_to_topk=args.restrict_source_topn_to_topk,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
