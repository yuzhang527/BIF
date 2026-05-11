"""CLI entry point for the BIF-only split."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bif",
        description="BIF-only toolkit: run-bif, analyze-bif, extract-top",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    p_run = sub.add_parser("run-bif", help="Run BIF trace collection")
    p_run.add_argument("--config", default=None, help="YAML config file")
    p_run.add_argument("--model_name_or_path", default=None)
    p_run.add_argument("--model_root", default=None)
    p_run.add_argument("--base_model_path", default=None)
    p_run.add_argument("--tokenizer_path", default=None)
    p_run.add_argument("--run_all_checkpoints", action="store_true")
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--pool_jsonl", default=None)
    p_run.add_argument("--query_jsonl", default=None)
    p_run.add_argument("--out_dir", default=None)
    p_run.add_argument("--num_chains", type=int, default=4)
    p_run.add_argument("--draws_per_chain", type=int, default=60)
    p_run.add_argument("--max_length", type=int, default=256)
    p_run.add_argument("--train_batch_size", type=int, default=16)
    p_run.add_argument("--eval_batch_size", type=int, default=32)
    p_run.add_argument("--pool_eval_subset", type=int, default=0)
    p_run.add_argument("--lr", type=float, default=5e-6)
    p_run.add_argument("--gamma", type=float, default=1e-3)
    p_run.add_argument("--beta", type=float, default=1.0)
    p_run.add_argument(
        "--nbeta_mode",
        type=str,
        default="devinterp",
        choices=["devinterp", "dataset"],
    )
    p_run.add_argument("--nbeta", type=float, default=-1.0)
    p_run.add_argument("--noise_level", type=float, default=1.0)
    p_run.add_argument("--num_burnin_steps", type=int, default=0)
    p_run.add_argument("--num_steps_bw_draws", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--grad_clip", type=float, default=None)
    p_run.add_argument("--weight_decay", type=float, default=0.0)
    p_run.add_argument(
        "--sampler_type",
        default="sgld",
        choices=["sgld", "rmsprop_sgld"],
    )
    p_run.add_argument("--rmsprop_alpha", type=float, default=0.99)
    p_run.add_argument("--rmsprop_eps", type=float, default=1e-1)
    p_run.add_argument("--batches_per_draw", type=int, default=0)
    p_run.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p_run.add_argument("--chain_id", type=int, default=None)
    p_run.add_argument("--checkpoints", default=None)
    p_run.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--experiment_name", default=None)
    p_run.add_argument("--model_tag", default=None)
    p_run.add_argument("--run_name", default=None)

    p_analyze = sub.add_parser("analyze-bif", help="Analyze BIF results")
    p_analyze.add_argument("--config", default=None, help="YAML config file")
    p_analyze.add_argument("--bif_root", default=None)
    p_analyze.add_argument("--out_dir", default=None)
    p_analyze.add_argument("--score_col", default=None)
    p_analyze.add_argument("--top_k", type=int, default=None)
    p_analyze.add_argument("--enable_aux_query_plots", action="store_true")
    p_analyze.add_argument(
        "--negate_scores",
        action="store_true",
        default=False,
    )
    p_analyze.add_argument("--experiment_name", default=None)
    p_analyze.add_argument("--run_name", default=None)
    for flag, typ in [
        ("hist_bins", int),
        ("scatter_max_points", int),
        ("heatmap_max_pool", int),
        ("heatmap_max_query", int),
        ("rhat_max_samples", int),
        ("eigenvalue_max_pool", int),
        ("eigenvalue_max_ev", int),
        ("boxplot_max_sources", int),
        ("boxplot_min_per_source", int),
        ("rhat_min_draws", int),
        ("chain_scatter_min_draws", int),
        ("trajectory_top_n", int),
        ("source_label_max_len", int),
        ("heatmap_topk_max", int),
        ("convergence_min_draws", int),
    ]:
        p_analyze.add_argument(f"--{flag}", type=typ, default=None)
    p_analyze.add_argument(
        "--convergence_checkpoints",
        type=int,
        nargs="+",
        default=None,
    )

    p_extract = sub.add_parser("extract-top", help="Extract top-influence samples")
    p_extract.add_argument("--pool_jsonl", required=True)
    p_extract.add_argument("--ranking_csv", required=True)
    p_extract.add_argument("--out_dir", required=True)
    p_extract.add_argument("--id_col", default="sample_id")
    p_extract.add_argument("--source_col", default="source")
    p_extract.add_argument("--text_col", default="text")
    p_extract.add_argument("--score_col", default=None)
    p_extract.add_argument("--top_k", type=int, default=500)
    p_extract.add_argument("--top_n_per_source", type=int, default=3)
    p_extract.add_argument("--preview_chars", type=int, default=600)
    p_extract.add_argument("--restrict_source_topn_to_topk", action="store_true")
    p_extract.add_argument("--ascending", action="store_true")
    p_extract.add_argument("--experiment_name", default=None)
    p_extract.add_argument("--run_name", default=None)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run-bif":
        from bif.analysis import bif_runner as runner

        argv = sys.argv[2:]
        saved_argv = sys.argv
        sys.argv = ["bif run-bif"] + argv
        try:
            runner.main()
        finally:
            sys.argv = saved_argv
        return

    if args.command == "analyze-bif":
        from bif.analysis.bif_analyzer import AnalyzeConfig, analyze_bif_results

        bif_root = args.bif_root
        out_dir = args.out_dir
        if out_dir is None and bif_root is not None:
            out_dir = f"{bif_root}/analysis"

        acfg = AnalyzeConfig()
        for field_name in (
            "score_col",
            "top_k",
            "negate_scores",
            "enable_aux_query_plots",
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
        ):
            val = getattr(args, field_name, None)
            if val is not None and not (isinstance(val, bool) and not val):
                setattr(acfg, field_name, val)
        if args.convergence_checkpoints is not None:
            acfg.convergence_checkpoints = args.convergence_checkpoints

        analyze_bif_results(
            bif_root=bif_root,
            out_dir=out_dir,
            acfg=acfg,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
        return

    if args.command == "extract-top":
        from bif.analysis.extractor import extract_top_samples

        extract_top_samples(
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
            ascending=args.ascending,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
        )
        return


if __name__ == "__main__":
    main()
