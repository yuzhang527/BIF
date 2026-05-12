# BIF sweep implementation patch

This bundle adds a `sweep-bif` workflow that runs a grid over `lr`, `gamma`, and `nbeta`, adds matched `nbeta=0` baselines, and computes ranking-stability diagnostics.

## Files

- `src/bif/analysis/sweep_runner.py`
  - Builds the grid.
  - Writes per-run configs.
  - Calls existing `python -m bif.cli run-bif` and `python -m bif.cli analyze-bif`.
  - Runs diagnostics.
  - Aggregates `sweep_plan.csv`, `sweep_summary.csv`, and `baseline_comparisons.csv`.

- `src/bif/analysis/sweep_diagnostics.py`
  - Loads traces with existing `load_checkpoint_traces`.
  - Computes full-draw diagnostic scores.
  - Runs repeated random half-split draw stability.
  - Runs chain-to-chain stability.
  - Compares nonzero-nbeta runs against matched `nbeta=0` baselines.

- `configs/example_sweep_bif.yaml`
  - Example sweep config.

- `src/bif/cli.py`
  - Full replacement of the current CLI file, preserving existing commands and adding `sweep-bif`.

- `tests/test_sweep_diagnostics.py`
  - Lightweight unit tests for overlap and sweep-axis parsing.

## Apply

From the repository root, copy this bundle over the repo, replacing `src/bif/cli.py`:

```bash
cp -r /path/to/bif_sweep_patch/src ./
cp -r /path/to/bif_sweep_patch/configs ./
cp -r /path/to/bif_sweep_patch/tests ./
```

Then run:

```bash
python -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml --dry_run
```

A real run uses:

```bash
python -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml
```

## Output layout

```text
runs/bif_sweeps/example_sweep/
  sweep_config_resolved.yaml
  sweep_plan.csv
  sweep_summary.partial.csv
  sweep_summary.csv
  baseline_comparisons.csv
  runs/
    grid_000_lr.../
      run_config.yaml
      analysis_config.yaml
      commands.log
      sweep_run_manifest.json
      traces/
      analysis/
      diagnostics/
        diagnostics_summary.csv
        final_model/
          pool_scores_diagnostic.csv
          split_stability.csv
          chain_stability.csv
          diagnostics_summary.json
```
