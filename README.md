# BIF

`BIF` is a slimmed-down split of the BIF part of BIFrost.

It keeps only the BIF workflow:
- `run-bif`
- `analyze-bif`
- `extract-top`

It removes the fine-tuning and replay pipeline:
- `prepare-finetune`
- `train`
- `schedule-compare`
- `schedule-analyze`
- `pipeline`

## Scope

This repo is for:
- collecting BIF traces from a base model or checkpoints
- analyzing BIF structure and source-level rankings
- extracting top-ranked samples for downstream use

This repo is not for:
- stage-2 fine-tuning
- replay schedule experiments
- full end-to-end pipeline orchestration

## Data

Data is intentionally left unchanged.

You can:
- keep using the existing JSONL data files under your old BIFrost `data/` directory
- point commands to any external `pool_jsonl`, `query_jsonl`, and checkpoint paths

This split does not move, rewrite, or duplicate your data by default.

## Environment

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If you already have a working Torch / Transformers environment, that is enough.

## Commands

Show help:

```bash
python -m bif.cli run-bif --help
python -m bif.cli analyze-bif --help
python -m bif.cli extract-top --help
```

Run BIF:

```bash
python -m bif.cli run-bif \
  --config configs/example_run_bif.yaml
```

Analyze traces:

```bash
python -m bif.cli analyze-bif \
  --config configs/example_analyze_bif.yaml
```

Extract top samples:

```bash
python -m bif.cli extract-top \
  --pool_jsonl /path/to/pool.jsonl \
  --ranking_csv /path/to/bif_analysis/final_model/pool_scores.csv \
  --out_dir ./runs/top_samples
```

## Layout

```text
.
├── configs/
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/bif/
└── tests/
```

## Notes

- The CLI flags for the kept BIF commands stay close to the original BIFrost behavior.
- Tracking still uses SwanLab if it is installed and configured.
- This repo is intentionally conservative: it preserves the BIF path first, then removes finetune and schedule concerns around it.
