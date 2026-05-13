# BIF

`BIF` is a slimmed-down split of the BIF part of BIFrost.

It keeps only the BIF workflow:

- `run-bif`
- `analyze-bif`
- `extract-top`
- `sweep-bif`

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
- sweeping BIF sampler parameters and running diagnostics

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
python -m bif.cli sweep-bif --help
```

### Sweep BIF, single process

```bash
python -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml
```

### Sweep BIF, multi-GPU with torchrun

After applying the torchrun-safe `sweep_runner.py` patch, this is the recommended multi-GPU sweep mode:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml
```

For an 8-GPU node:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml
```

If multiple torchrun sweep jobs run on the same machine at the same time, prefer an auto-selected rendezvous port:

```bash
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
  --nnodes=1 --nproc-per-node=4 \
  -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml
```

Torchrun sweep semantics:

- outer torchrun ranks split the sweep grid by `idx % WORLD_SIZE == RANK`
- child `run-bif` / `analyze-bif` processes do not inherit `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, or `MASTER_PORT`
- each child process is masked to one GPU through `CUDA_VISIBLE_DEVICES`
- each run directory gets a `.sweep_claim` file to prevent duplicate writers
- each rank writes `sweep_summary.rankXXX.partial.csv`
- only rank 0 writes the final `sweep_summary.csv` and `sweep_summary_meta.json`

If a job is killed hard, a stale `.sweep_claim` file may remain in a run directory. After confirming no process is still writing that run, delete the stale `.sweep_claim` file and rerun with `skip_existing: true`.

### Extract top samples

```bash
python -m bif.cli extract-top \
  --pool_jsonl /path/to/pool.jsonl \
  --ranking_csv /path/to/bif_analysis/final_model/pool_scores.csv \
  --out_dir ./runs/top_samples
```

## Important note about run-bif and analyze-bif config files

Do not document or rely on these forms until the CLI implements YAML loading for them:

```bash
python -m bif.cli run-bif --config configs/example_run_bif.yaml
python -m bif.cli analyze-bif --config configs/example_analyze_bif.yaml
```

The current safe documented config entrypoint is:

```bash
python -m bif.cli sweep-bif --config configs/example_sweep_bif.yaml
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


