"""Shared IO utilities for BIF."""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, payload: dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


_READ_EXTS = {".parquet", ".json", ".jsonl"}


def list_input_files(path_str: str) -> list[str]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path_str}")
    if path.is_file():
        if path.suffix.lower() not in _READ_EXTS:
            raise ValueError(f"Unsupported file type: {path}")
        return [str(path)]
    files: list[str] = []
    for ext in _READ_EXTS:
        files.extend(glob.glob(str(path / f"**/*{ext}"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No supported files found under: {path_str}")
    return files


def iter_json_records(file_path: str) -> Iterator[dict[str, Any]]:
    try:
        with open(file_path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            yield from (r for r in obj if isinstance(r, dict))
        elif isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                yield from (r for r in obj["data"] if isinstance(r, dict))
            else:
                yield obj
        return
    except json.JSONDecodeError:
        with open(file_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON/JSONL at {file_path}:{line_no}: {e}"
                    ) from e
                if isinstance(row, dict):
                    yield row


def iter_jsonl_records(file_path: str) -> Iterator[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def iter_parquet_records(file_path: str) -> Iterator[dict[str, Any]]:
    df = pd.read_parquet(file_path)
    yield from df.to_dict(orient="records")


def iter_records(path_str: str) -> Iterator[dict[str, Any]]:
    for fp in list_input_files(path_str):
        suffix = Path(fp).suffix.lower()
        if suffix == ".parquet":
            yield from iter_parquet_records(fp)
        elif suffix == ".json":
            yield from iter_json_records(fp)
        elif suffix == ".jsonl":
            yield from iter_jsonl_records(fp)


def extract_text(row: dict[str, Any], text_key: str | None = None) -> str | None:
    if text_key is not None:
        val = row.get(text_key)
        return normalize_text(val) if val is not None else None
    for k in ("text", "content", "body", "document", "article"):
        if k in row and row[k] is not None and str(row[k]).strip():
            return normalize_text(row[k])
    return None
