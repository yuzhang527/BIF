"""Dataset classes for BIF (CPT mode — all tokens contribute to loss)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bif.io import read_jsonl

from bif.constants import IGNORE_INDEX


class JsonlSequenceDataset(Dataset):
    """Dataset for BIF trace collection with per-sample metadata."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        max_length: int = 256,
        text_key: str = "text",
        id_key: str = "id",
        source_type_key: str = "source",
        subtype_key: str = "subtype",
        task_type_key: str = "task_type",
        strict: bool = True,
    ):
        self.examples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        rows = read_jsonl(path)
        for idx, obj in enumerate(rows):
            if text_key not in obj:
                if strict:
                    raise ValueError(f"Missing '{text_key}' in {path} line {idx + 1}")
                continue
            ex_id = obj.get(id_key, idx)
            meta = {
                "id": ex_id,
                "source_type": obj.get(source_type_key),
                "subtype": obj.get(subtype_key),
                "task_type": obj.get(task_type_key),
            }
            if "answer_start_char" in obj:
                meta["answer_start_char"] = obj["answer_start_char"]
            self.examples.append({**meta, "text": str(obj[text_key])})

        if not self.examples:
            raise ValueError(f"No usable examples in {path}")

        self._cache: list[dict[str, torch.Tensor]] = [
            self._encode(ex)
            for ex in tqdm(self.examples, desc=f"Tokenizing {path}", leave=False)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, ex: dict[str, Any]) -> dict[str, torch.Tensor]:
        answer_start_char = ex.get("answer_start_char")

        if answer_start_char is not None:
            question_text = ex["text"][:answer_start_char]
            q_enc = self.tokenizer(
                question_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            )
            answer_token_count = q_enc["input_ids"].shape[1]

        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = item["input_ids"].clone()
        item["labels"][item["attention_mask"] == 0] = IGNORE_INDEX

        if answer_start_char is not None and answer_token_count > 0:
            item["labels"][:answer_token_count] = IGNORE_INDEX

        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        item = {k: v.clone() for k, v in self._cache[idx].items()}
        item["attention_mask"] = item["attention_mask"].long()
        item["sample_id"] = ex["id"]
        item["dataset_index"] = idx
        item["source_type"] = ex["source_type"]
        item["subtype"] = ex["subtype"]
        item["task_type"] = ex["task_type"]
        return item

    def subset(self, indices: list[int]) -> "JsonlSequenceDataset":
        """Return a new dataset containing only the given indices.

        Shares the tokenization cache with the parent — O(1) extra memory.
        """
        child = object.__new__(JsonlSequenceDataset)
        child.examples = [self.examples[i] for i in indices]
        child._cache = [self._cache[i] for i in indices]
        child.tokenizer = self.tokenizer
        child.max_length = self.max_length
        return child


def collate_bif_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate batch for BIF trace collection."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "sample_ids": [b["sample_id"] for b in batch],
        "dataset_indices": torch.tensor(
            [b["dataset_index"] for b in batch], dtype=torch.long
        ),
        "source_types": [b.get("source_type") for b in batch],
        "subtypes": [b.get("subtype") for b in batch],
        "task_types": [b.get("task_type") for b in batch],
    }


def get_batch_by_indices(dataset: Dataset, indices: list[int]) -> dict[str, Any]:
    return collate_bif_batch([dataset[int(i)] for i in indices])


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class LMTextDataset(Dataset):
    """LM dataset for training.

    Supports two modes:
      - 'full': all tokens contribute to loss (CPT mode)
      - 'response_only': only tokens after the prompt contribute to loss (SFT mode)
        Requires 'prompt' and 'response' keys in each row.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
        text_key: str = "text",
        loss_mode: str = "full",
        prompt_key: str = "prompt",
        response_key: str = "response",
        group_labels: list[str] | None = None,
    ):
        if loss_mode not in ("full", "response_only"):
            raise ValueError(f"loss_mode must be 'full' or 'response_only', got '{loss_mode}'")

        self.examples: list[dict[str, torch.Tensor]] = []
        self.group_labels: list[str] = []

        for i, row in enumerate(rows):
            if loss_mode == "full":
                if text_key not in row:
                    raise ValueError(f"Missing '{text_key}' in row {i}")
                text = str(row[text_key]).strip()
                if not text:
                    continue
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = IGNORE_INDEX
            else:
                if prompt_key not in row or response_key not in row:
                    raise ValueError(f"Missing '{prompt_key}' or '{response_key}' in row {i}")
                prompt = str(row[prompt_key])
                response = str(row[response_key])

                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                if tokenizer.eos_token_id is not None:
                    response_ids = response_ids + [tokenizer.eos_token_id]

                all_ids = prompt_ids + response_ids
                all_labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids[:]

                if len(all_ids) > max_length:
                    all_ids = all_ids[:max_length]
                    all_labels = all_labels[:max_length]

                attention_mask = [1] * len(all_ids)
                pad_len = max_length - len(all_ids)
                if pad_len > 0:
                    pad_id = tokenizer.pad_token_id or 0
                    all_ids = all_ids + [pad_id] * pad_len
                    all_labels = all_labels + [IGNORE_INDEX] * pad_len
                    attention_mask = attention_mask + [0] * pad_len

                input_ids = torch.tensor(all_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                labels = torch.tensor(all_labels, dtype=torch.long)

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
            if group_labels is not None:
                self.group_labels.append(group_labels[i])

        if not self.examples:
            raise ValueError("No usable training examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v.clone() for k, v in self.examples[idx].items()}
        if self.group_labels:
            item["group"] = self.group_labels[idx]
        return item


class DataCollatorForLM:
    """Dynamic padding collator for LM training."""

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        groups = [f.get("group", "") for f in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        if self.pad_to_multiple_of:
            seq_len = batch_input_ids.size(1)
            target_len = int(
                math.ceil(seq_len / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )
            pad_len = target_len - seq_len
            if pad_len > 0:
                batch_input_ids = torch.nn.functional.pad(
                    batch_input_ids,
                    (0, pad_len),
                    value=self.tokenizer.pad_token_id,
                )
                batch_attention_mask = torch.nn.functional.pad(
                    batch_attention_mask,
                    (0, pad_len),
                    value=0,
                )
                batch_labels = torch.nn.functional.pad(
                    batch_labels,
                    (0, pad_len),
                    value=IGNORE_INDEX,
                )

        result = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
        if any(g for g in groups):
            result["groups"] = groups
        return result
