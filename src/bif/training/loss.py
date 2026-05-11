"""Loss functions for language model training and evaluation.

Aligned with devinterp's lm_loss.py:
- per_token_causal_lm_loss: shape (batch, seq-1), matches compute_per_token_loss
- per_example_causal_lm_loss: shape (batch,), sequence-level mean over tokens
"""

from __future__ import annotations

import torch
from torch import nn

from bif.constants import IGNORE_INDEX


def per_token_causal_lm_loss(
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute per-token causal LM loss (aligned with devinterp compute_per_token_loss).

    Args:
        input_ids: Token IDs of shape (batch, seq_len).
        logits: Model logits of shape (batch, seq_len, vocab_size).
        ignore_index: Label value to ignore in loss computation.

    Returns:
        Per-token loss tensor of shape (batch, seq_len - 1).
    """
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    shift_log_probs = log_probs[:, :-1, :]
    shift_input_ids = input_ids[:, 1:, None]
    return -shift_log_probs.gather(dim=-1, index=shift_input_ids)[:, :, 0]


def per_example_causal_lm_loss(
    labels: torch.Tensor,
    logits: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Compute per-example causal LM loss (mean over valid tokens).

    Args:
        labels: Token labels of shape (batch, seq_len).
        logits: Model logits of shape (batch, seq_len, vocab_size).
        ignore_index: Label value to ignore in loss computation.

    Returns:
        Per-example loss tensor of shape (batch,).
    """
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(ignore_index)
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    vocab_size = shift_logits.size(-1)
    flat_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    per_token_loss = flat_loss.view_as(shift_labels)
    valid_counts = valid_mask.sum(dim=1).clamp(min=1)
    return per_token_loss.sum(dim=1) / valid_counts
