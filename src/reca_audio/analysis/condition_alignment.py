"""Statistics for paired audio/text condition-alignment diagnostics.

The functions in this module are deliberately model-agnostic.  The heavy Qwen
and DiT extraction lives in ``runs/scripts/diagnose_condition_alignment.py``;
keeping the statistics separate makes the positive/negative bookkeeping easy
to unit test without GPUs or model checkpoints.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch


def audio_payload_token_inventory(vocab: dict[str, int]) -> dict[str, int]:
    """Select Qwen audio payload placeholders, excluding lexical audio tokens.

    Qwen2.5-Omni in this repository expands the input waveform into repeated
    ``<|AUDIO|>`` positions and does not expose ``<|audio_pad|>``.  Matching
    every vocabulary token containing the substring ``audio`` would silently
    include ordinary prompt words such as ``audio`` and ``Audio``.
    """

    inventory = {
        token: int(token_id)
        for token, token_id in vocab.items()
        if token.lower() in {"<|audio|>", "<|audio_pad|>"}
    }
    return inventory


def find_last_subsequence_mask(
    input_ids: torch.Tensor | Sequence[int],
    needle: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Return a boolean mask for the last exact occurrence of ``needle``.

    Selecting the last occurrence is intentional: prompt templates can repeat
    words in the system message, while the user caption/instruction is the last
    matching span in the formatted chat prompt.
    """

    ids = torch.as_tensor(input_ids, dtype=torch.long).flatten()
    target = torch.as_tensor(needle, dtype=torch.long).flatten()
    mask = torch.zeros(ids.numel(), dtype=torch.bool)
    if target.numel() == 0 or target.numel() > ids.numel():
        return mask
    start = None
    for index in range(ids.numel() - target.numel() + 1):
        if torch.equal(ids[index : index + target.numel()], target):
            start = index
    if start is not None:
        mask[start : start + target.numel()] = True
    return mask


def mask_from_token_ids(
    input_ids: torch.Tensor | Sequence[int], token_ids: Iterable[int]
) -> torch.Tensor:
    """Return positions whose token id belongs to ``token_ids``."""

    ids = torch.as_tensor(input_ids, dtype=torch.long).flatten()
    candidates = {int(item) for item in token_ids}
    if not candidates:
        return torch.zeros_like(ids, dtype=torch.bool)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for token_id in candidates:
        mask |= ids == token_id
    return mask


def masked_pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Pool ``[sequence, feature]`` hidden states over an explicit mask."""

    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [sequence, feature], got {tuple(hidden.shape)}")
    mask = torch.as_tensor(mask, dtype=torch.bool, device=hidden.device).flatten()
    if mask.numel() != hidden.shape[0]:
        raise ValueError("mask length does not match hidden sequence length")
    selected = hidden[mask]
    if selected.shape[0] == 0:
        raise ValueError("cannot pool an empty token selection")
    if mode == "mean":
        return selected.float().mean(dim=0)
    if mode == "last":
        return selected[-1].float()
    if mode == "mean_std":
        return torch.cat(
            [selected.float().mean(dim=0), selected.float().std(dim=0, unbiased=False)],
            dim=0,
        )
    raise ValueError(f"unknown pooling mode: {mode!r}")


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """Linear centered-kernel alignment for two sample-by-feature matrices."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA expects two 2D matrices with equal sample count")
    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x.T @ y).square()
    x_norm = torch.linalg.matrix_norm(x.T @ x)
    y_norm = torch.linalg.matrix_norm(y.T @ y)
    denominator = x_norm * y_norm
    if float(denominator) <= eps:
        return float("nan")
    return float(cross / denominator)


def _rbf_mmd(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Biased RBF MMD and its data-dependent bandwidth."""

    joined = torch.cat([x.double(), y.double()], dim=0)
    squared = torch.cdist(joined, joined).square()
    positive = squared[squared > 0]
    bandwidth = float(positive.median()) if positive.numel() else 1.0
    bandwidth = max(bandwidth, 1e-12)
    kernel = torch.exp(-squared / (2.0 * bandwidth))
    n = x.shape[0]
    k_xx = kernel[:n, :n]
    k_yy = kernel[n:, n:]
    k_xy = kernel[:n, n:]
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(mmd), bandwidth


def _mean_off_diagonal(matrix: torch.Tensor) -> float | None:
    if matrix.shape[0] < 2:
        return None
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].mean())


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.double().flatten()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def alignment_metrics(
    audio_vectors: torch.Tensor,
    text_vectors: torch.Tensor,
    audio_group_ids: Sequence[str],
    text_group_ids: Sequence[str],
    recall_ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Compare paired audio/text representations without positional alignment.

    Each audio row must identify one semantic group.  Text can contain multiple
    captions for the same group.  Retrieval treats every caption belonging to
    an audio group as a positive and all other groups as negatives.
    """

    audio = torch.as_tensor(audio_vectors, dtype=torch.float64)
    text = torch.as_tensor(text_vectors, dtype=torch.float64)
    if audio.ndim != 2 or text.ndim != 2 or audio.shape[1] != text.shape[1]:
        raise ValueError("audio/text vectors must be 2D with the same feature dimension")
    if audio.shape[0] != len(audio_group_ids) or text.shape[0] != len(text_group_ids):
        raise ValueError("group id count does not match vector count")
    if len(set(audio_group_ids)) != len(audio_group_ids):
        raise ValueError("audio_group_ids must be unique")
    audio_index = {group: index for index, group in enumerate(audio_group_ids)}
    missing = sorted(set(text_group_ids) - set(audio_index))
    if missing:
        raise ValueError(f"text groups have no audio vector: {missing[:3]}")

    audio_norms = torch.linalg.vector_norm(audio, dim=1)
    text_norms = torch.linalg.vector_norm(text, dim=1)
    audio_unit = audio / audio_norms.clamp_min(1e-12).unsqueeze(1)
    text_unit = text / text_norms.clamp_min(1e-12).unsqueeze(1)
    cosine = audio_unit @ text_unit.T
    positive = torch.tensor(
        [[a == t for t in text_group_ids] for a in audio_group_ids], dtype=torch.bool
    )
    if not positive.any() or bool(positive.all()):
        raise ValueError("alignment metrics require both positive and negative pairs")

    audio_ranks = []
    a2t_recall = {}
    for row in range(cosine.shape[0]):
        order = torch.argsort(cosine[row], descending=True)
        hits = positive[row, order]
        audio_ranks.append(int(torch.nonzero(hits, as_tuple=False)[0, 0]) + 1)
    for k in recall_ks:
        a2t_recall[f"r@{k}"] = sum(rank <= k for rank in audio_ranks) / len(audio_ranks)

    text_ranks = []
    t2a_recall = {}
    for column, group in enumerate(text_group_ids):
        order = torch.argsort(cosine[:, column], descending=True)
        target = audio_index[group]
        text_ranks.append(int(torch.nonzero(order == target, as_tuple=False)[0, 0]) + 1)
    for k in recall_ks:
        t2a_recall[f"r@{k}"] = sum(rank <= k for rank in text_ranks) / len(text_ranks)

    grouped_text = torch.stack(
        [text_unit[[group == audio_group for group in text_group_ids]].mean(dim=0) for audio_group in audio_group_ids]
    )
    grouped_text = grouped_text / torch.linalg.vector_norm(grouped_text, dim=1).clamp_min(1e-12).unsqueeze(1)
    paired_group_cosine = (audio_unit * grouped_text).sum(dim=1)
    mmd, bandwidth = _rbf_mmd(audio_unit, grouped_text)

    matched = cosine[positive]
    unmatched = cosine[~positive]
    return {
        "n_audio": audio.shape[0],
        "n_text": text.shape[0],
        "feature_dim": audio.shape[1],
        "matched_cosine": _summary(matched),
        "unmatched_cosine": _summary(unmatched),
        "cosine_margin": float(matched.mean() - unmatched.mean()),
        "paired_group_cosine": _summary(paired_group_cosine),
        "audio_to_text": {
            **a2t_recall,
            "median_rank": float(torch.tensor(audio_ranks, dtype=torch.float64).median()),
        },
        "text_to_audio": {
            **t2a_recall,
            "median_rank": float(torch.tensor(text_ranks, dtype=torch.float64).median()),
        },
        "linear_cka": linear_cka(audio_unit, grouped_text),
        "rbf_mmd": mmd,
        "rbf_bandwidth": bandwidth,
        "audio_norm": _summary(audio_norms),
        "text_norm": _summary(text_norms),
        "audio_internal_cosine": _mean_off_diagonal(audio_unit @ audio_unit.T),
        "text_internal_cosine": _mean_off_diagonal(grouped_text @ grouped_text.T),
    }

