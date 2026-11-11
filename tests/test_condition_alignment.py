import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "reca_audio"
    / "analysis"
    / "condition_alignment.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("condition_alignment_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_last_subsequence_mask_uses_last_occurrence():
    mask = _module().find_last_subsequence_mask([1, 2, 3, 2, 3, 4], [2, 3])
    assert mask.tolist() == [False, False, False, True, True, False]


def test_audio_payload_inventory_excludes_words_and_boundaries():
    result = _module().audio_payload_token_inventory(
        {
            "audio": 1,
            "Audio": 2,
            "<|audio_bos|>": 3,
            "<|audio_eos|>": 4,
            "<|AUDIO|>": 5,
            "<|audio_pad|>": 6,
        }
    )
    assert result == {"<|AUDIO|>": 5, "<|audio_pad|>": 6}


def test_group_manifest_rows_uses_audio_content_not_distinct_paths(tmp_path):
    first = tmp_path / "caption_a.wav"
    second = tmp_path / "caption_b.wav"
    third = tmp_path / "other.wav"
    first.write_bytes(b"same audio bytes")
    second.write_bytes(b"same audio bytes")
    third.write_bytes(b"different audio bytes")
    rows = [
        {"audio": str(first), "caption": "caption one"},
        {"audio": str(second), "caption": "caption two"},
        {"audio": str(third), "caption": "caption three"},
    ]
    groups = _module().group_manifest_rows_by_audio_sha(rows)
    assert len(groups) == 2
    assert groups[0]["captions"] == ["caption one", "caption two"]
    assert groups[0]["row_index"] == 0


def test_masked_pool_rejects_empty_selection():
    with pytest.raises(ValueError, match="empty"):
        _module().masked_pool(torch.ones(3, 2), torch.zeros(3, dtype=torch.bool))


def test_alignment_metrics_separates_matches_and_supports_multiple_captions():
    module = _module()
    audio = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    text = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [-0.9, 0.1]]
    )
    result = module.alignment_metrics(
        audio,
        text,
        ["a", "b", "c"],
        ["a", "a", "b", "c", "c"],
        recall_ks=(1, 2),
    )
    assert result["cosine_margin"] > 0.9
    assert result["audio_to_text"]["r@1"] == 1.0
    assert result["text_to_audio"]["r@1"] == 1.0
    assert result["linear_cka"] > 0.99


def test_alignment_metrics_rejects_duplicate_audio_groups():
    with pytest.raises(ValueError, match="unique"):
        _module().alignment_metrics(
            torch.eye(2), torch.eye(2), ["same", "same"], ["same", "same"]
        )

