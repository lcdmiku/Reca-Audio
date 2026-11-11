#!/usr/bin/env python3
"""Measure audio/text condition alignment at Qwen, projection and DiT stages.

This is a read-only diagnostic.  It never calls the VAE, never computes a flow
loss and never performs an optimizer step.  Different-length token sequences
are reduced with explicit masks before cosine/retrieval/CKA statistics are
computed; no positional audio/text token alignment is attempted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from reca_audio.analysis.condition_alignment import (
    alignment_metrics,
    audio_payload_token_inventory,
    find_last_subsequence_mask,
    group_manifest_rows_by_audio_sha,
    mask_from_token_ids,
    masked_pool,
)
from reca_audio.train.conditioning import RECA_INSTRUCTION, _apply_bottleneck
from reca_audio.train.data import AudioManifestDataset
from audio_omni.loader import load_model_bundle
from audio_omni.prompts import build_prompt, build_reconstruct_prompt


VIEWS = ("content", "semantic", "full", "last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    parser.add_argument("--base-ckpt", default="Audio-Omni/model/model.ckpt")
    parser.add_argument(
        "--dit",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="optional trained DiT checkpoint; repeat for Pure/T2A/Mix",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditioner-device", default="cuda:1")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--omni-downsample", type=int, default=4)
    parser.add_argument("--omni-seconds", type=float, default=0.0)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-captions-per-group", type=int, default=5)
    parser.add_argument("--task", default="T2A")
    parser.add_argument("--reca-instruction", default=RECA_INSTRUCTION)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_dit_specs(items: list[str]) -> OrderedDict[str, Path]:
    result: OrderedDict[str, Path] = OrderedDict()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dit must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name or name == "base":
            raise ValueError("DiT name must be non-empty and cannot be 'base'")
        if name in result:
            raise ValueError(f"duplicate DiT name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def select_groups(
    dataset: AudioManifestDataset,
    max_groups: int,
    max_captions_per_group: int,
) -> list[dict[str, Any]]:
    """Preserve manifest order and retain all positive captions per audio SHA."""

    return group_manifest_rows_by_audio_sha(
        dataset.rows,
        max_groups=max_groups,
        max_captions_per_group=max_captions_per_group,
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def _audio_token_inventory(tokenizer: Any) -> dict[str, int]:
    return audio_payload_token_inventory(tokenizer.get_vocab())


def _align_1d(tensor: torch.Tensor, length: int, pad_value: int | bool) -> torch.Tensor:
    tensor = tensor.flatten().cpu()
    if tensor.numel() >= length:
        return tensor[:length]
    return F.pad(tensor, (0, length - tensor.numel()), value=pad_value)


def _semantic_mask(tokenizer: Any, ids: torch.Tensor, text: str) -> tuple[torch.Tensor, bool]:
    candidates = [text, " " + text, "\n" + text]
    for candidate in candidates:
        needle = _token_ids(tokenizer, candidate)
        mask = find_last_subsequence_mask(ids, needle)
        if mask.any():
            return mask, False
    return torch.zeros(ids.numel(), dtype=torch.bool), True


def _record_masks(
    *,
    kind: str,
    ids: torch.Tensor,
    valid: torch.Tensor,
    tokenizer: Any,
    audio_token_ids: set[int],
    semantic_text: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    semantic, fallback = _semantic_mask(tokenizer, ids, semantic_text)
    semantic &= valid
    audio = mask_from_token_ids(ids, audio_token_ids) & valid
    if kind == "audio":
        content = audio
        if not content.any():
            raise RuntimeError("audio prompt contains no recognized audio token positions")
    else:
        content = semantic
    if not semantic.any():
        # Context-sensitive tokenization can prevent exact subsequence matching.
        # A deterministic tail span keeps the run alive and is explicitly counted.
        length = max(1, len(_token_ids(tokenizer, semantic_text)))
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        semantic = torch.zeros_like(valid)
        semantic[indices[-min(length, indices.numel()) :]] = True
        if kind == "text":
            content = semantic.clone()
    last = torch.zeros_like(valid)
    last[torch.nonzero(valid, as_tuple=False).flatten()[-1]] = True
    return {
        "content": content,
        "semantic": semantic,
        "full": valid,
        "last": last,
    }, {
        "semantic_fallback": fallback,
        "valid_tokens": int(valid.sum()),
        "content_tokens": int(content.sum()),
        "semantic_tokens": int(semantic.sum()),
        "audio_tokens": int(audio.sum()),
    }


def _prepare_qwen_inputs(
    conditioner: Any,
    prompt: str,
    audio: Any,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    inputs = conditioner.processor(
        text=[prompt],
        audio=None if audio is None else [audio],
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    ids = inputs["input_ids"][0].detach().cpu()
    attention = inputs.get("attention_mask")
    valid = (
        torch.ones(ids.numel(), dtype=torch.bool)
        if attention is None
        else attention[0].detach().cpu().bool()
    )
    moved = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    return moved, ids, valid


def _project_tokens(conditioner: Any, hidden: torch.Tensor) -> torch.Tensor:
    hidden = hidden.to(conditioner.proj_features.weight.device)
    normalized = conditioner.norm(hidden)
    return conditioner.proj_features(normalized.to(conditioner.proj_features.weight.dtype))


def _exact_projected_sequence(conditioner: Any, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    tokens = hidden[valid.to(hidden.device)]
    if tokens.shape[0] > conditioner.pad_length:
        tokens = F.interpolate(
            tokens.T.unsqueeze(0),
            size=conditioner.pad_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T
    return _project_tokens(conditioner, tokens)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_condition_mlps(
    base_mlp: torch.nn.Module,
    specs: OrderedDict[str, Path],
    device: torch.device,
) -> OrderedDict[str, torch.nn.Module]:
    mlps: OrderedDict[str, torch.nn.Module] = OrderedDict()
    mlps["base"] = copy.deepcopy(base_mlp).to(device).eval().requires_grad_(False)
    for name, path in specs.items():
        payload = _load_checkpoint(path)
        state = payload["dit"] if isinstance(payload, dict) and "dit" in payload else payload
        prefix = "model.to_cond_embed."
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not selected:
            raise KeyError(f"{path} contains no {prefix}* tensors")
        module = copy.deepcopy(base_mlp).cpu()
        module.load_state_dict(selected, strict=True)
        mlps[name] = module.to(device).eval().requires_grad_(False)
        del payload, state, selected, module
    return mlps


def _append(storage: dict[str, dict[str, list[torch.Tensor]]], stage: str, view: str, value: torch.Tensor) -> None:
    storage.setdefault(stage, {}).setdefault(view, []).append(value.detach().float().cpu())


def _pool_transformed(
    hidden: torch.Tensor,
    masks: dict[str, torch.Tensor],
    conditioner: Any,
    mlps: OrderedDict[str, torch.nn.Module],
    storage: dict[str, dict[str, list[torch.Tensor]]],
) -> None:
    # Subset projections preserve token identities for content/semantic views.
    for view in ("content", "semantic"):
        projected = _project_tokens(conditioner, hidden[masks[view].to(hidden.device)])
        _append(storage, "projected", view, projected.float().mean(dim=0))
        for name, mlp in mlps.items():
            transformed = mlp(projected.to(next(mlp.parameters()).device, dtype=next(mlp.parameters()).dtype))
            _append(storage, f"dit:{name}", view, transformed.float().mean(dim=0))

    exact = _exact_projected_sequence(conditioner, hidden, masks["full"])
    _append(storage, "projected", "full", exact.float().mean(dim=0))
    _append(storage, "projected", "last", exact[-1].float())
    for name, mlp in mlps.items():
        transformed = mlp(exact.to(next(mlp.parameters()).device, dtype=next(mlp.parameters()).dtype))
        _append(storage, f"dit:{name}", "full", transformed.float().mean(dim=0))
        _append(storage, f"dit:{name}", "last", transformed[-1].float())


def _evaluate_storage(
    audio_storage: dict[str, dict[str, list[torch.Tensor]]],
    text_storage: dict[str, dict[str, list[torch.Tensor]]],
    audio_groups: list[str],
    text_groups: list[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stage in audio_storage:
        if stage not in text_storage:
            continue
        results[stage] = {}
        for view in audio_storage[stage]:
            if view not in text_storage[stage]:
                continue
            audio = torch.stack(audio_storage[stage][view])
            text = torch.stack(text_storage[stage][view])
            results[stage][view] = alignment_metrics(audio, text, audio_groups, text_groups)
    return results


def _metric_row(label: str, view: str, metric: dict[str, Any]) -> str:
    return (
        f"| {label} | {view} | {metric['matched_cosine']['mean']:.4f} | "
        f"{metric['unmatched_cosine']['mean']:.4f} | {metric['cosine_margin']:.4f} | "
        f"{metric['audio_to_text']['r@1']:.3f} | {metric['text_to_audio']['r@1']:.3f} | "
        f"{metric['linear_cka']:.4f} | {metric['rbf_mmd']:.4f} |"
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Audio/Text Condition Alignment Diagnostic",
        "",
        "This report compares masked pooled representations. It does **not** align audio and text tokens position by position.",
        "",
        "## Contract",
        "",
        f"- Audio groups: {result['contract']['n_audio_groups']}",
        f"- Text captions: {result['contract']['n_text_captions']}",
        f"- Qwen configured hidden state: {result['contract']['configured_hidden_index']}",
        f"- Audio bottleneck: low-pass {result['contract']['omni_downsample']}x, crop seconds={result['contract']['omni_seconds']}",
        "- Qwen and OmniConditioner projection are frozen; only checkpoint-specific DiT `to_cond_embed` differs.",
        "",
        "Views: `content` = audio tokens vs caption tokens; `semantic` = post-audio instruction span vs caption span; `full` = all valid tokens; `last` = final valid token.",
        "",
        "## Qwen layerwise",
        "",
        "| Layer | View | matched cosine | unmatched cosine | margin | A→T R@1 | T→A R@1 | linear CKA | RBF MMD |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stages = result["metrics"]
    qwen_keys = sorted(
        (key for key in stages if key.startswith("qwen:")),
        key=lambda key: int(key.split(":", 1)[1]),
    )
    for key in qwen_keys:
        layer = key.split(":", 1)[1]
        for view, metric in stages[key].items():
            lines.append(_metric_row(layer, view, metric))
    lines += [
        "",
        "## Projection and DiT condition embedding",
        "",
        "| Stage | View | matched cosine | unmatched cosine | margin | A→T R@1 | T→A R@1 | linear CKA | RBF MMD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, views in stages.items():
        if stage.startswith("qwen:"):
            continue
        for view, metric in views.items():
            lines.append(_metric_row(stage, view, metric))
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- A reconstruction loss reduction is not evidence that text-conditioned generation improved.",
        "- High matched cosine alone is insufficient; inspect the matched-unmatched margin and both retrieval directions.",
        "- `semantic` uses a fixed post-audio instruction whose hidden states are causally conditioned on audio; lexical template effects remain a confound.",
        "- CKA is computed across matched audio groups after averaging their positive captions, not across token positions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing existing output: {out}")
    out.mkdir(parents=True)
    started = time.time()
    specs = _parse_dit_specs(args.dit)
    manifest = Path(args.manifest)
    config_path = Path(args.model_config)
    base_ckpt = Path(args.base_ckpt)
    for path in (manifest, config_path, base_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = AudioManifestDataset(
        str(manifest), seconds=args.seconds, sr=args.sample_rate
    )
    groups = select_groups(dataset, args.max_groups, args.max_captions_per_group)
    audio_group_ids = [item["group_id"] for item in groups]
    text_records = [
        (item["group_id"], caption)
        for item in groups
        for caption in item["captions"]
    ]
    text_group_ids = [group for group, _ in text_records]

    device = torch.device(args.device)
    cond_device = torch.device(args.conditioner_device)
    model, _, _, _ = load_model_bundle(
        json.loads(config_path.read_text(encoding="utf-8")),
        str(base_ckpt),
        device=str(device),
        conditioner_device=str(cond_device),
        dtype=args.dtype,
    )
    model.eval().requires_grad_(False)
    conditioner = model.conditioner.conditioners["omni_prompt"]
    tokenizer = conditioner.processor.tokenizer
    audio_inventory = _audio_token_inventory(tokenizer)
    if not audio_inventory:
        raise RuntimeError("tokenizer exposes no token containing 'audio'")
    base_mlp = model.model.model.to_cond_embed
    mlps = build_condition_mlps(base_mlp, specs, device)

    audio_storage: dict[str, dict[str, list[torch.Tensor]]] = {}
    text_storage: dict[str, dict[str, list[torch.Tensor]]] = {}
    token_stats = {"audio": [], "text": []}
    hidden_count = None
    configured_index = None

    def encode_one(
        *,
        kind: str,
        prompt: str,
        semantic_text: str,
        audio: Any,
        storage: dict[str, dict[str, list[torch.Tensor]]],
    ) -> None:
        nonlocal hidden_count, configured_index
        inputs, ids, valid = _prepare_qwen_inputs(conditioner, prompt, audio, cond_device)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
            output = conditioner.model.thinker(
                **inputs, output_hidden_states=True, return_dict=True
            )
        sequence_length = output.hidden_states[0].shape[1]
        ids = _align_1d(ids, sequence_length, -1).long()
        valid = _align_1d(valid, sequence_length, False).bool()
        masks, stats = _record_masks(
            kind=kind,
            ids=ids,
            valid=valid,
            tokenizer=tokenizer,
            audio_token_ids=set(audio_inventory.values()),
            semantic_text=semantic_text,
        )
        token_stats[kind].append(stats)
        hidden_count = len(output.hidden_states)
        configured_index = conditioner.layer_idx % hidden_count
        for layer_index, layer_hidden in enumerate(output.hidden_states):
            hidden = layer_hidden[0]
            for view in VIEWS:
                _append(storage, f"qwen:{layer_index}", view, masked_pool(hidden, masks[view]))
        configured_hidden = output.hidden_states[configured_index][0]
        with torch.inference_mode():
            _pool_transformed(configured_hidden, masks, conditioner, mlps, storage)
        del output, inputs

    for position, item in enumerate(groups, start=1):
        clip = dataset[item["row_index"]]
        cond_audio = _apply_bottleneck(
            clip.audio_omni,
            downsample=args.omni_downsample,
            seconds=args.omni_seconds,
        ).cpu().numpy()
        encode_one(
            kind="audio",
            prompt=build_reconstruct_prompt(args.reca_instruction),
            semantic_text=args.reca_instruction,
            audio=cond_audio,
            storage=audio_storage,
        )
        print(f"[audio] {position}/{len(groups)}", flush=True)

    for position, (_, caption) in enumerate(text_records, start=1):
        encode_one(
            kind="text",
            prompt=build_prompt(args.task, caption),
            semantic_text=caption,
            audio=None,
            storage=text_storage,
        )
        print(f"[text] {position}/{len(text_records)}", flush=True)

    metrics = _evaluate_storage(
        audio_storage, text_storage, audio_group_ids, text_group_ids
    )
    result = {
        "contract": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "model_config": str(config_path.resolve()),
            "base_ckpt": str(base_ckpt.resolve()),
            "dit_checkpoints": {name: str(path.resolve()) for name, path in specs.items()},
            "n_audio_groups": len(audio_group_ids),
            "n_text_captions": len(text_group_ids),
            "seconds": args.seconds,
            "omni_downsample": args.omni_downsample,
            "omni_seconds": args.omni_seconds,
            "task": args.task,
            "reca_instruction": args.reca_instruction,
            "seed": args.seed,
            "qwen_hidden_count": hidden_count,
            "configured_hidden_index": configured_index,
            "audio_token_inventory": audio_inventory,
            "elapsed_seconds": time.time() - started,
        },
        "token_stats": token_stats,
        "groups": groups,
        "metrics": metrics,
    }
    (out / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_report(out / "report.md", result)
    print(f"[done] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

