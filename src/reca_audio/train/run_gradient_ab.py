"""Controlled equal-exposure gradient-accumulation experiment for ReCA."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence


def _helpers():
    path = Path(__file__).with_name("gradient_experiment.py")
    spec = importlib.util.spec_from_file_location("reca_gradient_experiment_helpers", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    spec.loader.exec_module(module)
    return module


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Controlled ReCA gradient A/B experiment")
    parser.add_argument("--arm", choices=["a", "b", "static"], default="a")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--diagnostic-manifest", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-digest", default="")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-formal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--total-exposures", type=int, default=400)
    parser.add_argument("--warmup-exposures", type=int, default=20)
    parser.add_argument("--peak-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--diagnostic-samples", type=int, default=8)
    parser.add_argument("--diagnostic-exposures", default="0,100,200,400")
    parser.add_argument("--save-exposures", default="400")
    parser.add_argument("--keep-final-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    parser.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conditioner-device", default="cuda:1")
    parser.add_argument("--vae-device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--omni-downsample", type=int, default=4)
    parser.add_argument("--omni-seconds", type=float, default=0.0)
    return parser.parse_args(argv)


def _positions(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def validate_args(args: argparse.Namespace) -> None:
    if args.total_exposures <= 0:
        raise ValueError("total-exposures must be positive")
    if not 0 <= args.warmup_exposures < args.total_exposures:
        raise ValueError("warmup-exposures must be in [0, total-exposures)")
    if args.prepare_only:
        _helpers().validate_experiment_shape(args.total_exposures, accumulation=4)
        return
    if args.arm in {"a", "b"} and args.total_exposures > 8 and not args.allow_formal:
        raise ValueError("formal training requires --allow-formal")
    if not args.prepare_only:
        if not args.out:
            raise ValueError("--out is required unless --prepare-only")
        if not args.diagnostic_manifest:
            raise ValueError("--diagnostic-manifest is required unless --prepare-only")
    accumulation = 1 if args.arm != "b" else 4
    _helpers().validate_experiment_shape(args.total_exposures, accumulation)
    for name, values in (
        ("diagnostic", _positions(args.diagnostic_exposures)),
        ("save", _positions(args.save_exposures)),
    ):
        if any(value < 0 or value > args.total_exposures for value in values):
            raise ValueError(f"{name} exposure lies outside the experiment")


def experiment_contract(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    if args.arm == "static":
        accumulation = 0
        updates = 0
    else:
        spec = _helpers().arm_spec(
            args.arm, args.total_exposures, args.warmup_exposures, args.peak_lr
        )
        accumulation = spec["accumulation"]
        updates = spec["optimizer_updates"]
    return {
        "arm": args.arm,
        "schedule": args.schedule,
        "total_exposures": args.total_exposures,
        "warmup_exposures": args.warmup_exposures,
        "peak_lr": args.peak_lr,
        "accumulation": accumulation,
        "optimizer_updates": updates,
        "equal_exposure_not_equal_updates": True,
    }


def adafactor_kwargs(lr: float, weight_decay: float) -> dict[str, Any]:
    return {
        "lr": lr,
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
        "weight_decay": weight_decay,
    }


def _manifest_size(path: str | Path) -> int:
    with Path(path).open(encoding="utf-8") as handle:
        count = sum(bool(line.strip()) for line in handle)
    if not count:
        raise ValueError(f"empty manifest: {path}")
    return count


def _prepare_schedule(args: argparse.Namespace) -> str:
    helpers = _helpers()
    records = helpers.generate_schedule(
        args.total_exposures, _manifest_size(args.manifest), args.seed
    )
    digest = helpers.write_schedule(args.schedule, records)
    Path(args.schedule).with_suffix(Path(args.schedule).suffix + ".sha256").write_text(
        digest + "\n", encoding="utf-8"
    )
    print(json.dumps({"schedule": args.schedule, "digest": digest, "records": len(records)}))
    return digest


def _average(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    return sum(kept) / len(kept) if kept else None


def _write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _run(args: argparse.Namespace) -> int:
    import torch
    from audio_omni.loader import load_model_bundle
    from transformers.optimization import Adafactor

    from reca_audio.train.data import AudioManifestDataset, collate_clips
    from reca_audio.train.flow import rectified_flow_loss
    from reca_audio.train.train import build_metadata, move_cond

    helpers = _helpers()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "contract.json").write_text(
        json.dumps(experiment_contract(args), indent=2, sort_keys=True), encoding="utf-8"
    )

    digest = args.schedule_digest
    digest_path = Path(args.schedule).with_suffix(Path(args.schedule).suffix + ".sha256")
    if not digest and digest_path.exists():
        digest = digest_path.read_text(encoding="utf-8").strip()
    if not digest:
        raise ValueError("schedule digest is required")
    schedule = helpers.load_schedule(args.schedule, expected_digest=digest)
    if len(schedule) != args.total_exposures:
        raise ValueError("schedule length does not match total exposures")

    device = torch.device(args.device)
    cond_device = torch.device(args.conditioner_device)
    vae_device = torch.device(args.vae_device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[build] arm={args.arm} DiT={device} conditioner={cond_device} VAE={vae_device}", flush=True)
    model, _, _, _ = load_model_bundle(
        json.loads(Path(args.model_config).read_text(encoding="utf-8")),
        args.ckpt,
        device=str(device),
        conditioner_device=str(cond_device),
        dtype=args.dtype,
    )
    model.conditioner_device = cond_device
    if vae_device != device:
        model.pretransform.to(vae_device)
    dit = model.model
    dit.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in dit.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in dit.parameters() if parameter.requires_grad]
    baseline = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, parameter in dit.named_parameters()
        if parameter.requires_grad
    }

    train_dataset = AudioManifestDataset(
        args.manifest, seconds=args.seconds, sr=args.sample_rate
    )
    diagnostic_dataset = AudioManifestDataset(
        args.diagnostic_manifest, seconds=args.seconds, sr=args.sample_rate
    )

    def prepare(dataset, sample_index: int, mode: str, seeds: dict[str, int]):
        with helpers.isolated_rng(seeds["data_seed"]):
            clip = dataset[sample_index % len(dataset)]
        batch = collate_clips([clip])
        audio = batch["audio"].to(device)
        with torch.no_grad():
            latents = model.pretransform.encode(audio.to(vae_device)).detach().to(device)
            metadata = [
                build_metadata(
                    batch,
                    0,
                    mode,
                    args,
                    cond_device,
                    prompt_rng=random.Random(seeds["prompt_seed"]),
                )
            ]
            cond_tensors = model.conditioner(metadata, cond_device)
            cond_inputs = model.get_conditioning_inputs(move_cond(cond_tensors, device))
        return latents, cond_inputs

    def loss_closure(latents, cond_inputs, flow_seed: int):
        generator = torch.Generator(device=device)
        generator.manual_seed(flow_seed)

        def closure():
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
                loss, _ = rectified_flow_loss(
                    dit, latents, cond_inputs, cfg_dropout_prob=0.0, generator=generator
                )
            return loss

        return closure

    def diagnostics(exposure: int) -> dict[str, Any]:
        reca_moments = helpers.GradientMoments()
        t2a_moments = helpers.GradientMoments()
        pair_cosines: list[float | None] = []
        block_cosines: dict[str, list[float | None]] = {}
        losses = {"reca": [], "t2a": []}
        sample_count = min(args.diagnostic_samples, len(diagnostic_dataset))
        for index in range(sample_count):
            seeds = {
                "data_seed": args.seed + 100_000 + index,
                "prompt_seed": args.seed + 200_000 + index,
                "flow_seed": args.seed + 300_000 + index,
            }
            results = {}
            for mode in ("reca", "t2a"):
                with helpers.isolated_rng(args.seed + 400_000 + index):
                    latents, cond_inputs = prepare(diagnostic_dataset, index, mode, seeds)
                    result = helpers.diagnostic_gradient(
                        dit,
                        loss_closure(latents, cond_inputs, seeds["flow_seed"]),
                        seed=seeds["flow_seed"],
                    )
                losses[mode].append(result["loss"])
                (reca_moments if mode == "reca" else t2a_moments).add(result["gradients"])
                results[mode] = result["gradients"]
                del latents, cond_inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pair_cosines.append(helpers.gradient_cosine(results["reca"], results["t2a"]))
            for block, value in helpers.block_gradient_cosines(
                results["reca"], results["t2a"]
            ).items():
                block_cosines.setdefault(block, []).append(value)
            del results
            gc.collect()

        return {
            "kind": "diagnostic",
            "arm": args.arm,
            "exposure": exposure,
            "samples": sample_count,
            "reca_loss": sum(losses["reca"]) / sample_count,
            "t2a_loss": sum(losses["t2a"]) / sample_count,
            "reca_gradients": reca_moments.summary(),
            "t2a_gradients": t2a_moments.summary(),
            "reca_blocks": reca_moments.block_summary(),
            "t2a_blocks": t2a_moments.block_summary(),
            "paired_gradient_cosine": _average(pair_cosines),
            "paired_block_cosines": {
                block: _average(values) for block, values in block_cosines.items()
            },
            "parameter_drift": helpers.parameter_drift(
                baseline,
                {
                    name: parameter
                    for name, parameter in dit.named_parameters()
                    if parameter.requires_grad
                },
            ),
        }

    metrics_path = out / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics:
        static_record = diagnostics(0)
        _write_jsonl(metrics, static_record)
        print(f"[diagnostic] exposure=0 cosine={static_record['paired_gradient_cosine']}", flush=True)
        if args.arm == "static":
            return 0

        accumulation = 1 if args.arm == "a" else 4
        optimizer = Adafactor(parameters, **adafactor_kwargs(args.peak_lr, args.weight_decay))
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.dtype == "float32")
        diagnostic_exposures = _positions(args.diagnostic_exposures)
        save_exposures = _positions(args.save_exposures)
        update_index = 0
        started = time.time()
        for start_index in range(0, args.total_exposures, accumulation):
            optimizer.zero_grad(set_to_none=True)
            group = schedule[start_index : start_index + accumulation]
            task_losses = {"reca": [], "t2a": []}
            for record in group:
                latents, cond_inputs = prepare(
                    train_dataset, record["sample_index"], record["task"], record
                )
                closure = loss_closure(latents, cond_inputs, record["flow_seed"])
                loss = closure()
                scaler.scale(loss / accumulation).backward()
                task_losses[record["task"]].append(float(loss.detach().cpu()))
                del loss, latents, cond_inputs

            first_exposure = group[0]["exposure"]
            exposure = group[-1]["exposure"]
            lr = helpers.update_lr(
                first_exposure,
                accumulation,
                args.total_exposures,
                args.warmup_exposures,
                args.peak_lr,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = lr
            scaler.unscale_(optimizer)
            gradient_map = {
                name: parameter.grad
                for name, parameter in dit.named_parameters()
                if parameter.requires_grad
            }
            block_norms = helpers.block_gradient_norms(gradient_map)
            pre_clip_norm = math.sqrt(sum(value * value for value in block_norms.values()))
            clip = helpers.clipping_metrics(pre_clip_norm, args.grad_clip)
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_index += 1
            record = {
                "kind": "optimizer_step",
                "arm": args.arm,
                "optimizer_step": update_index,
                "start_exposure": first_exposure,
                "exposure": exposure,
                "lr": lr,
                "accumulation": accumulation,
                "reca_loss": _average(task_losses["reca"]),
                "t2a_loss": _average(task_losses["t2a"]),
                "gradient": clip,
                "block_gradient_norms": block_norms,
                "elapsed_seconds": time.time() - started,
            }
            _write_jsonl(metrics, record)
            print(
                f"[step] arm={args.arm} update={update_index} exposure={exposure} "
                f"lr={lr:.3e} grad={pre_clip_norm:.4f}",
                flush=True,
            )

            if exposure in diagnostic_exposures:
                diagnostic_record = diagnostics(exposure)
                _write_jsonl(metrics, diagnostic_record)
                print(
                    f"[diagnostic] exposure={exposure} "
                    f"cosine={diagnostic_record['paired_gradient_cosine']}",
                    flush=True,
                )

            if exposure in save_exposures:
                checkpoint = out / f"dit_exposure{exposure:04d}.pt"
                payload = {
                    "dit": {
                        name: value.detach().to(device="cpu", dtype=torch.float32)
                        for name, value in dit.state_dict().items()
                    },
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "exposure": exposure,
                    "optimizer_step": update_index,
                    "schedule_digest": digest,
                    "contract": experiment_contract(args),
                }
                torch.save(payload, checkpoint)
                reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if reloaded["exposure"] != exposure or reloaded["schedule_digest"] != digest:
                    raise RuntimeError("checkpoint reload validation failed")
                _write_jsonl(
                    metrics,
                    {
                        "kind": "checkpoint",
                        "arm": args.arm,
                        "exposure": exposure,
                        "optimizer_step": update_index,
                        "path": str(checkpoint),
                        "reload_validated": True,
                    },
                )
                keep = args.keep_final_checkpoint and exposure == args.total_exposures
                if not keep:
                    checkpoint.unlink()
                del payload, reloaded
                gc.collect()

    print(f"[done] arm={args.arm} exposures={args.total_exposures} -> {out}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.prepare_only:
        _prepare_schedule(args)
        return 0
    if args.dry_run:
        print(json.dumps(experiment_contract(args), indent=2, sort_keys=True))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
