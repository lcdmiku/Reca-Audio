"""Pure helpers for exposure-aligned ReCA gradient experiments."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


def exposure_lr(
    exposure: int,
    total_exposures: int,
    warmup_exposures: int,
    peak_lr: float,
) -> float:
    """Return the LR at a one-based micro-batch exposure position."""
    if total_exposures <= 0:
        raise ValueError("total_exposures must be positive")
    if not 1 <= exposure <= total_exposures:
        raise ValueError(f"exposure must be in [1, {total_exposures}], got {exposure}")
    if not 0 <= warmup_exposures < total_exposures:
        raise ValueError("warmup_exposures must be in [0, total_exposures)")
    if peak_lr < 0:
        raise ValueError("peak_lr must be non-negative")
    if warmup_exposures and exposure <= warmup_exposures:
        return peak_lr * exposure / warmup_exposures
    progress = (exposure - warmup_exposures) / (total_exposures - warmup_exposures)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def update_lr(
    start_exposure: int,
    accumulation: int,
    total_exposures: int,
    warmup_exposures: int,
    peak_lr: float,
) -> float:
    """Average the schedule values represented by one optimizer update."""
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    values = [
        exposure_lr(position, total_exposures, warmup_exposures, peak_lr)
        for position in range(start_exposure, start_exposure + accumulation)
    ]
    return sum(values) / len(values)


def validate_experiment_shape(total_exposures: int, accumulation: int) -> None:
    if total_exposures <= 0 or accumulation <= 0:
        raise ValueError("total_exposures and accumulation must be positive")
    if total_exposures % accumulation:
        raise ValueError("total_exposures must be divisible by accumulation")


def generate_schedule(
    total_exposures: int,
    manifest_size: int,
    seed: int,
    reca_probability: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate the immutable sample/task/RNG sequence shared by both arms."""
    if total_exposures <= 0 or manifest_size <= 0:
        raise ValueError("total_exposures and manifest_size must be positive")
    if not 0.0 <= reca_probability <= 1.0:
        raise ValueError("reca_probability must be in [0, 1]")
    rng = random.Random(seed)
    records = []
    for exposure in range(1, total_exposures + 1):
        records.append(
            {
                "exposure": exposure,
                "sample_index": rng.randrange(manifest_size),
                "task": "reca" if rng.random() < reca_probability else "t2a",
                "data_seed": rng.randrange(2**63),
                "flow_seed": rng.randrange(2**63),
                "prompt_seed": rng.randrange(2**63),
            }
        )
    return records


def _schedule_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")


def write_schedule(path: str | Path, records: Iterable[dict[str, Any]]) -> str:
    payload = _schedule_bytes(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_schedule(path: str | Path, expected_digest: str | None = None) -> list[dict[str, Any]]:
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(f"schedule digest mismatch: expected {expected_digest}, got {digest}")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
