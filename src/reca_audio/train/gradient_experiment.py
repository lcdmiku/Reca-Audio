"""Pure helpers for exposure-aligned ReCA gradient experiments."""
from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import torch


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


def arm_spec(
    arm: str,
    total_exposures: int,
    warmup_exposures: int,
    peak_lr: float,
) -> dict[str, Any]:
    if arm not in {"a", "b"}:
        raise ValueError("arm must be 'a' or 'b'")
    accumulation = 1 if arm == "a" else 4
    validate_experiment_shape(total_exposures, accumulation)
    updates = []
    for start in range(1, total_exposures + 1, accumulation):
        updates.append(
            {
                "start_exposure": start,
                "end_exposure": start + accumulation - 1,
                "lr": update_lr(
                    start, accumulation, total_exposures, warmup_exposures, peak_lr
                ),
            }
        )
    return {
        "arm": arm,
        "accumulation": accumulation,
        "total_exposures": total_exposures,
        "optimizer_updates": len(updates),
        "updates": updates,
    }


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


def block_name(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    for marker in ("transformer_blocks", "blocks", "layers"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts) and parts[index + 1].isdigit():
                return ".".join(parts[: index + 2])
    return parts[0]


def _tensor_items(
    tensors: Mapping[str, torch.Tensor | None],
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in tensors.items():
        if tensor is not None:
            yield name, tensor


def gradient_cosine(
    left: Mapping[str, torch.Tensor | None],
    right: Mapping[str, torch.Tensor | None],
) -> float | None:
    if left.keys() != right.keys():
        raise ValueError("gradient mappings must have identical parameter names")
    dot = left_sq = right_sq = 0.0
    for name, left_tensor in _tensor_items(left):
        right_tensor = right[name]
        if right_tensor is None:
            raise ValueError(f"gradient presence differs for {name}")
        lhs = left_tensor.detach().to(dtype=torch.float64)
        rhs = right_tensor.detach().to(device=lhs.device, dtype=torch.float64)
        dot += torch.sum(lhs * rhs).item()
        left_sq += torch.sum(lhs.square()).item()
        right_sq += torch.sum(rhs.square()).item()
    if left_sq == 0.0 or right_sq == 0.0:
        return None
    return dot / math.sqrt(left_sq * right_sq)


def block_gradient_norms(
    gradients: Mapping[str, torch.Tensor | None],
) -> dict[str, float]:
    squares: dict[str, float] = {}
    for name, tensor in _tensor_items(gradients):
        key = block_name(name)
        squares[key] = squares.get(key, 0.0) + torch.sum(
            tensor.detach().to(dtype=torch.float64).square()
        ).item()
    return {key: math.sqrt(value) for key, value in squares.items()}


def block_gradient_cosines(
    left: Mapping[str, torch.Tensor | None],
    right: Mapping[str, torch.Tensor | None],
) -> dict[str, float | None]:
    if left.keys() != right.keys():
        raise ValueError("gradient mappings must have identical parameter names")
    grouped_left: dict[str, dict[str, torch.Tensor | None]] = {}
    grouped_right: dict[str, dict[str, torch.Tensor | None]] = {}
    for name in left:
        key = block_name(name)
        grouped_left.setdefault(key, {})[name] = left[name]
        grouped_right.setdefault(key, {})[name] = right[name]
    return {
        key: gradient_cosine(grouped_left[key], grouped_right[key])
        for key in grouped_left
    }


class GradientMoments:
    """Exact vector moments while retaining only one running gradient sum."""

    def __init__(self) -> None:
        self.count = 0
        self._sum: dict[str, torch.Tensor] = {}
        self._sum_sample_sq_norm = 0.0
        self._sum_sample_norm = 0.0
        self._block_sum_sample_sq_norm: dict[str, float] = {}

    def add(self, gradients: Mapping[str, torch.Tensor | None]) -> None:
        sample_sq = 0.0
        present = dict(_tensor_items(gradients))
        if self.count and self._sum.keys() != present.keys():
            raise ValueError("all gradient samples must have identical parameter names")
        for name, tensor in present.items():
            detached = tensor.detach().to(device="cpu", dtype=torch.float32)
            tensor_sq = torch.sum(detached.to(torch.float64).square()).item()
            sample_sq += tensor_sq
            key = block_name(name)
            self._block_sum_sample_sq_norm[key] = (
                self._block_sum_sample_sq_norm.get(key, 0.0) + tensor_sq
            )
            if name not in self._sum:
                self._sum[name] = detached.clone()
            else:
                self._sum[name].add_(detached)
        self.count += 1
        self._sum_sample_sq_norm += sample_sq
        self._sum_sample_norm += math.sqrt(sample_sq)

    def summary(self) -> dict[str, float | int]:
        if not self.count:
            raise ValueError("no gradient samples were added")
        mean_sq = sum(
            torch.sum((value.to(torch.float64) / self.count).square()).item()
            for value in self._sum.values()
        )
        variance = max(0.0, self._sum_sample_sq_norm / self.count - mean_sq)
        return {
            "count": self.count,
            "mean_gradient_norm": math.sqrt(mean_sq),
            "mean_sample_norm": self._sum_sample_norm / self.count,
            "gradient_variance": variance,
        }

    def block_summary(self) -> dict[str, dict[str, float]]:
        if not self.count:
            raise ValueError("no gradient samples were added")
        block_mean_sq: dict[str, float] = {}
        for name, value in self._sum.items():
            key = block_name(name)
            block_mean_sq[key] = block_mean_sq.get(key, 0.0) + torch.sum(
                (value.to(torch.float64) / self.count).square()
            ).item()
        return {
            key: {
                "mean_gradient_norm": math.sqrt(mean_sq),
                "gradient_variance": max(
                    0.0,
                    self._block_sum_sample_sq_norm[key] / self.count - mean_sq,
                ),
            }
            for key, mean_sq in block_mean_sq.items()
        }


def clipping_metrics(pre_clip_norm: float, threshold: float) -> dict[str, float]:
    if pre_clip_norm < 0 or threshold <= 0:
        raise ValueError("norm must be non-negative and threshold must be positive")
    scale = min(1.0, threshold / max(pre_clip_norm, 1e-30))
    return {
        "pre_clip_norm": pre_clip_norm,
        "post_clip_norm": pre_clip_norm * scale,
        "clip_scale": scale,
        "clipped": float(pre_clip_norm > threshold),
    }


def collect_gradient_metrics(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    threshold: float,
) -> dict[str, Any]:
    """Reduce live ``.grad`` tensors to scalars without returning references."""
    gradients = {name: parameter.grad for name, parameter in named_parameters}
    block_norms = block_gradient_norms(gradients)
    pre_clip_norm = math.sqrt(sum(value * value for value in block_norms.values()))
    metrics: dict[str, Any] = clipping_metrics(pre_clip_norm, threshold)
    metrics["block_gradient_norms"] = block_norms
    return metrics


def parameter_drift(
    baseline: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if baseline.keys() != current.keys():
        raise ValueError("baseline and current parameters must have identical names")
    global_delta_sq = global_base_sq = 0.0
    block_delta_sq: dict[str, float] = {}
    block_base_sq: dict[str, float] = {}
    for name, base in baseline.items():
        now = current[name]
        base64 = base.detach().to(device="cpu", dtype=torch.float64)
        now64 = now.detach().to(device="cpu", dtype=torch.float64)
        delta_sq = torch.sum((now64 - base64).square()).item()
        base_sq = torch.sum(base64.square()).item()
        global_delta_sq += delta_sq
        global_base_sq += base_sq
        key = block_name(name)
        block_delta_sq[key] = block_delta_sq.get(key, 0.0) + delta_sq
        block_base_sq[key] = block_base_sq.get(key, 0.0) + base_sq

    def relative(delta_sq: float, base_sq: float) -> float | None:
        return math.sqrt(delta_sq / base_sq) if base_sq else None

    return {
        "global_relative_l2": relative(global_delta_sq, global_base_sq),
        "blocks": {
            key: relative(block_delta_sq[key], block_base_sq[key])
            for key in block_delta_sq
        },
    }


@contextmanager
def isolated_rng(seed: int) -> Iterator[None]:
    """Temporarily seed Python/Torch RNGs and restore their exact states."""
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def diagnostic_gradient(
    model: torch.nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    """Evaluate one gradient without touching ``.grad`` or optimizer state."""
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    with isolated_rng(seed):
        loss = loss_closure()
        if loss.ndim != 0:
            raise ValueError("diagnostic loss must be scalar")
        gradients = torch.autograd.grad(
            loss,
            [parameter for _, parameter in named_parameters],
            allow_unused=True,
        )
    return {
        "loss": float(loss.detach().to(device="cpu", dtype=torch.float64).item()),
        "gradients": {
            name: (
                gradient.detach().to(device="cpu", dtype=torch.float32).clone()
                if gradient is not None
                else None
            )
            for (name, _), gradient in zip(named_parameters, gradients)
        },
    }
