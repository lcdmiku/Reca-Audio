"""Pure task-scheduling helpers for ReCA/T2A mixed training."""
from __future__ import annotations


def choose_task_mode(
    task_mode: str,
    lambda_reca: float,
    lambda_t2a: float,
    draw: float,
) -> str:
    """Choose one batch mode from a caller-supplied uniform draw in [0, 1)."""
    if task_mode not in {"reca", "t2a", "mix"}:
        raise ValueError(f"unknown task mode: {task_mode!r}")
    if task_mode != "mix":
        return task_mode
    if lambda_reca < 0 or lambda_t2a < 0 or lambda_reca + lambda_t2a <= 0:
        raise ValueError("mix weights must be non-negative and have a positive sum")
    if not 0.0 <= draw < 1.0:
        raise ValueError(f"draw must be in [0, 1), got {draw}")
    p_reca = lambda_reca / (lambda_reca + lambda_t2a)
    return "reca" if draw < p_reca else "t2a"

