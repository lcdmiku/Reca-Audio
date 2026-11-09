# ReCA Full-duration B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled 10-second Pure/T2A/Mix matrix using the exact B optimization regime.

**Architecture:** Parameterize only the target clip duration in the guarded B launcher, leaving its default at 5 seconds. Add a Pure-only smoke selector so one accumulation-4 update can establish V100 feasibility before the formal three-arm launch.

**Tech Stack:** Bash, existing Python gradient runner, pytest static launcher test.

**Spec:** `docs/superpowers/specs/2026-11-09-reca-full-duration-b-design.md`

## Global Constraints

- Do not change optimizer, LR, prompt, low-pass, model, manifest, task schedule, or exposure count.
- Preserve the existing 5-second B baseline and outputs.
- Stop after a Pure 4-exposure OOM; do not silently add a second experimental variable.
- Use a new versioned output root and GPUs confirmed free at launch time.

## Review Focus

- The Bash special variable `SECONDS` must not be used.
- A custom duration must reach `run_gradient_ab --seconds` in smoke and formal modes.
- Pure-only smoke must not create T2A/Mix outputs.
- Formal mode must remain three parallel arms with identical schedules.
- Existing default behavior must remain 5 seconds.

---

### Task 1: Parameterize duration and minimal smoke selection

**Files:**
- Modify: `runs/scripts/train_reca_long_matrix_b.sh`
- Modify: `tests/test_long_matrix_b_launcher.py`

- [ ] Update the focused test to require `CLIP_SECONDS` default 5 and `SMOKE_ARMS` default `pure t2a mix`; run it and observe failure.
- [ ] Pass `CLIP_SECONDS` to `--seconds` and loop over selected smoke arms without changing formal behavior.
- [ ] Run only `tests/test_long_matrix_b_launcher.py` and commit.

### Task 2: Deploy and run

**Files:**
- Remote output: `runs/train/reca_long_matrix_b_10s_v1/`
- Remote smoke: `runs/train/reca_long_matrix_b_10s_smoke/`

- [ ] Deploy through the hash-verified project process.
- [ ] Run Pure-only four-exposure smoke at 10 seconds on one free GPU pair.
- [ ] If smoke succeeds, delete the disposable smoke checkpoint and launch Pure/T2A/Mix formal training on three free GPU pairs.
- [ ] Report PIDs, logs, output paths, GPU assignment, and estimated completion time.
