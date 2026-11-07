# ReCA Gradient Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, exposure-aligned ReCA/T2A gradient diagnostic and small A/B runner, validate it locally, and stop after a server smoke test.

**Architecture:** Keep formal training unchanged. Add pure scheduling/statistics helpers and a dedicated controlled-experiment entry point that reuses the existing data, conditioning, VAE, flow-loss, checkpoint, and optimizer construction paths. Persist a pregenerated micro-batch schedule so A and B consume byte-identical experiment records.

**Tech Stack:** Python 3.11, PyTorch, pytest, existing Audio-Omni/ReCA training modules, Bash launcher.

**Spec:** `docs/superpowers/specs/2026-11-07-reca-gradient-stability-design.md`

## Global Constraints

- Do not change LR peak (`2e-5`), optimizer (Adafactor), loss, Prompt, or model structure.
- Schedule LR by cumulative micro-batch exposure with 20 warmup exposures out of 400.
- Use the same pregenerated micro-batch and task sequence for A and B.
- Do not start the 400-exposure A/B run before explicit user confirmation after smoke.
- Do not mutate parameters, optimizer/scaler state, or caller RNG during diagnostics.
- Keep `/home/linchudeng` below 400 GB and remove disposable smoke model weights after validation.

## Review Focus

- Accumulation groups at the tail: reject schedules not divisible by the arm's accumulation factor.
- LR indexing off by one: assert the first, warmup-boundary, and last exposure values.
- Diagnostic side effects: assert parameters, optimizer state, and RNG are byte-for-byte unchanged.
- Zero-norm gradients: return a defined nullable cosine without division by zero.
- Resume/order mismatch: persist and verify the pregenerated schedule digest before consuming it.

---

### Task 1: Exposure Schedule and Deterministic Records

**Files:**
- Create: `src/reca_audio/train/gradient_experiment.py`
- Create: `tests/test_gradient_experiment.py`

**Interfaces:**
- Produces: `exposure_lr(exposure: int, total_exposures: int, warmup_exposures: int, peak_lr: float) -> float`
- Produces: `update_lr(start_exposure: int, accumulation: int, ...) -> float`
- Produces: deterministic schedule generation/loading with a stable SHA256 digest.

- [ ] **Step 1: Write failing tests for LR boundary values, four-exposure averaging, deterministic records, divisibility validation, and digest verification.**
- [ ] **Step 2: Run `python -m pytest tests/test_gradient_experiment.py -v` and confirm failures are caused by missing interfaces.**
- [ ] **Step 3: Implement the minimal pure helpers in `gradient_experiment.py`.**
- [ ] **Step 4: Re-run the focused tests and confirm they pass.**

### Task 2: Streaming Gradient and Drift Statistics

**Files:**
- Modify: `src/reca_audio/train/gradient_experiment.py`
- Modify: `tests/test_gradient_experiment.py`

**Interfaces:**
- Produces: streaming gradient snapshot, norm/cosine/variance reducers, block grouping, clipping metrics, and parameter-drift reducers.

- [ ] **Step 1: Write failing tensor-level tests for norm, cosine, variance, clipping scale, block aggregation, drift, zero norm, and absence of input mutation.**
- [ ] **Step 2: Run the focused tests and verify expected failures.**
- [ ] **Step 3: Implement minimal streaming reducers without concatenating the full model gradient.**
- [ ] **Step 4: Re-run focused tests and the full pytest suite.**

### Task 3: Side-effect-free Fixed Diagnostics

**Files:**
- Modify: `src/reca_audio/train/gradient_experiment.py`
- Modify: `tests/test_gradient_experiment.py`

**Interfaces:**
- Produces: an RNG-preserving diagnostic context and fixed-batch gradient evaluation API used by the runner.

- [ ] **Step 1: Write failing tests proving diagnostic execution restores Python/Torch RNG, parameters, existing gradients, and optimizer state.**
- [ ] **Step 2: Run focused tests and verify failures.**
- [ ] **Step 3: Implement the diagnostic context/evaluator using isolated seeds and restored state.**
- [ ] **Step 4: Re-run focused and full tests.**

### Task 4: Controlled A/B Runner and Launcher

**Files:**
- Create: `src/reca_audio/train/run_gradient_ab.py`
- Create: `runs/scripts/train_reca_gradient_ab.sh`
- Modify: `tests/test_gradient_experiment.py`
- Modify: `runs/scripts/deploy.ps1`

**Interfaces:**
- Consumes: scheduling/statistics/diagnostic APIs from Tasks 1-3 and existing ReCA model/data/flow functions.
- Produces: `schedule.jsonl`, `schedule.sha256`, per-arm `metrics.jsonl`, diagnostic JSON, checkpoints at exposure milestones, and launcher status output.

- [ ] **Step 1: Write failing CLI/launcher contract tests for dry-run, shared schedule path, exposure counts, A/B update counts, diagnostic milestones, and Adafactor flags.**
- [ ] **Step 2: Run tests and verify expected failures.**
- [ ] **Step 3: Implement the runner and guarded launcher; require explicit `smoke` or `run`, with `run` refusing unless separately enabled.**
- [ ] **Step 4: Re-run focused tests, full pytest, and `git diff --check`.**

### Task 5: Deploy and Server Smoke

**Files:**
- Modify: `docs/reca_training.md`
- Modify: `docs/result.md` only after smoke results exist.

**Interfaces:**
- Consumes: deploy script and smoke launcher.
- Produces: smoke logs demonstrating static diagnostics and both accumulation paths without a formal A/B run.

- [ ] **Step 1: Deploy only the versioned source/scripts/tests through the documented deploy path.**
- [ ] **Step 2: Run a tiny server smoke with a shared pregenerated schedule and validate output schemas, update counts, LR averaging, checkpoint reload, and zero unintended GPU processes.**
- [ ] **Step 3: Delete disposable smoke checkpoint weights, record retained logs/metrics and disk usage, and keep user storage below 400 GB.**
- [ ] **Step 4: Update documentation with smoke evidence and exact artifact paths.**
- [ ] **Step 5: Run fresh local verification, commit the implementation, and stop for user confirmation before any 400-exposure A/B run.**

