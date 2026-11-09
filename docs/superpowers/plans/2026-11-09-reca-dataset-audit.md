# ReCA Dataset Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, read-only audit of the exact AudioCaps data and online 5-second crops consumed by the completed ReCA long-matrix experiment.

**Architecture:** A standalone audit CLI reads manifests, audio headers/content, immutable training schedules, and benchmark references without importing or changing the training loop. It emits machine-readable CSV/JSON plus a Markdown report; optional local CLAP scoring measures caption similarity for full clips and the exact scheduled crops.

**Tech Stack:** Python 3, standard library, NumPy, soundfile, torchaudio, existing LAION-CLAP benchmark backend, pytest.

**Spec:** User request in the 2026-11-09 chat turn.

## Global Constraints

- Do not modify `src/reca_audio/train/*` or any model/training configuration.
- Do not start training or call `optimizer.step()`.
- Audit the exact manifests and schedules used by `reca_long_matrix_b_v1`.
- Treat caption-crop CLAP as a screening signal, not ground-truth annotation correctness.
- Keep temporary crop audio under `/tmp` and remove it after scoring.

## Review Focus

- Repeated AudioCaps caption rows must not be mistaken for unique audio files.
- Relative audio paths must resolve from the repository root, matching current training behavior.
- Crop starts must reproduce `random.seed(data_seed); random.randint(...)` after 44.1 kHz resampling.
- Short clips must be right-padded rather than assigned a negative crop start.
- Benchmark overlap must distinguish row ID, filename/stem, and byte SHA256.

---

### Task 1: Standalone audit engine

**Files:**
- Create: `tools/audit_reca_dataset.py`
- Create: `tests/test_audit_reca_dataset.py`

**Interfaces:**
- Consumes: source/split JSONL manifests, immutable schedule JSONL, audio paths, benchmark prompt/reference pairs.
- Produces: `summary.json`, `audio_inventory.csv`, `caption_crop_alignment.csv`, `problem_samples.csv`, and `report.md`.

- [ ] Write failing tests for path resolution, repeated-audio accounting, exact crop starts, short-clip padding, and overlap counts.
- [ ] Run `python -m pytest tests/test_audit_reca_dataset.py -q`; expect failure because the audit module does not exist.
- [ ] Implement the minimal CLI and pure helpers needed by those tests.
- [ ] Run the focused test and the repository test suite.
- [ ] Commit the audit engine.

### Task 2: Server audit and research report

**Files:**
- Create: `docs/audits/2026-11-09-reca-training-data-audit.md`
- Generated remotely: `runs/audits/reca_training_data_20261109/*`

**Interfaces:**
- Consumes: Task 1 CLI outputs and current checkout preprocessing code.
- Produces: dataset statistics, caption-crop alignment findings, issue list, split/benchmark leakage assessment, and an exact explanation of the 4x bottleneck.

- [ ] Deploy with the documented hash-verified process and run the audit without training.
- [ ] Verify row/audio counts, SHA intersections, schedule coverage, CLAP score distributions, and cleanup of temporary crops.
- [ ] Copy the generated report/problem list to durable project documentation and record artifact paths.
- [ ] Run `git diff --check` and focused tests, then commit the report.
