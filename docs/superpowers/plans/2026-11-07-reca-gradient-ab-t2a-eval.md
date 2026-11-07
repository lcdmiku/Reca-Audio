# ReCA Gradient A/B T2A Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducibly compare Base, accumulation-1 A, and accumulation-4 B on a deterministic 89-clip SHA-disjoint AudioCaps T2A diagnostic without retaining merged full checkpoints.

**Architecture:** A focused Python utility derives one prompt/reference pair per unique test-audio SHA and records provenance. A guarded Bash launcher validates inputs and GPUs, evaluates Base/A/B sequentially with identical generation settings, strictly merges only the active A/B DiT into a `/tmp` checkpoint, scores complete outputs, and removes temporary checkpoints through a cleanup trap.

**Tech Stack:** Python 3, `unittest`, PyTorch checkpoint merger, Bash, existing AudioCaps benchmark scripts, six NVIDIA GPUs.

**Spec:** `docs/superpowers/specs/2026-11-07-reca-gradient-ab-t2a-eval-design.md`

## Global Constraints

- No training and no changes to model, loss, Prompt semantics, optimizer, or checkpoint contents.
- Models are Base, A exposure 400, and B exposure 400 only.
- Use exactly 89 first-occurrence unique-SHA AudioCaps test items from the disjoint split.
- Use seed 1234, 100 ODE steps, CFG 6.0, 10 seconds, and fp16 for every model.
- Use three two-GPU processes on physical GPUs 1–6 after an occupancy check.
- Keep at most one merged 21 GB checkpoint in `/tmp`; always delete it on success or failure.
- Preserve the two canonical 12 GB A/B DiT checkpoints and keep user-home usage below 400 GB.
- Treat CLAP text/audio as primary; treat FAD/KL/IS/FD on 89 clips as diagnostic.

## Review Focus

- Duplicate audio content under different paths must produce one item, selected by first manifest occurrence.
- Duplicate output IDs for different selected hashes must fail before generation.
- Missing audio, non-zero train/test SHA overlap, or a final count other than 89 must fail closed.
- Existing/non-empty output directories must not be overwritten or silently resumed.
- Interruptions during merge, generation, or scoring must remove only the exact `mktemp` directory and never either canonical DiT checkpoint.

---

### Task 1: Deterministic Unique-SHA Evaluation Subset

**Files:**
- Create: `runs/Benchmark/AudioCaps/build_unique_sha_subset.py`
- Create: `tests/test_unique_sha_subset.py`

**Interfaces:**
- Consumes: test manifest JSONL rows with `id`, `prompt`, and `audio`; split report JSON with train/test intersection counts.
- Produces: `build_unique_sha_subset(manifest: Path, split_report: Path, prompts_out: Path, refs_out: Path, metadata_out: Path, expected_count: int) -> dict` plus a CLI with matching arguments.

- [ ] **Step 1: Write tests for first-occurrence SHA grouping and provenance**

Create temporary audio files where two paths have identical bytes. Assert stable first-occurrence selection, prompt order, absolute reference symlinks, SHA values, source/prompt manifest hashes, and the expected output count.

- [ ] **Step 2: Write fail-closed tests**

Assert rejection of missing audio, duplicate selected IDs, non-zero train/test ID or SHA intersection, non-empty output targets, and an expected-count mismatch.

- [ ] **Step 3: Run tests and confirm the new module is absent**

Run: `python -m unittest tests.test_unique_sha_subset -v`

Expected: FAIL because `build_unique_sha_subset.py` does not exist.

- [ ] **Step 4: Implement the builder and CLI**

Hash audio by streaming SHA256, retain the first row per hash in manifest order, create one absolute symlink named from each selected ID, write prompts and metadata atomically, and validate the split report and exact count before publishing outputs.

- [ ] **Step 5: Run focused and regression tests**

Run: `python -m unittest tests.test_unique_sha_subset tests.test_split_manifest -v`

Expected: PASS.

- [ ] **Step 6: Commit the subset builder**

```bash
git add runs/Benchmark/AudioCaps/build_unique_sha_subset.py tests/test_unique_sha_subset.py
git commit -m "feat: build unique-audio T2A eval subset"
```

### Task 2: Guarded Sequential Base/A/B Evaluator

**Files:**
- Create: `runs/scripts/eval_gradient_ab_t2a.sh`
- Create: `tests/test_gradient_ab_eval_launcher.py`

**Interfaces:**
- Consumes: Task 1 CLI, Base full checkpoint, A/B DiT checkpoints, existing strict merger, `run_shards.sh`, and AudioCaps scorer.
- Produces: `bash runs/scripts/eval_gradient_ab_t2a.sh {dry-run|run|status}`; output tree containing subset provenance, per-model WAV/manifests/results, pipeline log, and run contract.

- [ ] **Step 1: Write launcher contract tests**

In `dry-run`, assert the exact Base/A/B order, GPUs `1,2,3,4,5,6`, three two-GPU shards, seed 1234, steps 100, CFG 6.0, 10 seconds, fp16, strict A/B merge commands, and the same subset paths for all models.

- [ ] **Step 2: Write safety tests**

Assert rejection of missing checkpoints, non-empty formal output, an invalid subset count, and occupied requested GPUs. Use fake merger/generator/scorer commands to assert that normal failure and signal-triggered exit remove the unique temp directory without touching canonical checkpoint files.

- [ ] **Step 3: Run tests and confirm the launcher is absent**

Run: `python -m unittest tests.test_gradient_ab_eval_launcher -v`

Expected: FAIL because the launcher does not exist.

- [ ] **Step 4: Implement launcher validation and contracts**

Add `dry-run`, `run`, and `status`; resolve the conda Python; validate split/checkpoints/output/GPU occupancy; create the Task 1 subset; write a self-contained run contract; and refuse partial or pre-existing outputs.

- [ ] **Step 5: Implement sequential evaluation and cleanup**

Evaluate Base directly. For A then B, create a strict merged checkpoint under one `mktemp -d` root, generate 89 items, require 89 success records and files, score with CLAP/CLAP-FAD/PANNs, delete the merged file before advancing, and install an exact-path cleanup trap.

- [ ] **Step 6: Run focused launcher tests**

Run: `python -m unittest tests.test_gradient_ab_eval_launcher -v`

Expected: PASS.

- [ ] **Step 7: Run all repository tests and shell syntax checks**

Run: `python -m unittest discover -s tests -v`

Run: `bash -n runs/scripts/eval_gradient_ab_t2a.sh`

Expected: all tests PASS and Bash exits 0.

- [ ] **Step 8: Commit the evaluator**

```bash
git add runs/scripts/eval_gradient_ab_t2a.sh tests/test_gradient_ab_eval_launcher.py
git commit -m "feat: evaluate gradient A/B T2A preservation"
```

### Task 3: Deploy and Server Acceptance

**Files:**
- Verify: `README.md`
- Verify: `docs/deploy.md`
- Runtime output: `runs/Benchmark/output/GradientAB_T2A_89_<commit>/`

**Interfaces:**
- Consumes: committed Tasks 1–2 and the documented deployment pipeline.
- Produces: a verified remote commit marker and a server dry-run acceptance record.

- [ ] **Step 1: Verify clean local state and deployment scope**

Run: `git status --short` and `git log -1 --oneline`.

Expected: clean worktree; latest commit contains the evaluator.

- [ ] **Step 2: Deploy with the repository deployment script**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File runs/scripts/deploy.ps1 -Push`

Expected: remote hashes verify and `DEPLOYED_COMMIT` records the evaluator commit.

- [ ] **Step 3: Run remote focused tests or documented fallback checks**

Run the focused Python tests when the server test environment supports them; otherwise run the builder against temporary fixtures and `bash -n` explicitly, recording why the fallback was used.

- [ ] **Step 4: Run server dry-run acceptance**

Invoke the launcher with the formal paths and output root in `dry-run` mode. Confirm 89 selected hashes, all checkpoint paths, generation controls, GPU mapping, output refusal rules, `/tmp` cleanup target, free disk, home usage, and GPU occupancy.

### Task 4: Formal Diagnostic Evaluation and Report

**Files:**
- Runtime output: `runs/Benchmark/output/GradientAB_T2A_89_<commit>/`
- Modify: `docs/result.md`

**Interfaces:**
- Consumes: Task 3 accepted launcher.
- Produces: complete Base/A/B metrics and a source-fact versus research-interpretation report.

- [ ] **Step 1: Start the synchronous evaluator under a durable remote log**

Start one formal run only after rechecking GPUs 1–6 and disk usage. Record PID, start time, output path, and pipeline log path.

- [ ] **Step 2: Monitor meaningful transitions**

Check after subset completion, after each model generation/score, and on any failure. Do not start training or another evaluator. If SSH connectivity itself fails, stop and report it to the user without retrying.

- [ ] **Step 3: Validate final artifacts**

Require Base/A/B each to have 89 successful generated files, complete manifests, and metric JSON. Confirm no merged checkpoint remains in `/tmp`, both formal DiT checkpoints remain intact, and user-home usage stays below 400 GB.

- [ ] **Step 4: Analyze the controlled comparison**

Tabulate Base/A/B CLAP, CLAP-FAD, KL, IS, and FD; report deltas B-A, A-Base, and B-Base. State that A/B are both 211-ReCA/189-T2A mixed objectives at equal exposure but unequal update counts, and that 89-item distributional metrics are diagnostic.

- [ ] **Step 5: Record results and commit**

Update `docs/result.md` with paths, controls, metrics, timing, storage checks, conclusions, and the next decision gate. Run `git diff --check`, then commit the report without deploying it unless later code changes require deployment.
