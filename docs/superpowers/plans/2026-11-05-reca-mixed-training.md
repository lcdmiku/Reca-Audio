# ReCA Mixed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, deploy, and launch a controlled Pure ReCA / T2A-only / Mix-1:1 experiment on a SHA-grouped held-out split.

**Architecture:** Keep the existing Audio-Omni model and online conditioning path. Add a single-placeholder prompt helper, a standalone deterministic split tool, small training-loop reproducibility/checkpoint improvements, and a guarded server launcher for the three disjoint-GPU arms.

**Tech Stack:** Python 3, standard-library `unittest`, PyTorch/torchaudio on the server, Bash, existing Audio-Omni code.

**Spec:** `docs/superpowers/specs/2026-11-05-reca-mixed-training-design.md`

## Global Constraints

- Do not edit server source files directly; deploy only committed local files with the documented hash checks.
- Freeze Qwen2.5-Omni, all non-DiT conditioners, and Audio VAE.
- Do not change hidden layer, token selection, pooling, bottleneck, DiT architecture, or flow target.
- Formal arms use the same base checkpoint, split, steps, optimizer, seed family, and generation protocol.
- Never overwrite an existing formal run or split directory.

## Review Focus

- ReCA instructions that already contain `<|AUDIO|>` must still produce exactly one placeholder.
- Duplicate waveform bytes under different filenames must never cross splits.
- Empty or non-positive mix weights must fail before training.
- Resume checkpoints must preserve fp32 DiT weights and scaler/task-count state.
- Launcher reruns must refuse non-empty output directories rather than mix experiments.

---

### Task 1: Enforce the ReCA prompt contract

**Files:**
- Modify: `Audio-Omni/audio_omni/prompts.py`
- Modify: `Audio-Omni/audio_omni/api.py`
- Modify: `src/reca_audio/train/conditioning.py`
- Create: `tests/test_reca_prompts.py`

**Interfaces:**
- Produces: `build_reconstruct_prompt(text: str = "") -> str` with exactly one `<|AUDIO|>`.
- Consumers: ReCA training metadata and `AudioOmni.reconstruct`.

- [ ] Write `unittest` cases for plain, empty, and already-prefixed instructions; import the real prompt module and assert exactly one marker.
- [ ] Run `python -m unittest tests.test_reca_prompts -v`; expect ImportError for the missing helper.
- [ ] Implement `build_reconstruct_prompt`, then route training and API reconstruction through it.
- [ ] Run the prompt tests; expect all cases to pass.
- [ ] Commit the prompt contract change.

### Task 2: Add deterministic SHA-grouped held-out splitting

**Files:**
- Create: `src/reca_audio/train/split_manifest.py`
- Create: `tests/test_split_manifest.py`

**Interfaces:**
- Consumes: source manifest rows with `audio` and `caption`/`prompt`.
- Produces: `build_split(source_manifest, output_dir, seed, ratios) -> dict` and the versioned artifacts defined by the design.

- [ ] Write tests using temporary WAV-like byte files, including two filenames with identical bytes; assert deterministic output, disjoint SHA sets, grouped duplicates, prompt IDs, and refusal to overwrite.
- [ ] Run `python -m unittest tests.test_split_manifest -v`; expect module import failure.
- [ ] Implement SHA grouping, deterministic allocation, symlink/reference creation, manifest hashing, and report output using the standard library.
- [ ] Run split tests; expect all cases to pass.
- [ ] Commit the split tool.

### Task 3: Make task sampling, logging, and checkpoints reproducible

**Files:**
- Create: `src/reca_audio/train/scheduling.py`
- Modify: `src/reca_audio/train/train.py`
- Modify: `src/reca_audio/train/conditioning.py`
- Create: `tests/test_training_schedule.py`

**Interfaces:**
- Produces: `choose_task_mode(task_mode, lambda_reca, lambda_t2a, draw) -> str` with validated weights.
- Training logs produce cumulative `reca_count`, `t2a_count`, and per-mode recent loss.
- Checkpoints preserve fp32 DiT state, optimizer, scaler, task counts, and available RNG states.

- [ ] Write pure scheduling tests for fixed modes, 1:1 boundaries, 4:1 boundaries, and invalid weights.
- [ ] Run `python -m unittest tests.test_training_schedule -v`; expect module import failure.
- [ ] Implement the pure scheduler and use separate data/task/flow generators plus seeded prompt RNG in `train.py`.
- [ ] Add per-mode counters/loss logging and fp32/scaler/RNG checkpoint fields with backward-compatible resume.
- [ ] Run scheduling tests and an AST parse of all modified Python files; expect success.
- [ ] On the server, run a 2-step smoke for `reca`, `t2a`, and `mix`; inspect args, counts, loss, checkpoint dtype, and resume one extra step.
- [ ] Commit the training-state change.

### Task 4: Make checkpoint merge strict

**Files:**
- Modify: `src/reca_audio/train/merge_ckpt.py`
- Create: `tests/test_merge_mapping.py`

**Interfaces:**
- Produces: a pure mapping validator that returns complete prefixed replacements or raises on missing/shape mismatch.
- Consumer: merge CLI before writing the output checkpoint.

- [ ] Write tests for complete mapping, missing key, and shape mismatch with lightweight fake tensor shapes.
- [ ] Run `python -m unittest tests.test_merge_mapping -v`; expect failure because strict mapping is absent.
- [ ] Implement strict validation and keep a non-strict option only when explicitly requested.
- [ ] Run merge tests; expect all cases to pass.
- [ ] Commit the strict merge change.

### Task 5: Add the guarded formal experiment launcher

**Files:**
- Create: `runs/scripts/train_reca_matrix.sh`
- Modify: `README.md`
- Modify: `docs/reca_training.md`

**Interfaces:**
- Consumes: generated `train_manifest.jsonl`, original checkpoint, GPU pairs, and shared training hyperparameters.
- Produces: three isolated run directories with PIDs and launcher logs.

- [ ] Implement `smoke`, `launch`, and `status` actions; `launch` validates split report, free GPU pairs, checkpoint presence, and empty output directories.
- [ ] Run `bash -n runs/scripts/train_reca_matrix.sh`; expect success.
- [ ] Document the internal held-out protocol, exact matrix, and status commands.
- [ ] Commit the launcher and docs.

### Task 6: Verify, deploy, prepare data, and start training

**Files:**
- Verify all files changed by Tasks 1-5.
- Server-generated data: `runs/Benchmark/_data/audiocaps/reca_split_v1/`.
- Server-generated runs: `runs/train/reca_matrix_v1/{pure,t2a,mix}`.

**Interfaces:**
- Consumes: clean committed branch and documented deployment scripts.
- Produces: verified deployed commit, split report, smoke evidence, and three running formal jobs.

- [ ] Run the complete local `unittest` suite, AST parse, and Bash syntax check.
- [ ] Review the branch diff against the design and resolve Critical/Important findings.
- [ ] Deploy the clean branch with manifest/hash verification; confirm `DEPLOYED_COMMIT` matches the local commit.
- [ ] Generate `reca_split_v1`; verify zero ID/path/SHA intersections and record counts/hashes.
- [ ] Run all three smoke modes and a resume check.
- [ ] Launch Pure on GPUs 0,1; T2A-only on 2,3; Mix-1:1 on 4,5.
- [ ] Confirm each process is alive, GPU memory is allocated to the intended pair, and at least the first logged losses/task counts are finite.

