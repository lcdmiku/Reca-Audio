# ReCA Long Matrix B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a reproducible Pure/T2A/Mix matrix with 2000 micro-batch exposures and exact arm-B exposure-indexed accumulation.

**Architecture:** Extend the already-tested gradient experiment schedule helper and runner instead of approximating arm B through `train.py`. Add task-specific deterministic schedules and a six-GPU launcher that runs three independent two-GPU jobs while preserving the existing A/B interfaces.

**Tech Stack:** Python 3, PyTorch, pytest, Bash, Adafactor, six NVIDIA V100 GPUs

**Spec:** `docs/superpowers/specs/2026-11-08-reca-long-matrix-b-design.md`

## Global Constraints

- Each arm uses exactly 2000 exposures, micro-batch size 1, accumulation 4, and 500 optimizer updates.
- LR is exposure-indexed cosine with peak `2e-5`, 100-exposure warmup, and the four represented LR values averaged per update.
- Target duration stays 5 seconds; prompts, bottleneck, loss, model structure, optimizer family, and precision stay unchanged.
- Pure/T2A/Mix share sample indices and all RNG seeds; Mix is exactly 1000/1000 and every accumulation group is 2/2.
- Qwen2.5-Omni and Audio VAE stay frozen; only DiT is trainable.
- Formal launch uses six currently free GPUs and must keep `/home/linchudeng` below 400 GB.
- Only final checkpoints are retained; disposable smoke checkpoints are removed after verification.

## Review Focus

- A new task mode must not alter sample indices or RNG seeds across schedules; Task 1 tests non-task fields row-by-row.
- Balanced Mix must remain 2/2 in every four-exposure group, including all 500 groups; Task 1 tests every group.
- Existing A/B random schedule behavior must remain the default; Task 1 retains and extends the existing determinism test.
- Formal jobs must not silently start with occupied GPUs or an existing output root; Task 3 adds launcher guard tests.
- A formal checkpoint must encode exposure 2000, optimizer update 500, and the matching schedule digest; Task 4 verifies this first in smoke and Task 5 monitors formal logs.

---

### Task 1: Deterministic task-specific exposure schedules

**Files:**
- Modify: `src/reca_audio/train/gradient_experiment.py:90-118`
- Modify: `tests/test_gradient_experiment.py:69-96`

**Interfaces:**
- Produces: `generate_schedule(total_exposures: int, manifest_size: int, seed: int, reca_probability: float = 0.5, task_mode: str = "random_mix") -> list[dict[str, Any]]`
- Task modes: `random_mix`, `pure`, `t2a`, `balanced_mix`.

- [ ] **Step 1: Write failing schedule-mode tests**

Add tests asserting 2000-row task counts, 2/2 balance in every four-row Mix group, invalid-mode rejection, and exact equality of every non-`task` field across Pure/T2A/Mix schedules generated with the same seed.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/test_gradient_experiment.py -q`
Expected: FAIL because `task_mode` is not accepted.

- [ ] **Step 3: Implement task-mode assignment**

Always consume the existing per-row task RNG draw so all subsequent RNG fields remain identical across modes. For `balanced_mix`, choose the first task of each adjacent pair from the odd row's draw and assign the opposite task to the even row.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_gradient_experiment.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit the schedule change**

```bash
git add src/reca_audio/train/gradient_experiment.py tests/test_gradient_experiment.py
git commit -m "feat: add balanced long-matrix schedules"
```

### Task 2: Expose task mode through the exact-B runner

**Files:**
- Modify: `src/reca_audio/train/run_gradient_ab.py:25-140`
- Modify: `tests/test_gradient_ab_runner.py:13-91`

**Interfaces:**
- Consumes: Task 1 `generate_schedule(..., task_mode=...)`.
- Produces: CLI option `--task-mode {random_mix,pure,t2a,balanced_mix}` and a `task_mode` field in `experiment_contract()`.

- [ ] **Step 1: Write failing runner-contract tests**

Assert the parser accepts all four task modes, rejects invalid modes, includes the selected mode in the contract, and passes it to schedule preparation without changing the default `random_mix` behavior.

- [ ] **Step 2: Run the runner tests and verify failure**

Run: `pytest tests/test_gradient_ab_runner.py -q`
Expected: FAIL because `--task-mode` is unknown.

- [ ] **Step 3: Add the minimal CLI and schedule plumbing**

Add the option, contract field, and `task_mode=args.task_mode` call in `_prepare_schedule`; leave the optimizer loop, AMP, Adafactor, flow loss, and checkpoint payload unchanged.

- [ ] **Step 4: Run runner and helper tests**

Run: `pytest tests/test_gradient_ab_runner.py tests/test_gradient_experiment.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit the runner change**

```bash
git add src/reca_audio/train/run_gradient_ab.py tests/test_gradient_ab_runner.py
git commit -m "feat: parameterize gradient runner task schedule"
```

### Task 3: Add the six-GPU long-matrix launcher

**Files:**
- Create: `runs/scripts/train_reca_long_matrix_b.sh`
- Create: `tests/test_long_matrix_b_launcher.py`

**Interfaces:**
- Consumes: Task 2 runner CLI.
- Produces: `train_reca_long_matrix_b.sh {smoke|run|status}` with environment overrides `OUT_ROOT`, `GPU_PAIRS`, `ALLOW_FORMAL`, `MANIFEST`, and `DIAGNOSTIC_MANIFEST`.

- [ ] **Step 1: Write failing launcher source-contract tests**

Assert the launcher defines `TOTAL=2000`, `WARMUP=100`, peak LR `2e-5`, `--arm b`, `--seconds 5`, and task modes Pure/T2A/balanced Mix; starts all three jobs in parallel; validates six unique free GPUs; refuses an existing output root; and requires `ALLOW_FORMAL=1` for `run`.

- [ ] **Step 2: Run the launcher test and verify failure**

Run: `pytest tests/test_long_matrix_b_launcher.py -q`
Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

For `smoke`, use 4 exposures and one two-GPU pair sequentially across the three arms to minimize GPU use. For `run`, prepare three task-specific schedules, assign one two-GPU pair per arm, launch all arms concurrently, record PIDs, wait for all jobs, and fail if any child exits nonzero. Retain only exposure-final checkpoints.

- [ ] **Step 4: Run launcher and full unit tests**

Run: `pytest tests/test_long_matrix_b_launcher.py tests/test_gradient_ab_runner.py tests/test_gradient_experiment.py -q`
Expected: all tests pass.

- [ ] **Step 5: Run repository checks and commit**

Run: `pytest -q`
Expected: all tests pass.

Run: `git diff --check`
Expected: no output.

```bash
git add runs/scripts/train_reca_long_matrix_b.sh tests/test_long_matrix_b_launcher.py
git commit -m "feat: launch exact-B long training matrix"
```

### Task 4: Deploy and verify on the Linux server

**Files:**
- Modify through deployment: remote copies of the three source/test/script changes
- Create at runtime: `runs/train/reca_long_matrix_b_smoke/`

**Interfaces:**
- Consumes: Tasks 1-3 and the repository deployment procedure.
- Produces: server-side passing tests and three reload-validated four-exposure smoke runs.

- [ ] **Step 1: Deploy through the documented script**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File runs/scripts/deploy.ps1`
Expected: only allowlisted project files are copied and the remote source commit marker updates.

- [ ] **Step 2: Run focused Linux tests**

Run remotely in `audio-omni`: `pytest tests/test_gradient_experiment.py tests/test_gradient_ab_runner.py tests/test_long_matrix_b_launcher.py -q`
Expected: all tests pass.

- [ ] **Step 3: Run the server smoke**

Run remotely with free GPUs: `OUT_ROOT=runs/train/reca_long_matrix_b_smoke GPUS=... bash runs/scripts/train_reca_long_matrix_b.sh smoke`
Expected: Pure/T2A/Mix each finish 4 exposures, one optimizer update, finite loss/gradient metrics, and reload-valid final checkpoint metadata.

- [ ] **Step 4: Verify freeze and checkpoint contracts**

Inspect contracts and checkpoint metadata for accumulation 4, update 1, exposure 4, task modes, and schedule digests; verify logs show the DiT/Qwen/VAE device layout and no OOM or NaN.

- [ ] **Step 5: Remove disposable smoke model files**

Delete only the exact smoke checkpoint paths after successful verification; retain small logs/contracts if useful.

### Task 5: Reclaim storage and launch the formal matrix

**Files:**
- Create at runtime: `runs/train/reca_long_matrix_b_v1/{pure,t2a,mix}/`
- Update after launch: experiment documentation with paths, PIDs, and exact contract

**Interfaces:**
- Consumes: successful Task 4 evidence.
- Produces: three running formal jobs with final outputs projected to keep home usage below 400 GB.

- [ ] **Step 1: Inventory exact disposable checkpoint candidates**

List sizes for smoke checkpoints and already-evaluated A/B checkpoints. Preserve original baseline, existing matrix checkpoints, metrics, evaluation outputs, and logs.

- [ ] **Step 2: Reclaim 15--20 GB safely**

Remove exact disposable model files only after verifying their evaluations and metrics exist. Re-run `du -sh /home/linchudeng` and require projected current usage plus 36 GB of final checkpoints to remain below 400 GB.

- [ ] **Step 3: Recheck six GPUs immediately before launch**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader`
Expected: the selected six GPUs are below the launcher's memory threshold.

- [ ] **Step 4: Start the formal matrix**

Run remotely under `nohup` with a new output root, `ALLOW_FORMAL=1`, and three selected GPU pairs.
Expected: launcher PID and three child PIDs are recorded; each arm creates its contract, schedule digest, and update logs.

- [ ] **Step 5: Perform an early health check**

Verify all six selected GPUs are active, each arm advances beyond its first optimizer update, actual LR matches the four-exposure average, task counts match its mode, and logs contain no OOM, NaN, traceback, or checkpoint collision.

- [ ] **Step 6: Record the launch and estimate**

Document output paths, GPU pairs, PIDs, start time, projected finish time, starting disk usage, and exact experiment contract. Do not claim completion while jobs are still running.
