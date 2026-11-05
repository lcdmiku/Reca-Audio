# ReCA Mixed Training Design

## Intent

Run the first controlled ReCA experiment that can distinguish reconstruction
alignment from text-to-audio retention.  The experiment must use the frozen
Qwen2.5-Omni and Audio VAE pathways already present in Audio-Omni, train only
the DiT, and compare all trained arms from the same original checkpoint.

## Goals

- Guarantee exactly one Qwen audio placeholder for every ReCA sample.
- Replace the contaminated all-4411 AudioCaps protocol with a deterministic,
  content-hash-grouped internal train/validation/test split.
- Run Pure ReCA, T2A-only, and ReCA+T2A 1:1 from the same base checkpoint.
- Record realized task counts and make checkpoints safe for normal resume.
- Start the three formal training arms concurrently on disjoint GPU pairs.

## Non-goals

- Do not change Qwen hidden layer, token selection, pooling, compression, DiT
  architecture, Rectified Flow mathematics, or the 4x low-pass bottleneck.
- Do not claim official AudioCaps generalization from the internal held-out
  split.  Official AudioCaps train/test evaluation is a second-stage run.
- Do not add a lambda sweep before the 1:1 mixture is evaluated.
- Do not edit server source files directly.

## Constraints

- Local repository is the only source of code; deployment must use the hash-
  checked process in `docs/deploy.md`.
- Qwen2.5-Omni, its projection, the auxiliary conditioners, and Audio VAE stay
  frozen; only the full DiT is optimized.
- Each trained arm uses 2000 optimizer steps, batch size 1, accumulation 1,
  Adafactor, lr 2e-5, warmup 100, fp32 master weights with fp16 autocast,
  5-second targets, and the 16k -> 4k -> 16k ReCA bottleneck.
- GPU pairs are isolated through `CUDA_VISIBLE_DEVICES`: Pure 0,1; T2A 2,3;
  Mix 4,5.  GPUs 6,7 remain available for base evaluation or recovery.

## Approaches Considered

### Continue training on all 4411 clips

Fastest, but the training targets and AudioCaps evaluation reference directory
are identical.  Rejected because it cannot produce a credible held-out result.

### Deterministic internal held-out split (selected)

Hash every waveform, keep identical audio in one group, then deterministically
assign approximately 80/10/10 percent to train/validation/test.  This supports
an immediate controlled mechanism experiment with existing server data.  It is
not an official AudioCaps test and will be labelled accordingly.

### Download official AudioCaps train audio before any experiment

Most rigorous, but blocks the immediate experiment on a large data transfer.
Retained as the required second-stage external validation after the mechanism
experiment identifies a viable objective.

## Prompt Contract

All new ReCA arms use the current high-level semantic reconstruction system
prompt, not the historical b6 acoustic-detail prompt.  Pure ReCA is rerun in
the same matrix, so objective mode remains the only difference among the new
arms.  A shared helper strips accidental leading `<|AUDIO|>` markers from the
instruction and lets the prompt wrapper insert exactly one marker.

Any historical b6 result remains labelled as a separate protocol.

## Dataset Contract

The split tool consumes the existing JSONL manifest.  It computes SHA256 for
each audio file, groups rows by SHA256, and assigns whole groups using a stable
seeded rank.  It writes versioned outputs without overwriting an existing
directory:

- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `test_manifest.jsonl`
- `val_prompts.jsonl`
- `test_prompts.jsonl`
- `val_ref/` and `test_ref/` symlinks
- `split_report.json`

The report records row counts, unique IDs, unique hashes, intersections, seed,
source manifest hash, and per-split manifest hashes.

## Experiment Matrix

| Arm | Mode | Expected 2000-step exposure | Hypothesis |
|---|---|---:|---|
| Base | no update | 0 | Reference T2A and reconstruction behavior |
| Pure | `reca` | 2000 ReCA | Reconstruction alignment alone may interfere with text conditioning |
| T2A-only | `t2a` | 2000 T2A | Controls for ordinary caption supervision |
| Mix-1:1 | `mix`, lambda 1:1 | about 1000 + 1000 | T2A replay may retain text ability while preserving alignment |

All arms use the same train manifest and target construction.  Under fixed
compute, Mix receives half the ReCA exposure of Pure; an exposure-matched run
is deferred unless this becomes the limiting interpretation.

## Evaluation Contract

- Reconstruction uses held-out clips with matched, shuffled, and muted Qwen
  audio conditions at CFG 1.0.
- T2A uses `test_prompts.jsonl` and `test_ref/`, with identical generation
  steps, CFG, and per-item seeds for every arm.
- The internal test is reported separately from the historical AudioCaps
  1000/4411 protocol.
- Training loss alone is not evidence that ReCA works.

## Deployment and Launch

The implementation is committed on a `codex/` branch, deployed through the
repository scripts, smoke-tested in each task mode, and only then launched via
a versioned matrix script.  Each arm writes a PID, launcher log, `args.json`,
training log, and checkpoints to a distinct output directory.  The launcher
must refuse to overwrite a non-empty formal output directory.
