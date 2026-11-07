# ReCA Gradient A/B: Small T2A Evaluation Design

## Intent

Determine whether the lower-clipping, lower-drift accumulation-4 checkpoint from
the 400-micro-batch controlled experiment preserves text-conditioned generation
better than accumulation-1 at the same data exposure. This is a diagnostic T2A
comparison, not a new training run and not a full AudioCaps benchmark claim.

## Hypothesis

At equal 400-micro-batch exposure, arm B (`gradient_accumulation=4`) changed the
DiT less than arm A and clipped fewer optimizer updates. If that reduced
catastrophic interference with the original text-conditioning path, B should
retain Base-like text/audio alignment better than A. Reconstruction or flow-loss
results alone cannot establish this; generated audio must be evaluated.

## Controlled Models

Evaluate exactly three model states:

1. **Base**: `Audio-Omni/model/model.ckpt`.
2. **A**: Base with
   `runs/train/reca_gradient_ab_v1/a/dit_exposure0400.pt` merged into the DiT.
3. **B**: Base with
   `runs/train/reca_gradient_ab_v1/b/dit_exposure0400.pt` merged into the DiT.

A and B share the same initial checkpoint, 400-record task/sample schedule,
optimizer, maximum LR, and data exposure. They do not share the same optimizer
update count: A made 400 updates and B made 100. The comparison therefore tests
optimization regimes at equal data budget.

## Held-out Evaluation Set

Use only the SHA-disjoint AudioCaps test partition under
`runs/Benchmark/_data/audiocaps/reca_split_v1`. The recorded split contains 445
caption rows from 89 unique audio hashes and has zero train/test ID or SHA
intersection.

Build a deterministic diagnostic subset with one caption/reference pair per
unique audio SHA:

- preserve the first occurrence in test-manifest order;
- write exactly 89 prompts;
- link exactly one matching reference audio per prompt;
- record source manifest hash, selected IDs/paths/audio hashes, and the derived
  prompt manifest hash;
- fail on a duplicate output ID, missing audio file, or any count other than 89.

The derived subset is a runtime benchmark artifact, not training input. Its
prompt manifest and selection metadata are shared unchanged by Base, A, and B.

## Generation Controls

All three states use the same:

- prompt order and shard assignment;
- `seed=1234` and the existing per-shard deterministic seed rule;
- 100 ODE steps;
- CFG 6.0;
- 10-second output duration;
- fp16 inference;
- Audio-Omni model config and Qwen checkpoint;
- three two-GPU processes on physical GPUs 1–6, rechecking occupancy before
  each model.

Run models sequentially. A failed shard fails that model evaluation; partial
manifests are not scored as if complete. Require 89 successful outputs before
scoring.

## Merge and Storage Safety

Base is evaluated directly. A and B are strictly merged one at a time using
`python -m reca_audio.train.merge_ckpt` with its default strict key/shape
validation.

Each merged full checkpoint is written beneath a uniquely named directory in
`/tmp`, evaluated, and deleted immediately afterward. A shell cleanup trap must
remove that exact temporary directory on normal exit or failure. Never keep A
and B full merged checkpoints simultaneously, and never write them beneath the
user home directory. Peak additional checkpoint storage is therefore about
21 GB; the 12 GB DiT checkpoints remain the canonical retained artifacts.

Generated WAVs, manifests, logs, selection metadata, and metric JSON are kept
under one named directory in `runs/Benchmark/output`. Before and after the run,
record filesystem usage and user-home usage. Do not delete the two formal A/B
DiT checkpoints.

## Metrics and Interpretation

For each complete 89-item output, compute the same available AudioCaps metrics:

- CLAP text/audio score (primary for the T2A preservation question);
- CLAP-FAD;
- PANNs KL, IS, and FD when the installed scorer succeeds.

Also report generation failures and paired per-prompt CLAP deltas when the
scoring interface permits extraction. Because 89 unique clips are a small
diagnostic set, distributional metrics are high variance. A result supports
promotion to a larger held-out evaluation only if B consistently improves over
A without a material degradation from Base. It does not prove general T2A
improvement, and it does not attribute changes solely to gradient variance.

## Execution Gate and Expected Repository Changes

Before execution, write an implementation plan and obtain explicit approval.
The plan may add only the minimal deterministic subset/evaluation launcher and
focused tests needed for reproducibility; it must not change model, training,
loss, prompt semantics, optimizer, or checkpoint contents. Deployment follows
the repository README and `docs/deploy.md`. No training starts as part of this
evaluation.
