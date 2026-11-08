# ReCA Long Matrix with Exposure-Aligned Accumulation Design

**Date:** 2026-11-08  
**Status:** Approved  
**Scope:** Pure ReCA, pure T2A, and 1:1 Mix training under the optimizer regime validated by gradient experiment arm B

## Objective

Run a controlled 2000-micro-batch training matrix that tests whether gradient accumulation preserves T2A behavior while retaining reconstruction alignment. The experiment changes the optimization regime relative to the existing `reca_matrix_v1`, while holding the data path, target duration, prompts, model initialization, optimizer family, precision, and total data exposure fixed.

The primary comparison is:

- Existing matrix: 2000 exposures, accumulation 1, 2000 optimizer updates.
- New matrix: 2000 exposures, accumulation 4, 500 optimizer updates.

Equal exposure does not imply equal optimizer-update count. Results must be interpreted as a comparison of optimization regimes under the same data budget.

## Duration Decision

The low-level `audio_input_prompt` remains a fixed 10-second silent waveform because the pretrained mel conditioner expects 1723 input frames. Generated audio is approximately 10.96 seconds because diffusion uses a fixed latent/sample size.

This mismatch is not treated as a missing final second of frame-aligned conditioning: the silent mel representation is a constant global condition and is not aligned one-to-one with generated latent frames. It therefore does not provide source-audio detail and is unlikely to be a severe confound in the controlled comparison.

Target audio remains 5 seconds for this matrix. Changing target duration to 11 seconds at the same time as accumulation would violate the single-variable principle and break direct comparability with `reca_matrix_v1`. Duration alignment can be tested later as a separate ablation.

## Experiment Arms

All arms start from the same original Audio-Omni checkpoint.

| Arm | Per-exposure task | Expected task counts |
| --- | --- | --- |
| Pure | ReCA only | 2000 ReCA / 0 T2A |
| T2A | T2A only | 0 ReCA / 2000 T2A |
| Mix | deterministic balanced sequence | 1000 ReCA / 1000 T2A |

The three schedules use the same sample indices and RNG seeds at every exposure. Only the task label differs. Mix must be exactly balanced rather than merely balanced in expectation.

## Shared Training Contract

- Total exposure: 2000 micro-batches per arm.
- Micro-batch size: 1.
- Gradient accumulation: 4.
- Optimizer updates: 500 per arm.
- Optimizer: Adafactor.
- Adafactor internal schedule: `relative_step=False`, `scale_parameter=False`, `warmup_init=False`.
- Peak learning rate: `2e-5`.
- Warmup: 100 exposure positions (5% of 2000).
- Decay: cosine, indexed by exposure position.
- Update learning rate: arithmetic mean of the four exposure-position LRs represented by that optimizer update.
- Gradient clipping: global norm 1.0.
- Precision: existing AMP configuration.
- Target duration: 5 seconds at 44.1 kHz.
- Qwen bottleneck: existing `omni_downsample=4` behavior.
- Qwen2.5-Omni: frozen.
- Audio VAE: frozen and kept on CPU as in the existing controlled runs.
- DiT: trainable.
- Prompt templates, conditioning construction, Rectified Flow loss, and model structure: unchanged.
- Seed: one shared fixed seed across all arms.

## Schedule and Reproducibility

A canonical exposure schedule records, for each exposure:

- one-based exposure index;
- manifest sample index;
- data RNG seed;
- flow RNG seed;
- prompt RNG seed.

Task labels are derived deterministically for each arm. Pure and T2A override every task label. Mix assigns exactly one ReCA and one T2A exposure in each adjacent pair, with the order determined reproducibly from the shared seed. This prevents accidental imbalance and keeps each accumulation group balanced at two ReCA and two T2A samples.

The schedule and its SHA256 digest are stored with each run. Checkpoints store the digest and training contract so that resume cannot silently use a different sequence.

## Minimal Implementation

Reuse the exposure-indexed LR and accumulation logic already exercised by `run_gradient_ab.py`. Add a long-matrix mode or a narrowly scoped runner option that supports `pure`, `t2a`, and exactly balanced `mix` task schedules. Do not approximate arm B through the optimizer-step scheduler in the normal `train.py` path.

Required validation before formal launch:

1. Unit tests verify 2000 exposures, accumulation 4, 500 updates, 100-exposure warmup, and averaged update LR.
2. Unit tests verify identical sample/RNG fields across all three arms and exact task counts.
3. Unit tests verify Mix has two ReCA and two T2A samples in every four-exposure accumulation group.
4. Server smoke uses a small exposure count and confirms checkpoint save/reload, frozen Qwen/VAE, trainable DiT, and finite losses.
5. Formal launch starts only after the smoke artifacts are removed or marked disposable.

## Logging and Checkpoints

Each optimizer update records:

- start and end exposure;
- optimizer update index;
- actual LR;
- ReCA and/or T2A loss;
- pre-clip gradient norm;
- clipping indicator or clip scale;
- elapsed time;
- cumulative task counts.

Save the final checkpoint at exposure 2000. Intermediate model checkpoints are not retained by default because each DiT checkpoint is approximately 12 GB and the user home is already approximately 373 GB. Lightweight JSONL metrics, schedule, contract, arguments, PID, and launcher logs are retained.

The three final checkpoints add approximately 36 GB. To keep `/home/linchudeng` under the requested 400 GB ceiling, at least approximately 10 GB of disposable prior artifacts must be removed or moved before formal saving; a safer target is 15--20 GB of reclaimed space. Any deletion requires an explicit inventory of exact candidates and must preserve baselines and evaluation evidence.

## Execution Layout

Use six V100 GPUs as three independent two-device jobs:

- one GPU for DiT and one GPU for Qwen conditioning per arm;
- VAE remains on CPU;
- exact GPU assignments are selected immediately before launch after checking current occupancy.

Based on the 400-exposure B run and the previous 2000-exposure matrix, estimated wall time with all three arms running concurrently is approximately 3.5--4 hours, with a 4--5 hour operational allowance for initialization, checkpoint serialization, and variance.

## Evaluation and Interpretation

After training, evaluate all three arms and the unchanged original baseline with identical benchmark manifests and generation parameters. At minimum:

- reconstruction sanity check for ReCA-capable arms;
- AudioCaps T2A metrics using the hardened evaluator;
- available T2S benchmark if the current project benchmark supports it consistently.

Loss reduction alone is not evidence that T2A improved. The main questions are:

1. Does accumulated Mix preserve more T2A quality than the earlier accumulation-1 Mix?
2. Does Pure retain reconstruction while continuing to damage text-conditioned generation?
3. Does T2A provide an upper control for preserving text-conditioned generation under the same optimizer regime?
4. Are differences explained by task objective rather than unequal data exposure or schedule randomness?

## Affected Paths

- Training: yes; adds strict exposure-indexed Pure/T2A/Mix orchestration.
- Cache: no conditioning-cache format change; existing caches remain valid if used.
- Inference: no source or parameter change.
- Benchmark: no evaluator change; only new checkpoints and output directories.
- Retraining: required; all three arms start from the original checkpoint.

## Non-Goals

- No 11-second target-duration change.
- No lower-LR ablation in this matrix.
- No prompt, hidden-layer, token-selection, pooling, compression, model-structure, optimizer-family, or loss change.
- No claim that reconstruction loss predicts T2A quality.
