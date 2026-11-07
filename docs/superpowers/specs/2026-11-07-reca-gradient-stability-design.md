# ReCA Gradient Stability Controlled Experiment

## Intent

Determine whether the effective batch size and optimizer-update cadence explain
the instability observed in ReCA/T2A mixed training, without changing the model,
loss, prompt, optimizer, maximum learning rate, or training data budget.

## Scope

The experiment has three gated stages:

1. Static gradients from the original Audio-Omni checkpoint.
2. Local unit tests and a server smoke test.
3. A 400-micro-batch A/B run only after explicit user confirmation following
   the smoke test.

No full training run is authorized by this design.

## Controlled Arms

- A: micro-batch 1, accumulation 1, maximum LR `2e-5`.
- B: micro-batch 1, accumulation 4, maximum LR `2e-5`.
- Both consume the exact same pregenerated sequence of 400 sample/task records.
- A performs 400 optimizer updates; B performs 100. Results therefore compare
  optimization regimes at equal data exposure, not equal update counts.

## Exposure-indexed Learning Rate

The shared schedule is indexed by cumulative micro-batch exposure:

- Total exposure: 400 micro-batches.
- Linear warmup: first 20 exposures (5%).
- Peak LR: `2e-5`.
- Cosine decay after warmup.
- A uses the LR at each consumed exposure.
- B uses the arithmetic mean of the four exposure-position LRs represented by
  each optimizer update.
- Every optimizer step records the cumulative exposure and actual LR.

Adafactor must be instantiated with `relative_step=False` and
`scale_parameter=False`; the exposure-indexed external LR is the only schedule.

## Deterministic Input Contract

Pregenerate a JSONL schedule before either arm starts. Each record fixes at
least the manifest row, task (`reca` or `t2a`), and stochastic seeds required by
the conditioner and rectified-flow loss. Both arms consume the same records in
the same order. Diagnostic RNG must be isolated and restored so diagnostics do
not perturb training.

## Gradient Diagnostics

Use a fixed held-out diagnostic batch, with eight samples evaluated in both
ReCA and T2A modes. At exposure 0, compute static statistics from the untouched
base checkpoint. During smoke/training, repeat at fixed exposure milestones.

Record:

- ReCA and T2A loss and gradient norm.
- Cross-task cosine similarity on paired fixed samples and aggregate means.
- Within-task per-sample gradient variance.
- Per-block norm, variance, and cross-task cosine for representative DiT blocks.
- Pre-clip norm, post-clip norm, clipping scale, and whether clipping triggered.
- Global and per-block parameter drift relative to the original DiT.
- Actual optimizer-step LR and cumulative exposure.

Gradient comparisons are computed by streaming parameter dot products and
squared norms. Do not retain multiple full flattened DiT gradients in memory.
Static diagnostics and periodic fixed-batch diagnostics must not call
`optimizer.step()` or mutate model parameters, optimizer state, scaler state,
or the caller's RNG state.

## Artifacts and Reporting

Write machine-readable JSONL metrics plus a compact summary report. The report
must distinguish source-code facts from experimental findings and explicitly
state that equal exposure does not mean equal optimizer updates. It must answer:

- Whether accumulation reduced gradient variance.
- Whether ReCA and T2A gradients conflict.
- Which DiT blocks are least stable.
- Whether parameter drift tracks the observed T2A degradation risk.
- Whether a larger effective batch merits a later full run.

Prompt, loss, model structure, optimizer type, and maximum LR remain unchanged.

