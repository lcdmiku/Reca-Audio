# Audio Editing Transfer Implementation Plan

Use the independent evaluation protocol in `runs/Benchmark/audio_edit_transfer_protocol.md` to compare existing checkpoints.

- [ ] Build and freeze the held-out source, reference and intervention manifest in `prepare_transfer.py`. Verify source provenance and intersections before generation.
- [ ] Use the native edit conditioning and 100-step sampler in `evaluate_transfer.py`. Confirm source and text interventions change the intended conditioning tensors independently on Base and smoke-test every checkpoint.
- [ ] Score the complete paired outputs with `score_transfer.py`. Report edit success, source preservation, simultaneous success and their continuous diagnostics. Keep distributional quality and free-form style proxies separate.
- [ ] Verify and archive outcomes from the ordered six-model run. Interpret the matched-caption comparison and the two interventions independently.
