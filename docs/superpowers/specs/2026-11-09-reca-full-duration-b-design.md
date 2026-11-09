# ReCA Full-duration B Design

Run a single-variable follow-up to `reca_long_matrix_b_v1`: replace the online random 5-second target crop with a fixed 10-second window. Audio shorter than 10 seconds is right-padded; no source clip exceeds 10 seconds, so the complete captioned content is retained.

Pure, T2A, and balanced Mix all keep the exact B optimizer regime: identical initial checkpoint, manifests, generated schedule seeds, 2000 micro-batch exposures, accumulation 4, 500 updates, exposure-indexed 5% warmup plus cosine LR with peak `2e-5`, Adafactor, fp32 master weights with fp16 AMP, 4x Qwen low-pass, and `omni_seconds=0`.

The existing launcher must continue to default to 5 seconds. A new output root is mandatory. Before formal training, run only the Pure arm for the minimum four exposures needed for one accumulation-4 update. If it OOMs, do not alter LR, accumulation, model structure, or offloading under this experiment name; report the feasibility failure instead.

