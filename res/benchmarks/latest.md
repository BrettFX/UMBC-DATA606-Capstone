# Inference benchmark: run 20261006-130119 (accuracy-300)

12th Gen Intel(R) Core(TM) i9-12900H (20 logical cores, 32.7 GB RAM) | GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB | WSL2, python 3.12.4 | torch 2.5.1+cu124, ctranslate2 4.8.2 | commit 7b0dd21 (dirty)

Latency is wall time per utterance at batch size 1 (feature extraction + decoding), excluding model load and two warm-up calls. RTF = processing time / audio time (< 1 is faster than real time). The hybrid CPU schedules threads across performance and efficiency cores, so thread-count rows are approximate.

| config | median s | p95 s | RTF | speedup | WER | load s | warm-up s | peak RAM MB | GPU MB | vs baseline (median, WER) |
|---|---|---|---|---|---|---|---|---|---|---|
| medium/hf-fp16/gpu | 0.33 | 0.61 | 0.09 | - | 4.5% | 1.6 | 0.5 | 2412 | 1619 | +13% latency, -1.8 pts WER |
| medium/hf-fp16-dynwin/gpu | 0.26 | 0.55 | 0.08 | - | 4.5% | 1.7 | 0.5 | 2413 | 1574 | -9% latency, -1.8 pts WER (vs medium/hf-fp16/gpu) |
| medium/ct2-int8/cpu-4t | 2.72 | 3.16 | 0.73 | - | 4.5% | 0.8 | 2.7 | 2314 | - | +7% latency, -3.5 pts WER |
| medium/ct2-int8-dynwin/cpu-4t | 0.74 | 1.05 | 0.20 | - | 4.3% | 0.7 | 0.7 | 2286 | - | -71% latency, -3.7 pts WER (vs medium/ct2-int8/cpu-4t) |
| medium/hf-int8dyn-dynwin/cpu-4t | 1.24 | 1.81 | 0.34 | - | 6.8% | 6.6 | 1.3 | 7749 | - | -64% latency, +0.1 pts WER (vs medium/hf-int8dyn/cpu-4t) |
| small/hf-fp32/cpu-4t | 1.64 | 2.11 | 0.45 | - | 5.9% | 0.5 | 1.8 | 2085 | - | -1% latency, -0.4 pts WER |
| small/hf-fp32-dynwin/cpu-4t | 0.72 | 1.16 | 0.20 | - | 6.3% | 0.5 | 0.7 | 1953 | - | -57% latency, +0.0 pts WER (vs small/hf-fp32/cpu-4t) |
| small/ct2-int8-dynwin/cpu-4t | 0.25 | 0.36 | 0.07 | - | 6.7% | 0.3 | 0.3 | 1419 | - | -74% latency, -1.3 pts WER (vs small/ct2-int8/cpu-4t) |
