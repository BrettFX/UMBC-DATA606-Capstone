# Inference benchmark: run 20261006-120925 (baseline)

12th Gen Intel(R) Core(TM) i9-12900H (20 logical cores, 32.7 GB RAM) | GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB | WSL2, python 3.12.4 | torch 2.5.1+cu124, ctranslate2 4.8.2 | commit a489d1e (dirty)

Latency is wall time per utterance at batch size 1 (feature extraction + decoding), excluding model load and two warm-up calls. RTF = processing time / audio time (< 1 is faster than real time). The hybrid CPU schedules threads across performance and efficiency cores, so thread-count rows are approximate.

| config | median s | p95 s | RTF | speedup | WER | load s | warm-up s | peak RAM MB | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| medium/hf-fp16/gpu | 0.29 | 0.55 | 0.10 | 18.3x | 6.3% | 6.6 | 0.6 | 2347 | 1618 |
| medium/hf-fp32/cpu-4t | 5.26 | 6.03 | 1.64 | - | 6.3% | 6.0 | 5.6 | 5138 | - |
| medium/hf-fp32/cpu-8t | 4.85 | 5.82 | 1.54 | 1.1x | 6.3% | 6.5 | 5.1 | 5138 | - |
| medium/hf-int8dyn/cpu-4t | 3.41 | 4.01 | 1.07 | 1.5x | 6.7% | 11.8 | 3.5 | 7668 | - |
| medium/hf-int8dyn/cpu-8t | 3.34 | 3.94 | 1.05 | 1.6x | 6.7% | 11.4 | 3.5 | 7686 | - |
| medium/ct2-int8/cpu-2t | 3.81 | 4.06 | 1.18 | 1.4x | 7.6% | 6.2 | 4.4 | 2242 | - |
| medium/ct2-int8/cpu-4t | 2.54 | 2.95 | 0.81 | 2.1x | 8.0% | 6.1 | 2.7 | 2267 | - |
| medium/ct2-int8/cpu-8t | 2.69 | 2.87 | 0.84 | 2.0x | 7.6% | 5.1 | 2.7 | 2233 | - |
| small/hf-fp32/cpu-4t | 1.67 | 2.00 | 0.54 | - | 6.3% | 5.7 | 1.8 | 2002 | - |
| small/ct2-int8/cpu-2t | 1.31 | 1.47 | 0.41 | - | 8.0% | 5.3 | 1.4 | 1435 | - |
| small/ct2-int8/cpu-4t | 0.96 | 1.04 | 0.30 | - | 8.0% | 5.5 | 1.0 | 1431 | - |
| small/ct2-int8/cpu-8t | 1.03 | 1.19 | 0.32 | - | 7.1% | 5.3 | 1.2 | 1455 | - |
| ner/spacy/cpu | 0.001 | 0.001 | - | - | - | 2.4 | - | 905 | - |
