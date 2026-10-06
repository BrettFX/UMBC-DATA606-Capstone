# Inference optimization log (ASR + NER)

The app must run on devices without a GPU, so latency is a design constraint. This log records each speed experiment, what it
measured, and the decision taken. Raw numbers live next to it: `BASELINE.md` (the reference run), `history.csv` (every
configuration of every run), `runs/*.json` (full results with hardware, versions and per-utterance transcripts) and
`experiments/` (the window sweep and the paired accuracy analysis).

Test hardware: Intel i9-12900H laptop (14 cores, AVX2 + AVX-VNNI, no AVX512/AMX) under WSL2, RTX 3070 Ti Laptop GPU. This is a
high-end CPU, so slower devices will be slower in absolute terms; re-run the suite on the target with `--tag baseline-<device>`.
Latency is wall time per utterance at batch size 1, after two warm-up calls. Model: the 3-epoch medium.en LoRA model unless noted.

## 1. Baseline (2026-10-06, run `20261006-120925`, 24 utterances)

| config | median s | RTF | peak RAM | WER* |
|---|---|---|---|---|
| medium, GPU fp16 | 0.29 | 0.10 | 2.3 GB | 6.3% |
| medium, CPU PyTorch fp32, 4 threads | 5.26 | 1.64 | 5.1 GB | 6.3% |
| medium, CPU PyTorch dynamic int8, 4 threads | 3.41 | 1.07 | 7.7 GB | 6.7% |
| medium, CPU CTranslate2 int8, 4 threads | 2.54 | 0.81 | 2.2 GB | 8.0% |
| small, CPU CTranslate2 int8, 4 threads | 0.96 | 0.30 | 1.4 GB | 8.0% |
| NER (spaCy) | 0.001 | - | 0.9 GB | - |

\*24 utterances only, so WER differences of a point or two are noise (see section 4). Threads beyond 4 gave no gain.
The encoder always processes a fixed 30 s window, and on the CPU it accounts for ~2.4 of the 2.5 s of the CTranslate2 run.

## 2. Experiment: shorter encoder window, zero-shot accuracy (GPU, 300 utterances)

Hypothesis: utterances average 3.8 s, so a 30 s window wastes ~90% of the encoder's work; shorten it. The model was fine-tuned
on 30 s windows only, so accuracy had to be checked. Each window is compared with the 30 s window on the same clips (those that fit).

| window | clips that fit | WER at 30 s | WER at window | clips with WER > 50% |
|---|---|---|---|---|
| 20 s | 100% | 4.53% | 4.66% | 4 -> 3 |
| 15 s | 100% | 4.55% | 4.52% | 4 -> 3 |
| **10 s** | **99.7%** | **4.55%** | **4.55%** | **4 -> 3** |
| 8 s | 98% | 4.51% | 8.18% | 3 -> 4 |
| 6 s | 89% | 4.22% | 40.4% | 3 -> 22 |
| 4 s | 60% | 5.60% | 237.9% | 3 -> 67 |

Decision: 10 s is the floor without retraining. Each clip uses the smallest of 10/15/20/30 s that fits it ("dynamic windows",
`DYNAMIC_WINDOWS` in `src/asr/inference.py`). CTranslate2 accepts shorter inputs natively: the medium encoder takes 0.42 s for
10 s of input against 2.40 s for 30 s (4 threads).

## 3. Experiment: dynamic windows on CPU (24 utterances, same clips as the baseline)

| config (4 threads) | before | after | speedup |
|---|---|---|---|
| medium, CTranslate2 int8 | 2.54 s | **0.71 s** | 3.6x |
| medium, PyTorch dynamic int8 | 3.41 s | 1.28 s | 2.7x |
| medium, PyTorch fp32 | 5.26 s | 2.08 s | 2.5x |
| small, CTranslate2 int8 | 0.96 s | **0.26 s** | 3.7x |
| small, PyTorch fp32 | 1.67 s | 0.71 s | 2.3x |

Two threads: medium CTranslate2 int8 with dynamic windows still takes 0.97 s. GPU latency is unchanged (0.25 s; it is not encoder-bound).

## 4. Experiment: accuracy of the fast configurations at scale (300 utterances, paired)

Every configuration against the GPU fp16 model on the same 150 real + 150 simulated utterances; confidence intervals resample
utterances (`experiments/accuracy_20261006-130119.md` has the real/simulated breakdown).

| config | WER | vs GPU reference (95% CI) | verdict |
|---|---|---|---|
| medium, GPU fp16 (reference) | 4.53% | - | - |
| medium, GPU fp16, dynamic windows | 4.53% | +0.00 (-0.50, +0.50) | no clear difference |
| medium, CTranslate2 int8 | 4.50% | -0.03 (-0.42, +0.32) | no clear difference |
| **medium, CTranslate2 int8, dynamic windows** | **4.28%** | **-0.25 (-0.70, +0.22)** | **no clear difference** |
| medium, PyTorch dynamic int8, dynamic windows | 6.78% | +2.25 (-0.13, +6.75) | gross failures on real audio (+4.96, CI +0.13 to +14.7) |
| small, PyTorch fp32 | 5.89% | +1.36 (+0.51, +2.26) | worse |
| small, CTranslate2 int8, dynamic windows | 6.69% | +2.15 (+1.24, +3.12) | worse (real audio 12.0% vs 8.1%) |

Findings: the baseline's 8.0% vs 6.3% for int8 was noise. Neither CTranslate2 int8 nor dynamic windows measurably costs accuracy for
the medium model (differences larger than about 0.7 points are ruled out; "no clear difference" is not proof of equality).
PyTorch dynamic int8 is both less accurate and needs 3x the memory. The small model is about 2 points worse (about 4 on real audio).

## 5. Current recommendation

- **CPU:** medium model, CTranslate2 int8, dynamic windows, 4 threads: **~0.7 s per utterance (RTF ~0.2), ~2.2 GB RAM**, accuracy
  equal to the GPU model. `best_transcriber()` in `src/asr/inference.py` selects this automatically. Roughly 7x faster than the
  PyTorch fp32 baseline (5.3 s) and 3.7x faster than CTranslate2 without windows.
- **GPU:** PyTorch fp16, ~0.25 s.
- **Optional fast mode:** the small model with CTranslate2 int8 and dynamic windows, ~0.26 s, at a cost of about 2 points of WER.
- **Avoid:** PyTorch dynamic int8 quantization.
- **NER:** 1 ms; irrelevant to latency.

## 6. Next ideas (not yet tried)

1. Fine-tune with shorter windows (6-8 s) so the floor drops below 10 s; the encoder would shrink a further ~30-40%.
2. Ship the CTranslate2 int8 model as an S3 artifact so CPU devices download 0.77 GB instead of 1.5 GB and need neither torch nor transformers.
3. Measure cold start (model load 5-6 s) and keep the model in a long-lived process in the app.
4. Benchmark on the real target devices, including 2-thread configurations, with `--tag baseline-<device>`.
5. Streaming or chunked transcription for utterances longer than 30 s.

## Reproducing

```bash
python scripts/benchmark_inference.py --suite full --tag baseline            # all configurations (45-60 min)
python scripts/benchmark_inference.py --suite windows --tag short-window    # dynamic-window configurations
python scripts/benchmark_inference.py --n 300 --tag accuracy-300 --configs medium/hf-fp16/gpu medium/ct2-int8-dynwin/cpu-4t
python scripts/analyze_benchmark_accuracy.py --tag accuracy-300             # paired WER analysis
python scripts/experiment_short_window.py --model medium --n 300            # the window sweep (GPU)
```
Do not run other CPU-heavy work during a benchmark: it distorts the timings.
