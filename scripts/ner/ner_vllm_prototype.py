"""Prototype: annotate ATC transcripts with a vLLM-served Qwen3.5 using schema-guided JSON.

Compared with the notebook's HF `generate` + tool-calling loop, this (a) batches every
utterance in one engine call with continuous batching and prefix caching, (b) constrains
decoding to a JSON schema so output can never be malformed, and (c) retries only the
utterances whose entities fail validation, again as one batch.

Runs in a separate environment from the notebooks (vLLM pins its own torch):
    ~/venvs/vllm/bin/python scripts/ner/ner_vllm_prototype.py --mode gold
    ~/venvs/vllm/bin/python scripts/ner/ner_vllm_prototype.py --mode sample
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ner import (  # noqa: E402
    PROMPTS, ExampleRetriever, VllmBackend, annotate_all, entity_spans, load_gold, score_spans, with_spans,
)

NER_DIR = REPO / "data" / "processed" / "ner_dataset"
GOLD_PATH = REPO / "data" / "ner_gold" / "gold.jsonl"
WORK = REPO / "data" / "ner_gold" / "work"  # intermediates: candidates, pre-labels, queues
HF_BASELINE_KEY = "4B 4-bit | + constraints + hints"  # key in ablation_results.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen3.5-4B-quantized.w4a16")
    ap.add_argument("--mode", choices=["gold", "sample", "candidates"], default="gold")
    ap.add_argument("--candidates", default="candidates.jsonl", help="file in data/ner_gold/work/ to annotate in --mode candidates")
    ap.add_argument("--prompt", type=int, choices=sorted(PROMPTS), default=1,
                    help="system-prompt version: 1 original, 2 + decided guidelines, 3 + multi-word waypoint/correction rules")
    ap.add_argument("--postprocess", action="store_true", help="merge adjacent FACILITY spans, expand COMMAND phrases")
    ap.add_argument("--check-missed", action="store_true", help="retry when a known command verb is unlabeled")
    ap.add_argument("--fewshot", type=int, default=0, help="retrieve K similar gold utterances into the prompt")
    ap.add_argument("--no-guided", action="store_true", help="disable JSON-schema constrained decoding")
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--cpu-offload-gb", type=float, default=0, help="weights streamed from CPU RAM (for models too big for the GPU)")
    ap.add_argument("--offload-params", default="", help="comma-separated param-name segments to offload (e.g. mlp)")
    ap.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs (needed with CPU offload on WSL2)")
    ap.add_argument("--out", default=None, help="write predictions JSONL here")
    args = ap.parse_args()

    gold = load_gold(str(GOLD_PATH))
    if args.mode == "gold":
        texts = {u: g["text"] for u, g in gold.items()}
    elif args.mode == "candidates":
        cands = [json.loads(line) for line in open(WORK / args.candidates)]
        texts = {c["utterance_id"]: c["text"] for c in cands}
    else:  # the exact 500 utterances the HF run annotated, so results are comparable
        # the primary (4B) cache is the larger file; the 2B second-pass cache is the other one
        files = sorted(NER_DIR.glob("llm_annotations_*.jsonl"), key=lambda p: -p.stat().st_size)
        recs = [json.loads(line) for line in open(files[0])]
        texts = {r["utterance_id"]: r["text"] for r in recs}
        hf_ref = {r["utterance_id"]: r for r in recs}

    t0 = time.perf_counter()
    backend = VllmBackend(args.model, gpu_util=args.gpu_util, cpu_offload_gb=args.cpu_offload_gb,
                          offload_params=set(filter(None, args.offload_params.split(","))),
                          enforce_eager=args.enforce_eager, max_num_seqs=args.max_num_seqs,
                          guided=not args.no_guided)
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = annotate_all(backend, texts, not args.no_hints, args.max_rounds,
                          check_missed=args.check_missed, postprocess=args.postprocess,
                          system_prompt=PROMPTS[args.prompt],
                          retriever=ExampleRetriever(gold) if args.fewshot else None, n_examples=args.fewshot)
    gen_s = time.perf_counter() - t0
    n = len(texts)
    print(f"\n== {args.mode}: {n} utterances | engine load {load_s:.0f}s | annotate {gen_s:.1f}s "
          f"= {gen_s / n * 1000:.0f} ms/utt ({n / gen_s:.2f} utt/s) ==")
    print(f"valid after retries: {100 * sum(r['ok'] for r in result.values()) / n:.1f}% | "
          f"needed retry: {100 * sum(r['n_rounds'] > 1 for r in result.values()) / n:.1f}% | "
          f"rejected entities/utt: {sum(r['n_rejected'] for r in result.values()) / n:.2f}")

    if args.mode == "candidates":
        path = WORK / f"{Path(args.candidates).stem}_prelabeled.jsonl"
        with open(path, "w") as f:
            for c in cands:
                r = result[c["utterance_id"]]
                f.write(json.dumps({**c, "prelabels": with_spans(c["text"], r["entities"]), "prelabel_ok": r["ok"],
                                    "prelabel_model": args.model}) + "\n")
        print(f"wrote {len(cands)} pre-labeled candidates to {path}")
    elif args.mode == "gold":
        preds = {u: r["entities"] for u, r in result.items()}
        print("\nvLLM span scores vs gold:")
        print(score_spans(gold, preds).round(3).to_string())
        exact = sum(set(entity_spans(g["text"], g["entities"])) == set(entity_spans(g["text"], preds[u]))
                    for u, g in gold.items())
        print(f"\nexact-match utterances: {exact}/{len(gold)}")
        abl = NER_DIR / "ablation_results.json"
        if abl.exists() and HF_BASELINE_KEY in (cache := json.load(open(abl))):
            hf = cache[HF_BASELINE_KEY]
            hf_exact = sum(set(entity_spans(g["text"], g["entities"])) == set(entity_spans(g["text"], hf["preds"][u]))
                           for u, g in gold.items())
            print(f"\nHF bitsandbytes baseline ({HF_BASELINE_KEY}): "
                  f"{score_spans(gold, hf['preds']).loc['ALL'].round(3).to_dict()} | exact {hf_exact}/{len(gold)} | "
                  f"{hf['mean_latency_s'] * 1000:.0f} ms/utt")
    if args.mode == "sample":
        agree = sum(set(entity_spans(texts[u], hf_ref[u]["entities"])) == set(entity_spans(texts[u], r["entities"]))
                    for u, r in result.items())
        print(f"identical to the HF 4B annotations: {agree}/{n} ({100 * agree / n:.1f}%)")
        print(f"HF run: {sum(r['latency_s'] for r in hf_ref.values()) / n * 1000:.0f} ms/utt "
              f"(valid {100 * sum(r['ok'] for r in hf_ref.values()) / n:.1f}%)")

    if args.out:
        with open(args.out, "w") as f:
            for u, r in result.items():
                f.write(json.dumps({"utterance_id": u, "text": texts[u], **r,
                                    "entities": with_spans(texts[u], r["entities"])}) + "\n")


if __name__ == "__main__":
    main()
