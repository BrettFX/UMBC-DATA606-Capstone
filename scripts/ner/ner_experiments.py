"""Gold-set experiments: normalization, missed-verb validator and few-shot retrieval.

    ~/venvs/vllm/bin/python scripts/ner/ner_experiments.py [--model ...] [vLLM offload flags]

All variants run in one engine session on the hand-labeled gold utterances. Few-shot
examples are retrieved from the gold set itself with leave-one-out, so no utterance sees
its own label. With 41 gold utterances every difference here is a hypothesis, not a result.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner_vllm_prototype import GOLD_PATH  # noqa: E402  (sets up sys.path)

from ner import PROMPTS, ExampleRetriever, VllmBackend, annotate_all, entity_spans, load_gold, score_spans  # noqa: E402

V2 = dict(system_prompt=PROMPTS[2])
V3 = dict(system_prompt=PROMPTS[3])
VARIANTS = {
    "v1 prompt": dict(),
    "v1 +normalize +missed": dict(postprocess=True, check_missed=True),
    "v2 prompt": dict(**V2),
    "v2 +normalize +missed": dict(**V2, postprocess=True, check_missed=True),
    "v2 +fewshot k=4": dict(**V2, n_examples=4),
    "v2 +normalize +missed +fewshot k=4": dict(**V2, postprocess=True, check_missed=True, n_examples=4),
    "v3 prompt": dict(**V3),
    "v3 +normalize +missed": dict(**V3, postprocess=True, check_missed=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RedHatAI/Qwen3.5-4B-quantized.w4a16")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--cpu-offload-gb", type=float, default=0)
    ap.add_argument("--offload-params", default="")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--only", nargs="+", default=None, help="run only variants whose name matches one of these exactly")
    ap.add_argument("--gold", default=str(GOLD_PATH), help="gold JSONL to score on and retrieve few-shot examples from")
    ap.add_argument("--out", default=None, help="write the results table as JSON")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    texts = {u: g["text"] for u, g in gold.items()}
    retriever = ExampleRetriever(gold)
    backend = VllmBackend(args.model, max_model_len=4096,  # few-shot prompts are longer
                          gpu_util=args.gpu_util, cpu_offload_gb=args.cpu_offload_gb,
                          offload_params=set(filter(None, args.offload_params.split(","))),
                          enforce_eager=args.enforce_eager, max_num_seqs=args.max_num_seqs)

    rows = []
    for name, kw in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.perf_counter()
        res = annotate_all(backend, texts, True, 3, retriever=retriever, **kw)
        secs = time.perf_counter() - t0
        preds = {u: r["entities"] for u, r in res.items()}
        sc = score_spans(gold, preds)
        exact = sum(set(entity_spans(g["text"], g["entities"])) == set(entity_spans(g["text"], preds[u]))
                    for u, g in gold.items())
        row = {"variant": name, "P": sc.loc["ALL", "precision"], "R": sc.loc["ALL", "recall"], "F1": sc.loc["ALL", "f1"],
               "exact": exact, "COMMAND F1": sc.loc["COMMAND", "f1"], "FACILITY F1": sc.loc["FACILITY", "f1"],
               "CALLSIGN F1": sc.loc["CALLSIGN", "f1"], "retry %": 100 * sum(r["n_rounds"] > 1 for r in res.values()) / len(res),
               "s": secs}
        rows.append(row)
        print(f"{name:34s} F1={row['F1']:.3f} P={row['P']:.3f} R={row['R']:.3f} exact={exact}/{len(gold)} "
              f"CMD={row['COMMAND F1']:.2f} FAC={row['FACILITY F1']:.2f} CS={row['CALLSIGN F1']:.2f} "
              f"retry={row['retry %']:.0f}% {secs:.0f}s", flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
