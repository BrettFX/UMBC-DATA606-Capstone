"""LLM annotation engine: batched, schema-guided JSON output with validate-and-retry.

Every utterance is rendered into one chat prompt and sent to the backend in a single batch (the
engine's continuous batching and prefix caching do the speed-up). Output is constrained to a JSON
schema so it is always well-formed; entities are then checked against the label constraints and
aligned to the transcript, and only utterances with rejected entities are re-asked, again as one
batch, with the errors fed back. The backend is a small interface so this logic runs (and is
tested) without a GPU; `VllmBackend` is the real implementation and imports vLLM lazily.
"""

from __future__ import annotations

import json
import os
from typing import Protocol

from .postprocess import ExampleRetriever, format_entities, missed_command_feedback, normalize_entities
from .prompt import PROMPTS, build_user_message
from .schema import NER_LABELS, EntitySpan, align_entities, check_constraints

JSON_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "label": {"enum": list(NER_LABELS)}},
        "required": ["text", "label"], "additionalProperties": False}}},
    "required": ["entities"], "additionalProperties": False,
}
FORMAT_SUFFIX = ('\n\nReturn ONLY a JSON object {"entities": [{"text": ..., "label": ...}, ...]} listing '
                 'every entity in order of appearance. Use {"entities": []} if there are none.')


class Backend(Protocol):
    """What the annotation loop needs from an LLM server."""

    def render(self, messages: list[dict]) -> str: ...

    def generate(self, prompts: list[str]) -> list[str]: ...


class VllmBackend:
    """vLLM engine with JSON-schema constrained greedy decoding. Needs the vLLM environment."""

    def __init__(self, model: str, *, gpu_util: float = 0.85, cpu_offload_gb: float = 0,
                 offload_params: set[str] | None = None, enforce_eager: bool = False,
                 max_num_seqs: int = 32, max_model_len: int = 2048, guided: bool = True,
                 max_new_tokens: int = 256):
        # The FlashInfer sampler JIT-compiles with ninja/nvcc, which the annotation env doesn't have.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self._llm = LLM(
            model, max_model_len=max_model_len, gpu_memory_utilization=gpu_util,
            limit_mm_per_prompt={"image": 0, "video": 0},  # skip vision-tower memory profiling
            max_num_seqs=max_num_seqs, max_num_batched_tokens=2048, enable_prefix_caching=True,
            cpu_offload_gb=cpu_offload_gb, enforce_eager=enforce_eager,
            cpu_offload_params=offload_params or set())
        self._tok = self._llm.get_tokenizer()
        self._params = SamplingParams(
            temperature=0, max_tokens=max_new_tokens,
            structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA) if guided else None)

    def render(self, messages: list[dict]) -> str:
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def generate(self, prompts: list[str]) -> list[str]:
        return [o.outputs[0].text for o in self._llm.generate(prompts, self._params, use_tqdm=False)]


def validate(text: str, raw: str) -> tuple[list[EntitySpan], list[str]]:
    """Parse model JSON into accepted entities plus fix-it error messages for rejected ones."""
    try:
        items = json.loads(raw)["entities"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return [], [f"output was not valid JSON ({exc})"]
    accepted, errors = [], []
    for item in items:
        try:
            ent = EntitySpan(**item)
            check_constraints(ent, text)
            _, rejected = align_entities(text, accepted + [ent])
            if rejected:
                raise ValueError(rejected[-1][1])
            accepted.append(ent)
        except (ValueError, TypeError) as exc:
            errors.append(f"{item.get('label')} '{item.get('text')}': {str(exc).splitlines()[0]}")
    return accepted, errors


def annotate_all(backend: Backend, texts: dict[str, str], use_hints: bool = True, max_rounds: int = 3, *,
                 check_missed: bool = False, postprocess: bool = False,
                 retriever: ExampleRetriever | None = None, n_examples: int = 0,
                 system_prompt: str | None = None) -> dict[str, dict]:
    """Annotate `{utterance_id: text}` and return `{utterance_id: record}`.

    Each record has `entities` (dicts with `text`/`label`), `ok` (no hard validation error left),
    `n_rounds`, `n_rejected` (rejected entities summed over rounds) and `n_flags` (soft missed-verb
    flags still present after the last round). Missed-verb flags trigger a retry but never make a
    record `ok=False`.
    """
    def user_message(u: str, t: str) -> str:
        msg = build_user_message(t, use_hints)
        if retriever and n_examples:  # leave-one-out: an utterance never sees its own gold label
            shots = retriever.nearest(t, n_examples, exclude=u)
            msg = ("Correctly annotated transcripts similar to this one:\n" + "\n".join(
                f"Transcript: {x['text']}\nEntities: {format_entities(x['text'], x['entities'])}" for x in shots)
                + "\n\nNow annotate this one.\n" + msg)
        return msg + FORMAT_SUFFIX

    system = system_prompt or PROMPTS[1]
    convo = {u: [{"role": "system", "content": system}, {"role": "user", "content": user_message(u, t)}]
             for u, t in texts.items()}
    result = {u: {"entities": [], "ok": False, "n_rounds": 0, "n_rejected": 0, "n_flags": 0} for u in texts}
    pending = list(texts)
    for round_ in range(1, max_rounds + 1):
        outputs = backend.generate([backend.render(convo[u]) for u in pending])
        still = []
        for u, raw in zip(pending, outputs):
            accepted, hard = validate(texts[u], raw)
            soft = missed_command_feedback(texts[u], [e.model_dump() for e in accepted]) if check_missed else []
            r = result[u]
            r.update(entities=[e.model_dump() for e in accepted], ok=not hard, n_rounds=round_, n_flags=len(soft))
            r["n_rejected"] += len(hard)
            if (hard or soft) and round_ < max_rounds:
                convo[u] += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": "Some entities were rejected:\n- " + "\n- ".join(hard + soft)
                              + "\nReturn the complete corrected JSON object."}]
                still.append(u)
        pending = still
        if not pending:
            break
    if postprocess:
        for u, r in result.items():
            r["entities"] = normalize_entities(texts[u], r["entities"])
    return result
