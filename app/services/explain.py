"""Plain-English explanation of a transcript + its entities with the Claude API (bring-your-own-key).

Only transcript text and entity tags are sent, never audio. The key lives in the caller's session state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from app import config
from app.services.errors import AppError

SYSTEM_PROMPT = """You help air traffic controllers read automatic speech-recognition output of radio transmissions.
You receive a transcript (spoken numbers are spelled as words, e.g. 'tree' = 3, 'fife' = 5, 'niner' = 9) and entity
tags from an NER model. Entity labels: {labels}.

Explain the transmission in plain English. Rules:
- Use ONLY the transcript and tags. Never invent callsigns, values, or instructions that are not in the text.
- The transcript may contain recognition errors and the tags may be wrong or incomplete. Say so when something looks
  inconsistent (e.g. a frequency or heading that is implausible, a callsign that looks truncated).
- Point out corrections or retractions ("correction", "disregard", "negative", "I say again"), repeated or redundant
  phrases, and multiple instructions packed into one transmission. Say which instruction is final after a correction.
- Do not give safety judgements, operational advice, or say what anyone should do. Describe what was said.
- Write numbers in conventional form alongside the spoken form (e.g. 'flight level three one zero' -> FL310).
- Keep it concise and scannable."""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One or two sentences: who is talking to whom and what is asked or reported."},
        "instructions": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "entities": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "entities"], "additionalProperties": False},
            "description": "Ordered list of individual instructions/information items in plain English."},
        "corrections_or_repeats": {"type": "array", "items": {"type": "string"}},
        "unclear_or_possible_errors": {"type": "array", "items": {"type": "string"}},
        "verbosity_note": {"type": "string", "description": "Empty string unless the transmission is wordy or redundant."},
    },
    "required": ["summary", "instructions", "corrections_or_repeats", "unclear_or_possible_errors", "verbosity_note"],
    "additionalProperties": False,
}


@dataclass
class Explanation:
    summary: str
    instructions: list[dict] = field(default_factory=list)
    corrections_or_repeats: list[str] = field(default_factory=list)
    unclear_or_possible_errors: list[str] = field(default_factory=list)
    verbosity_note: str = ""


def _client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key, timeout=30.0, max_retries=1)


def _translate(exc: Exception) -> AppError:
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return AppError("The API key was rejected.", "Check that it is a valid Anthropic API key.")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return AppError("The API key does not have permission to use this model.")
    if isinstance(exc, anthropic.RateLimitError):
        return AppError("Rate limit reached for this API key.", "Wait a moment and try again.")
    if isinstance(exc, anthropic.APITimeoutError):
        return AppError("The request to the Claude API timed out.", "Try again.")
    if isinstance(exc, anthropic.APIConnectionError):
        return AppError("Could not reach the Claude API.", "Check your internet connection.")
    if isinstance(exc, anthropic.NotFoundError):
        return AppError("The selected model is not available to this API key.", "Choose another model.")
    if isinstance(exc, anthropic.BadRequestError):
        msg = getattr(exc, "message", str(exc))
        if "credit balance" in msg.lower():
            return AppError("The API account has no credit left.", "Add credit in the Anthropic Console.")
        return AppError("The Claude API rejected the request.", msg[:200])
    if isinstance(exc, anthropic.APIStatusError):
        return AppError(f"The Claude API returned an error ({exc.status_code}).", "Try again later.")
    return AppError("Unexpected error calling the Claude API.", f"{type(exc).__name__}")


def test_key(api_key: str, model: str) -> None:
    """Raises AppError unless a tiny request with `api_key` succeeds."""
    key = (api_key or "").strip()
    if not key:
        raise AppError("Enter an API key first.")
    if not key.startswith("sk-ant-"):
        raise AppError("That does not look like an Anthropic API key.", "Keys start with 'sk-ant-'.")
    try:
        _client(key).messages.create(model=model, max_tokens=16, output_config={"effort": "low"},
                                     messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    except Exception as exc:
        raise _translate(exc) from exc


def cache_key(transcript: str, entities: list[dict], model: str) -> str:
    payload = json.dumps([transcript, [(e["label"], e["text"]) for e in entities], model])
    return hashlib.sha1(payload.encode()).hexdigest()


def explain(api_key: str, model: str, transcript: str, entities: list[dict]) -> Explanation:
    from src.ner.schema import NER_LABELS

    transcript = transcript.strip()
    if not transcript:
        raise AppError("There is no transcript to explain.")
    if len(transcript) > 6000:
        raise AppError("The transcript is too long to explain in one request.", "Analyze a shorter clip.")
    tags = [{"label": e["label"], "text": e["text"]} for e in entities]
    user = f"Transcript:\n{transcript}\n\nEntity tags (JSON):\n{json.dumps(tags)}"
    labels = "; ".join(f"{k} = {v.split(' (e.g.')[0]}" for k, v in NER_LABELS.items())
    try:
        resp = _client(api_key.strip()).messages.create(
            model=model, max_tokens=4000, system=SYSTEM_PROMPT.format(labels=labels),
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user}])
    except Exception as exc:
        raise _translate(exc) from exc
    if resp.stop_reason == "refusal":
        raise AppError("The model declined to explain this transmission.")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        data = json.loads(text)
        return Explanation(
            summary=str(data.get("summary", "")), instructions=list(data.get("instructions", [])),
            corrections_or_repeats=list(data.get("corrections_or_repeats", [])),
            unclear_or_possible_errors=list(data.get("unclear_or_possible_errors", [])),
            verbosity_note=str(data.get("verbosity_note", "")))
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise AppError("The explanation came back in an unexpected format.", "Try again.") from exc
