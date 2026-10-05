"""Prompt, tool schema and output parsing for LLM annotation."""

import re

from .rules import rule_candidates
from .schema import NER_LABELS, EntitySpan

SYSTEM_PROMPT = (
    "You annotate air traffic control (ATC) radio transcripts for named-entity recognition. "
    "The transcript is lowercase with numbers spelled out as words. For every entity, call the "
    "add_entity tool once, copying the entity text EXACTLY and contiguously from the transcript. "
    "Use the shortest correct span. Do not paraphrase, do not merge separate entities, and do not "
    "label greetings, farewells or filler (good afternoon, good day, bye, tschuss, thank you, roger, ah). "
    "If there are no entities, make no tool calls.\n\n"
    "Labels:\n" + "\n".join(f"- {k}: {v}" for k, v in NER_LABELS.items()) + "\n\n"
    "Examples:\n"
    "Transcript: lufthansa four one alpha descend flight level one two zero\n"
    "Entities: CALLSIGN 'lufthansa four one alpha'; COMMAND 'descend'; ALTITUDE 'flight level one two zero'\n"
    "Transcript: praha good evening csa five mike bravo contact zurich one three three decimal four\n"
    "Entities: FACILITY 'praha'; CALLSIGN 'csa five mike bravo'; COMMAND 'contact'; FACILITY 'zurich'; FREQUENCY 'one three three decimal four'\n"
    "Transcript: ah stand by\n"
    "Entities: COMMAND 'stand by'\n"
    "Transcript: good evening speedbird two seven heading two seven zero squawk four two one zero runway two four left\n"
    "Entities: CALLSIGN 'speedbird two seven'; HEADING 'heading two seven zero'; SQUAWK 'squawk four two one zero'; RUNWAY 'runway two four left'"
)

# Guideline decisions made after the first double-annotation round (see data/ner_gold/README.md).
GUIDELINES_V2 = (
    "\n\nAdditional rules:\n"
    "- Label every instruction or readback verb as COMMAND, including generic ones: continue, fly, line up, "
    "lining up, set course, confirm, wait, report, maintain, identified, climbing, descending.\n"
    "- 'direct' is a COMMAND; 'proceed direct' is ONE COMMAND span. The place after it is a WAYPOINT.\n"
    "- A clearance is one COMMAND span of any length: 'cleared to land', 'cleared for takeoff', 'cleared for ils approach'.\n"
    "- A partial callsign counts as CALLSIGN (a lone airline name such as 'csa').\n"
    "- A frequency spoken only as digits (e.g. 'one three four five two') is a FREQUENCY; 'point' is a spoken decimal.\n"
    "- 'radar contact' is not an entity: do not label 'contact' or 'radar' there.\n"
    "- A multi-word facility ('rhein radar') is ONE FACILITY span.\n"
    "- 'heading of zero two zero' is one HEADING span including 'of'."
)
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT + GUIDELINES_V2

# Added after the held-out review. 'request' as COMMAND was deliberately NOT adopted: the original gold
# leaves 'request'/'requesting' unlabeled, so it contradicts existing labels.
GUIDELINES_V3 = (
    "\n- A multi-word place name is ONE WAYPOINT span ('st prex'); different waypoint names listed one after "
    "another in a route are separate spans ('trasadingen karlsruhe' is two).\n"
    "- A callsign repeated or corrected after the word 'correction' is a CALLSIGN."
)
SYSTEM_PROMPT_V3 = SYSTEM_PROMPT_V2 + GUIDELINES_V3
PROMPTS = {1: SYSTEM_PROMPT, 2: SYSTEM_PROMPT_V2, 3: SYSTEM_PROMPT_V3}

_ENTITY_SCHEMA = EntitySpan.model_json_schema()
_ENTITY_SCHEMA.pop("title", None)
NER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "add_entity",
        "description": "Record one entity span, copied verbatim from the transcript.",
        "parameters": _ENTITY_SCHEMA,
    },
}]

_CALL_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.S)


def parse_tool_calls(generated: str) -> list[tuple[str, dict]]:
    """Parse Qwen 3.5's XML-style tool calls into [(function_name, {param: value})]."""
    generated = generated.split("<|im_end|>")[0]
    return [(name, dict(_PARAM_RE.findall(body))) for name, body in _CALL_RE.findall(generated)]


def build_user_message(text: str, use_hints: bool) -> str:
    msg = f"Transcript: {text}"
    if use_hints:
        hint = "; ".join(f"{label} '{t}'" for label, t in rule_candidates(text)) or "none found"
        msg += ("\n\nPattern-matcher candidates (unverified: the list is INCOMPLETE and may contain wrong or "
                "mislabeled items. Verify each one, drop the wrong ones, and also annotate any entities it "
                f"missed, especially facility names, waypoints and commands): {hint}")
    return msg
