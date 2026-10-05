"""ATC named-entity recognition: label schema, rules, prompts and scoring shared by
the annotation notebooks and the standalone pipeline."""

from .prompt import NER_TOOLS, SYSTEM_PROMPT, SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V3, PROMPTS, build_user_message, parse_tool_calls
from .rules import FACILITY_WORDS, RULE_PATTERNS, rule_candidates
from .schema import (
    NER_LABELS, AnnotatedUtterance, EntitySpan, NerLabel, align_entities, check_constraints,
    entity_spans, is_trainable_utterance, strip_greetings, with_spans,
)
from .scoring import load_gold, score_spans
from .postprocess import ExampleRetriever, format_entities, missed_command_feedback, normalize_entities
from .annotate import FORMAT_SUFFIX, JSON_SCHEMA, VllmBackend, annotate_all, validate
from .pipeline import AnnotationConfig, build_pool, load_cache, run_annotation
