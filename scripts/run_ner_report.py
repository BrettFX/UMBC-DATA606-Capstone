#!/usr/bin/env python3
"""Per-label classification report (precision, recall, F1, support) on the frozen human-verified held-out set.

Same layout as scikit-learn's `classification_report`, computed on exact (start, end, label) spans (see src/ner/report.py).
Scores the LLM teacher (the production labels) and the trained student models against the human labels.

Usage:
    python scripts/run_ner_report.py                                  # teacher + selected student (spacy-balanced)
    python scripts/run_ner_report.py --students spacy spacy-balanced
Writes res/ner_classification_report.{md,json} and slide-ready tables res/figures/ner_classification_report_<model>.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ner.report import TEACHER, build_reports, render_image, to_text  # noqa: E402

STUDENT_NAMES = {"spacy": "spaCy student", "spacy-balanced": "spaCy student (rare-label balanced)"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--students", nargs="*", default=["spacy-balanced"], help="Model directories under models/ner/ to score.")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "res/ner_classification_report")
    p.add_argument("--figures", type=Path, default=REPO_ROOT / "res/figures")
    args = p.parse_args(argv)

    reports = build_reports(args.students)
    md = ["# NER classification report on the frozen held-out set",
          "100 human-verified utterances. Exact (start, end, label) span matching. `micro avg` pools all spans "
          "(the span-level counterpart of accuracy); `macro avg` is the unweighted mean over labels with support. "
          "HEADING (3) and SQUAWK (1) have too few spans for their scores to mean much.\n"]
    for title, rows in reports.items():
        print(f"\n{title}\n{to_text(rows)}")
        md += [f"## {title}", "", "```", to_text(rows), "```", ""]
        slug = "teacher" if title == TEACHER else title.removeprefix("Student: ")
        shown = "LLM teacher (Qwen3.5-9B labels)" if title == TEACHER else STUDENT_NAMES.get(slug, f"{slug} student")
        render_image(rows, f"Entity extraction per label: {shown}", args.figures / f"ner_classification_report_{slug}.png",
                     subtitle="Frozen held-out set, 100 human-verified utterances; exact span match (start, end, label)")
    args.out.with_suffix(".md").write_text("\n".join(md))
    args.out.with_suffix(".json").write_text(json.dumps(
        {t: r.round(4).reset_index(names="label").to_dict("records") for t, r in reports.items()}, indent=2))
    print(f"\nWrote {args.out.with_suffix('.md')}, {args.out.with_suffix('.json')} and tables in {args.figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
