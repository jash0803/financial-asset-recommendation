"""Parse the MiFID questionnaire from the FAR-Trans dataset file.

The file is NOT a CSV -- it's a structured text document with sections,
numbered questions, and lettered answers.  We parse it at import time
and expose QUESTIONS and OPTIONS dicts keyed by ``q<number>``.
"""

from __future__ import annotations

import re
from pathlib import Path

_DEFAULT_PATH = Path("FAR-Trans-Data/questionnaires.csv")


def parse_questionnaire(path: Path = _DEFAULT_PATH) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Parse the structured questionnaire text into (questions, options) dicts."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    questions: dict[str, str] = {}
    options: dict[str, dict[str, str]] = {}

    current_q_id: str | None = None
    q_pattern = re.compile(r"^\s*(\d+)\.\s+(.+)$")
    a_pattern = re.compile(r"^\s+([a-z])\.\s+(.+)$")

    for line in lines:
        q_match = q_pattern.match(line)
        if q_match:
            num = q_match.group(1)
            current_q_id = f"q{num}"
            questions[current_q_id] = q_match.group(2).strip()
            options[current_q_id] = {}
            continue

        a_match = a_pattern.match(line)
        if a_match and current_q_id is not None:
            letter = a_match.group(1)
            options[current_q_id][letter] = a_match.group(2).strip()

    return questions, options


QUESTIONS, OPTIONS = parse_questionnaire()
