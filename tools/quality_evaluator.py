"""Deterministic, explainable quality scoring for generated text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class QualityEvaluation:
    score: float
    matched_keywords: tuple[str, ...]
    missing_keywords: tuple[str, ...]
    total_keywords: int

    def to_dict(self) -> dict[str, float | int | list[str]]:
        result = asdict(self)
        result["matched_keywords"] = list(self.matched_keywords)
        result["missing_keywords"] = list(self.missing_keywords)
        return result


def _normalize(text: str) -> str:
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    words = re.findall(r"[a-z0-9]+", without_accents.casefold())
    return " ".join(words)


def evaluate_quality_details(expected_keywords: Iterable[str], generated_text: str) -> QualityEvaluation:
    """Score keyword or phrase coverage from 0 to 100 and explain the result."""

    if not isinstance(generated_text, str):
        raise TypeError("generated_text must be a string")

    unique_keywords: dict[str, str] = {}
    for keyword in expected_keywords:
        if not isinstance(keyword, str):
            raise TypeError("Every expected keyword must be a string")
        normalized_keyword = _normalize(keyword)
        if normalized_keyword:
            unique_keywords.setdefault(normalized_keyword, keyword.strip())

    if not unique_keywords:
        return QualityEvaluation(0.0, (), (), 0)

    normalized_text = f" {_normalize(generated_text)} "
    matched = []
    missing = []
    for normalized_keyword, original_keyword in unique_keywords.items():
        if f" {normalized_keyword} " in normalized_text:
            matched.append(original_keyword)
        else:
            missing.append(original_keyword)

    score = round(100 * len(matched) / len(unique_keywords), 2)
    return QualityEvaluation(score, tuple(matched), tuple(missing), len(unique_keywords))


def evaluate_quality(expected_keywords: Iterable[str], generated_text: str) -> float:
    """Return the reproducible 0-100 quality score required by the project."""

    return evaluate_quality_details(expected_keywords, generated_text).score
