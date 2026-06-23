"""Prompt-injection / jailbreak detectors (the M2 "redteam" input guard).

A small ensemble of independent detectors, each returning a calibrated risk
contribution in [0, 1] with a human-readable reason. The aggregate score is the
max over detectors (one strong signal is enough to act on), which keeps the
mapping to the policy thresholds interpretable. Pure-Python and offline.

This mirrors the detector style of the companion `llm-redteam` repo; vendored
here so the firewall has no hard cross-repo dependency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    name: str
    score: float
    reason: str


# (compiled pattern, weight, label) — weights are the score when the pattern hits.
_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)", re.I),
     0.9, "override of prior instructions"),
    (re.compile(r"disregard\s+(the\s+)?(above|previous|system)", re.I),
     0.9, "disregard-system directive"),
    (re.compile(r"\byou\s+are\s+now\b", re.I),
     0.7, "role reassignment ('you are now')"),
    (re.compile(r"\b(reveal|print|show|repeat)\b.{0,30}\bsystem\s+prompt\b", re.I),
     0.85, "system-prompt exfiltration attempt"),
    (re.compile(r"\bdeveloper\s+mode\b|\bDAN\b|\bjailbreak\b", re.I),
     0.8, "known jailbreak persona"),
    (re.compile(r"\bpretend\s+(you|to)\b|\bact\s+as\s+if\b", re.I),
     0.5, "pretend/act-as framing"),
    (re.compile(r"</?(system|instructions?)>", re.I),
     0.6, "injected role tags"),
)


def _pattern_signals(text: str) -> Iterable[Signal]:
    for pattern, weight, label in _PATTERNS:
        if pattern.search(text):
            yield Signal(label, weight, label)


def detect(text: str) -> list[Signal]:
    """Return every detector signal that fired on ``text`` (possibly empty)."""
    return list(_pattern_signals(text))


def score(text: str) -> tuple[float, list[str]]:
    """Aggregate detector signals into a (risk_score, reasons) tuple."""
    signals = detect(text)
    if not signals:
        return 0.0, []
    top = max(s.score for s in signals)
    reasons = [f"{s.name}" for s in sorted(signals, key=lambda s: -s.score)]
    return top, reasons
