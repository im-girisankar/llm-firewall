"""Guards turn text into a risk score; the policy turns the score into an action.

The input guard runs the M2 detector ensemble (`detectors.py`) over an inbound
prompt; M3 wires a pure-Python lexical faithfulness scorer into the output guard,
with a pluggable ``score_fn`` seam for dropping in a stronger detector later
(e.g. trust-probe embeddings or the ``llm-reliability-kit`` HRI scorer).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from llm_firewall import detectors
from llm_firewall.policy import Decision, Policy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPLIT_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "it", "its", "this", "that", "i", "we", "you", "he", "she", "they",
    }
)
# Minimum fraction of a sentence's content tokens that must appear in the
# context for that sentence to be considered "supported".
_SUPPORT_THRESHOLD = 0.5
# Maximum length (chars) for a reason string to keep logs readable.
_REASON_MAX_LEN = 120


def _content_tokens(text: str) -> set[str]:
    """Lower-cased, non-stopword word tokens from *text*."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Input guard (M2)
# ---------------------------------------------------------------------------


def score_injection(text: str) -> tuple[float, list[str]]:
    """Return a (score, reasons) tuple for prompt-injection risk.

    Thin wrapper over the detector ensemble so callers don't depend on the
    detector internals.
    """
    return detectors.score(text)


class InputGuard:
    """Screens an inbound prompt and renders a :class:`Decision`."""

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()

    def check(self, prompt: str) -> Decision:
        score, reasons = score_injection(prompt)
        return self.policy.decide(score, reasons)


# ---------------------------------------------------------------------------
# Output guard (M3)
# ---------------------------------------------------------------------------


def faithfulness_risk(response: str, context: str) -> tuple[float, list[str]]:
    """Pure-Python, offline faithfulness scorer.

    Splits *response* into sentences and checks each one for **lexical
    containment**: a sentence is "unsupported" when the fraction of its
    content tokens that also appear in *context* falls below
    ``_SUPPORT_THRESHOLD``.  The overall risk is the fraction of sentences
    that are unsupported.

    Returns
    -------
    (risk, reasons)
        *risk* in [0.0, 1.0]; *reasons* lists the unsupported sentences
        (truncated to ``_REASON_MAX_LEN`` chars each).  When *context* is
        empty or blank, returns ``(0.0, [])`` — without a reference we
        cannot judge faithfulness.

    Notes
    -----
    This is an intentionally lightweight baseline (pure token overlap).
    The injectable ``score_fn`` seam on :class:`OutputGuard` is the right
    place to swap in a stronger detector — e.g. the ``trust-probe``
    embedding similarity scorer or the HRI metric from ``llm-reliability-kit``
    — without touching the rest of the guard or proxy machinery.
    """
    context = context.strip()
    if not context:
        return 0.0, []

    context_tokens = _content_tokens(context)
    sentences = [s.strip() for s in _SPLIT_SENTENCES.split(response) if s.strip()]
    if not sentences:
        return 0.0, []

    unsupported: list[str] = []
    for sentence in sentences:
        s_tokens = _content_tokens(sentence)
        if not s_tokens:
            # Skip sentences that are purely stopwords / punctuation.
            continue
        overlap = s_tokens & context_tokens
        support = len(overlap) / len(s_tokens)
        if support < _SUPPORT_THRESHOLD:
            truncated = sentence[:_REASON_MAX_LEN] + ("…" if len(sentence) > _REASON_MAX_LEN else "")
            unsupported.append(f"unsupported: {truncated}")

    total_content_sentences = sum(1 for s in sentences if _content_tokens(s))
    if total_content_sentences == 0:
        return 0.0, []

    risk = len(unsupported) / total_content_sentences
    return risk, unsupported


# Signature of a faithfulness scorer: (response, context) -> (risk, reasons).
ScoreFn = Callable[[str, str], tuple[float, list[str]]]


class OutputGuard:
    """Screens an LLM response for faithfulness/hallucination and renders a :class:`Decision`.

    Parameters
    ----------
    policy:
        Thresholds that map a risk score to an action.  Defaults to
        ``Policy()`` (flag ≥ 0.5, block ≥ 0.8).
    score_fn:
        Callable ``(response, context) -> (risk_score, reasons)`` that
        measures how well *response* is supported by *context*.  Defaults to
        :func:`faithfulness_risk` — a pure-Python lexical baseline that is
        deterministic and needs no network or heavy dependencies.

        **Pluggable seam**: swap in a stronger scorer without touching the
        proxy or policy layers.  For example:

        * ``trust-probe`` cosine similarity between the response and the
          context embeddings (needs ``sentence-transformers``).
        * The HRI (Hallucination Risk Index) metric from
          ``llm-reliability-kit``, which combines NLI entailment with
          entity-level grounding.

        Any callable matching the ``ScoreFn`` signature works.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self.policy = policy or Policy()
        self.score_fn: ScoreFn = score_fn if score_fn is not None else faithfulness_risk

    def check(self, response: str, context: str) -> Decision:
        """Score *response* against *context* and return a :class:`Decision`."""
        score, reasons = self.score_fn(response, context)
        return self.policy.decide(score, reasons)
