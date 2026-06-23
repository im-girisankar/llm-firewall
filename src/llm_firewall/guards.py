"""Guards turn text into a risk score; the policy turns the score into an action.

M1 ships a deliberately simple keyword guard so the end-to-end path (proxy →
guard → policy → decision) is real and testable. M2 replaces the input guard
with the `llm-redteam` injection detectors and M3 wires the `llm-reliability-kit`
hallucination/faithfulness scoring into the output guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from llm_firewall.policy import Decision, Policy

# Crude stand-ins until the real detectors land (M2/M3). Kept lowercase.
_DEFAULT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "you are now",
    "system prompt",
)


def score_injection(text: str, markers: Sequence[str] = _DEFAULT_INJECTION_MARKERS) -> tuple[float, list[str]]:
    """Return a (score, reasons) tuple for prompt-injection risk.

    The M1 heuristic scores by how many known jailbreak markers appear, so a
    single marker already crosses the default block threshold. Real detectors
    arrive in M2.
    """
    low = text.lower()
    hits = [m for m in markers if m in low]
    if not hits:
        return 0.0, []
    score = min(1.0, 0.8 + 0.1 * (len(hits) - 1))
    return score, [f"injection marker: {m!r}" for m in hits]


class InputGuard:
    """Screens an inbound prompt and renders a :class:`Decision`."""

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()

    def check(self, prompt: str) -> Decision:
        score, reasons = score_injection(prompt)
        return self.policy.decide(score, reasons)
