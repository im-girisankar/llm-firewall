"""Guards turn text into a risk score; the policy turns the score into an action.

The input guard runs the M2 detector ensemble (`detectors.py`) over an inbound
prompt; M3 wires the `llm-reliability-kit` hallucination/faithfulness scoring
into the output guard.
"""

from __future__ import annotations

from llm_firewall import detectors
from llm_firewall.policy import Decision, Policy


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
