"""Pure-Python policy core for the firewall.

This module has no third-party dependencies so it runs and tests offline. The
proxy layer turns inbound/outbound text into a :class:`Decision` via a
:class:`Policy`. M1 ships the decision plumbing with a trivial built-in check;
later milestones wire in the real injection and hallucination detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    """What the firewall decided to do with a request or response."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


@dataclass
class Decision:
    """The outcome of evaluating a single piece of text against a policy."""

    action: Action
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action is Action.BLOCK


@dataclass
class Policy:
    """Thresholds that map a risk score onto an :class:`Action`.

    A score at or above ``block_at`` blocks; at or above ``flag_at`` flags;
    otherwise the request is allowed. ``block_at`` is expected to be the higher
    threshold.
    """

    flag_at: float = 0.5
    block_at: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.flag_at <= self.block_at <= 1.0:
            raise ValueError("require 0 <= flag_at <= block_at <= 1")

    def decide(self, score: float, reasons: list[str] | None = None) -> Decision:
        """Map a risk ``score`` in [0, 1] onto an action via the thresholds."""
        reasons = reasons or []
        if score >= self.block_at:
            return Decision(Action.BLOCK, score, reasons)
        if score >= self.flag_at:
            return Decision(Action.FLAG, score, reasons)
        return Decision(Action.ALLOW, score, reasons)
