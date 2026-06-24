"""Integration tests for the llm-reliability-kit bridge (reliability.py).

These tests exercise :func:`~llm_firewall.reliability.reliability_kit_score_fn`
plugged into :class:`~llm_firewall.guards.OutputGuard`.  They are guarded by
``pytest.importorskip("llm_reliability_kit")`` so the suite still passes in
environments where the kit is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llm_reliability_kit")

from llm_firewall.guards import OutputGuard  # noqa: E402
from llm_firewall.policy import Action, Decision  # noqa: E402
from llm_firewall.reliability import reliability_kit_score_fn  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."
)

# A response whose claims are all present in the context — should produce low risk.
_FAITHFUL_RESPONSE = (
    "The Eiffel Tower stands on the Champ de Mars in Paris. "
    "It was built between 1887 and 1889 for the World's Fair."
)

# A response whose claims are entirely absent from the context — should produce high risk.
_HALLUCINATED_RESPONSE = (
    "The Eiffel Tower was built in New York City in 1950. "
    "It is made entirely of reinforced concrete and stands 2000 metres tall. "
    "It was designed by Gustave Dupont for the Olympic Games."
)


# ---------------------------------------------------------------------------
# score_fn unit tests
# ---------------------------------------------------------------------------


def test_faithfulness_risk_returns_lower_risk_for_grounded_response():
    """reliability_kit_score_fn should give lower risk for a supported response."""
    score_fn = reliability_kit_score_fn()
    grounded_risk, _ = score_fn(_FAITHFUL_RESPONSE, _CONTEXT)
    hallucinated_risk, _ = score_fn(_HALLUCINATED_RESPONSE, _CONTEXT)
    assert grounded_risk < hallucinated_risk, (
        f"Expected grounded risk ({grounded_risk:.3f}) < "
        f"hallucinated risk ({hallucinated_risk:.3f})"
    )


def test_hallucinated_response_produces_high_risk():
    """A fully unsupported response should yield risk > 0.5."""
    score_fn = reliability_kit_score_fn()
    risk, reasons = score_fn(_HALLUCINATED_RESPONSE, _CONTEXT)
    assert risk > 0.5, f"Expected risk > 0.5 for hallucinated response, got {risk:.3f}"
    assert isinstance(reasons, list)


def test_grounded_response_produces_low_risk():
    """A response grounded in the context should yield risk < 0.5."""
    score_fn = reliability_kit_score_fn()
    risk, reasons = score_fn(_FAITHFUL_RESPONSE, _CONTEXT)
    assert risk < 0.5, f"Expected risk < 0.5 for grounded response, got {risk:.3f}"


def test_score_fn_returns_tuple_of_float_and_list():
    """score_fn must return (float, list[str]) regardless of input."""
    score_fn = reliability_kit_score_fn()
    risk, reasons = score_fn(_HALLUCINATED_RESPONSE, _CONTEXT)
    assert isinstance(risk, float)
    assert isinstance(reasons, list)
    assert all(isinstance(r, str) for r in reasons)


def test_risk_in_unit_interval():
    """risk must always be in [0.0, 1.0]."""
    score_fn = reliability_kit_score_fn()
    for response in (_FAITHFUL_RESPONSE, _HALLUCINATED_RESPONSE):
        risk, _ = score_fn(response, _CONTEXT)
        assert 0.0 <= risk <= 1.0, f"risk {risk} out of [0, 1]"


# ---------------------------------------------------------------------------
# OutputGuard integration tests
# ---------------------------------------------------------------------------


def test_output_guard_with_reliability_kit_returns_decision():
    """OutputGuard plugged with reliability_kit_score_fn must return a Decision."""
    guard = OutputGuard(score_fn=reliability_kit_score_fn())
    decision = guard.check(_HALLUCINATED_RESPONSE, _CONTEXT)
    assert isinstance(decision, Decision)
    assert decision.action in {Action.ALLOW, Action.FLAG, Action.BLOCK}


def test_output_guard_blocks_or_flags_hallucinated_response():
    """With a low flag threshold, a hallucinated response must not be ALLOW."""
    # Force a very low flag threshold so the high-risk hallucinated response is caught.
    from llm_firewall.policy import Policy

    guard_strict = OutputGuard(
        policy=Policy(flag_at=0.1, block_at=0.9),
        score_fn=reliability_kit_score_fn(),
    )
    decision = guard_strict.check(_HALLUCINATED_RESPONSE, _CONTEXT)
    assert decision.action in {Action.FLAG, Action.BLOCK}, (
        f"Expected FLAG or BLOCK for hallucinated response, got {decision.action}"
    )


def test_output_guard_allows_faithful_response():
    """A well-grounded response should be ALLOW under default thresholds."""
    from llm_firewall.policy import Policy

    guard = OutputGuard(
        policy=Policy(flag_at=0.5, block_at=0.8),
        score_fn=reliability_kit_score_fn(),
    )
    decision = guard.check(_FAITHFUL_RESPONSE, _CONTEXT)
    assert decision.action is Action.ALLOW, (
        f"Expected ALLOW for faithful response, got {decision.action} "
        f"(reasons: {decision.reasons})"
    )
