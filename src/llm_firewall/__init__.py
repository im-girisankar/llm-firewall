"""llm-firewall — a runtime guardrail proxy for LLM APIs.

The package is split so the policy/decision core is pure-Python and testable
offline; the FastAPI proxy layer (optional `server` extra) is a thin wrapper
around it.
"""

from llm_firewall.policy import Action, Decision, Policy

__all__ = ["Action", "Decision", "Policy"]
__version__ = "0.1.0"
