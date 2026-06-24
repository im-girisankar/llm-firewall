"""FastAPI passthrough proxy in front of an LLM provider.

Inbound requests are screened by the :class:`InputGuard`; if allowed, the body
is forwarded verbatim to the configured upstream (Anthropic's Messages API by
default) and the response is returned.  The upstream response is then screened
by the :class:`OutputGuard` for hallucination / faithfulness before being
passed back to the caller.

The upstream call goes through a small ``Upstream`` protocol so tests can
inject a mock and the core path runs with no network and no API key.

FastAPI/httpx are imported lazily (the optional ``server`` extra) so importing
this module — and the rest of the package — never requires them.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from llm_firewall.guards import InputGuard, OutputGuard
from llm_firewall.policy import Policy

# An upstream is any async callable: json body -> (status_code, json response).
Upstream = Callable[[dict[str, Any]], Awaitable[tuple[int, dict[str, Any]]]]

DEFAULT_UPSTREAM_URL = os.getenv(
    "LLM_FIREWALL_UPSTREAM", "https://api.anthropic.com/v1/messages"
)


def httpx_upstream(url: str = DEFAULT_UPSTREAM_URL) -> Upstream:
    """Build a real upstream that forwards to ``url`` via httpx.

    Imported lazily so the dependency is only needed when actually proxying.
    """
    import httpx

    async def _call(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            headers["x-api-key"] = key
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body, headers=headers)
            return resp.status_code, resp.json()

    return _call


def _extract_prompt(body: dict[str, Any]) -> str:
    """Flatten the user-authored text out of a Messages-style request body."""
    parts: list[str] = []
    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(b.get("text", "") for b in content if isinstance(b, dict))
    return "\n".join(parts)


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Extract the assistant's text from an Anthropic Messages response payload.

    The ``content`` field is a list of blocks; we collect those whose ``type``
    is ``"text"`` and join them.
    """
    parts: list[str] = []
    for block in payload.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            # Support both object-style (.text) and dict-style (["text"]) access.
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def _extract_context(body: dict[str, Any]) -> str:
    """Derive the faithfulness reference context from a request body.

    Priority:
    1. A top-level ``context`` field in the request body.
    2. The system prompt (``system`` field).

    When neither is present we return an empty string so that
    :func:`~llm_firewall.guards.faithfulness_risk` skips scoring — without an
    explicit reference document we cannot judge faithfulness, and scoring the
    response against the bare user question would produce spurious blocks.
    """
    if body.get("context"):
        return str(body["context"])
    if body.get("system"):
        return str(body["system"])
    return ""


def create_app(
    upstream: Upstream | None = None,
    policy: Policy | None = None,
    output_policy: Policy | None = None,
) -> Any:
    """Build the FastAPI app.

    Parameters
    ----------
    upstream:
        Async callable forwarding the request body to an LLM backend.
        Defaults to the real httpx forwarder.
    policy:
        Input-guard policy (injection / jailbreak thresholds).
    output_policy:
        Output-guard policy (faithfulness / hallucination thresholds).
        Defaults to the shared *policy* when provided, otherwise ``Policy()``.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    input_guard = InputGuard(policy)
    output_guard = OutputGuard(output_policy or policy)
    call_upstream = upstream or httpx_upstream()

    app = FastAPI(title="llm-firewall", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages")
    async def messages(body: dict[str, Any]) -> JSONResponse:
        # --- input guard ---
        in_decision = input_guard.check(_extract_prompt(body))
        if in_decision.blocked:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "blocked_by_firewall",
                    "score": in_decision.score,
                    "reasons": in_decision.reasons,
                },
            )

        # --- upstream call ---
        status, payload = await call_upstream(body)

        # Attach input-flag metadata before the output check so we don't lose it.
        if in_decision.action.value == "flag":
            payload = {**payload, "_firewall": {"action": "flag", "reasons": in_decision.reasons}}

        # --- output guard (only on successful upstream responses) ---
        if status == 200:
            response_text = _extract_response_text(payload)
            context = _extract_context(body)
            out_decision = output_guard.check(response_text, context)

            if out_decision.blocked:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "output_blocked",
                        "score": out_decision.score,
                        "reasons": out_decision.reasons,
                    },
                )

            if out_decision.action.value == "flag":
                payload = {
                    **payload,
                    "_firewall_output": {
                        "action": "flag",
                        "reasons": out_decision.reasons,
                    },
                }

        return JSONResponse(status_code=status, content=payload)

    return app
