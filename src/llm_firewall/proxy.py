"""FastAPI passthrough proxy in front of an LLM provider.

Inbound requests are screened by the :class:`InputGuard`; if allowed, the body
is forwarded verbatim to the configured upstream (Anthropic's Messages API by
default) and the response is returned. The upstream call goes through a small
``Upstream`` protocol so tests can inject a mock and the core path runs with no
network and no API key.

FastAPI/httpx are imported lazily (the optional ``server`` extra) so importing
this module — and the rest of the package — never requires them.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from llm_firewall.guards import InputGuard
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


def create_app(upstream: Upstream | None = None, policy: Policy | None = None):
    """Build the FastAPI app. ``upstream`` defaults to the real httpx forwarder."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    guard = InputGuard(policy)
    call_upstream = upstream or httpx_upstream()

    app = FastAPI(title="llm-firewall", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages")
    async def messages(body: dict[str, Any]) -> JSONResponse:
        decision = guard.check(_extract_prompt(body))
        if decision.blocked:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "blocked_by_firewall",
                    "score": decision.score,
                    "reasons": decision.reasons,
                },
            )
        status, payload = await call_upstream(body)
        if decision.action.value == "flag":
            payload = {**payload, "_firewall": {"action": "flag", "reasons": decision.reasons}}
        return JSONResponse(status_code=status, content=payload)

    return app
