# llm-firewall

A runtime **guardrail proxy** that sits in front of any LLM API. It screens
inbound prompts for prompt-injection / jailbreak attempts and scores outbound
responses for hallucination and faithfulness, then **blocks, flags, or allows**
each request according to a policy — a drop-in safety layer for LLM apps.

This is the productized form of the "real-time firewall" idea from my
hallucination-detection thesis, and it composes two of my other repos:
[`llm-redteam`](https://github.com/im-girisankar/llm-redteam) (input detectors)
and [`llm-reliability-kit`](https://github.com/im-girisankar/llm-reliability-kit)
(output scoring).

## Status — milestone roadmap
This repo is built in weekly milestones; each lands as its own commit.

- **M1 ✅ — scaffold + passthrough proxy.** FastAPI proxy forwards Messages-style
  requests to an upstream (Anthropic by default), with a pluggable/mockable
  upstream, a pure-Python policy core, and a first keyword-based input guard.
- **M2 — input guard.** Replace the M1 heuristic with the `llm-redteam`
  injection/jailbreak detectors; block/allow per policy.
- **M3 — output guard.** Wire `llm-reliability-kit` hallucination/faithfulness
  scoring on responses; flag or redact.
- **M4 — streaming, YAML policy config, request logging/dashboard, demo.**

## Design
The decision core (`policy.py`, `guards.py`) is pure-Python and tested offline.
The FastAPI proxy (`proxy.py`) is a thin async layer over it, and the upstream
LLM call is just an injectable async callable — so the whole request path runs
in tests with **no network and no API key**.

```
client ──▶ [ input guard ] ──block?──▶ 403
                 │ allow/flag
                 ▼
           upstream LLM ──▶ [ output guard ] ──▶ response (+flag metadata)
```

## Install & run
```bash
pip install -e ".[server]"
llmfw serve                      # http://127.0.0.1:8000  (Swagger at /docs)
```
Point it at an upstream and provider key via env vars:
```bash
export LLM_FIREWALL_UPSTREAM=https://api.anthropic.com/v1/messages
export ANTHROPIC_API_KEY=sk-...
```
Then send a normal Messages request to `POST /v1/messages` — the firewall
forwards clean prompts and blocks injection attempts with a `403`.

## Develop
```bash
pip install -e ".[dev]"
ruff check src tests
pytest
```
The policy tests run on the bare install; the proxy tests skip automatically if
the `server` extra isn't present.

## License
MIT © 2026 Girisankar G
