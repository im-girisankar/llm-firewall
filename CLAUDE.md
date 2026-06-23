# llm-firewall — working notes

Runtime guardrail proxy in front of an LLM API. Screens inbound prompts
(injection) and scores outbound responses (hallucination/faithfulness), then
blocks/flags/allows per policy.

## Layout
- `src/llm_firewall/policy.py` — pure decision core (thresholds → Action). No deps.
- `src/llm_firewall/guards.py` — text → (score, reasons); M1 keyword heuristic.
- `src/llm_firewall/proxy.py` — FastAPI app; upstream is an injectable async callable (mock in tests).
- `src/llm_firewall/cli.py` — `llmfw serve`.

## Conventions
- Heavy deps (fastapi/httpx/uvicorn) are optional (`server` extra) and lazy-imported;
  the policy core must stay import-clean and testable offline.
- ruff: line-length 100, select E/F/I/UP/B, ignore E501. `pytest` with `pythonpath=["src"]`.
- Commits authored solely as Girisankar G — no co-author trailers.

## Milestones
M1 scaffold+proxy (done) · M2 redteam input guard · M3 reliability output guard ·
M4 streaming + YAML policy + logging/dashboard + demo.
