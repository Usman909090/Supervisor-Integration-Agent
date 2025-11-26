# Testing Guide

This project uses `pytest` for API-level checks against the FastAPI supervisor. Tests run in offline mode by default (no OpenAI key needed) because the planner and agents have fallbacks/simulations.

## How to run
- Install dev deps: `pip install pytest httpx fastapi uvicorn openai` (httpx/openai are used by the app; openai may be optional if you stay in fallback mode).
- Run all tests: `pytest -q`
- Run a single file: `pytest tests/test_knowledge_base_builder.py -q`

## What the tests cover
- `tests/test_knowledge_base_builder.py`
  - `/agents` lists `knowledge_base_builder_agent` (registry exposure).
  - Planner monkeypatch forces routing to `knowledge_base_builder_agent`; `/query` returns success, used_agents contains the KB agent, and the answer/intermediate results are populated (validates handshake + supervisor flow with simulated output).

## Adding more tests
- Use `fastapi.testclient.TestClient` or `httpx.AsyncClient` to hit `/query` and `/agents`.
- Monkeypatch `plan_tools_with_llm` for deterministic routing in offline tests.
- Assert the handshake shape: each step should surface `used_agents[*].name/intent/status` and `intermediate_results.step_n` with `status/output/error` per contract.
- When enabling real OpenAI or real agents, add environment-guarded tests (skip if `OPENAI_API_KEY` not set) to verify planner choices and HTTP calls.
