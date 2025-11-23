# Project Info: Multi-Agent Supervisor Web App

This file captures the original system prompt describing what we are building and how we are building it.

## What We Are Building
- A web application with a Supervisor LLM that receives user queries, selects worker agents (tools), orchestrates them, and returns a unified answer.
- Worker agents expose a common JSON handshake and can be HTTP or CLI services.
- A minimal frontend sends queries and can show debug info (agents called and intermediate results).

## Architecture (From the Prompt)
- Language: Python 3, Framework: FastAPI, Server: Uvicorn.
- Frontend: simple HTML/JS (now React) via FastAPI route; fetch POST to `/query`.
- LLM: OpenAI Chat Completions (env `OPENAI_API_KEY`).
- Data: in-memory/JSON agent registry; no DB required.
- Single entrypoint (`main.py`) wiring to modular app package.

## Agent Registry
- Each agent has `name`, `description`, `intents`, `type` (http/cli), and connection details (`endpoint` or `command`, `healthcheck`, `timeout_ms`).
- Used for both LLM planning (capability briefing) and actual invocation.

## JSON Contracts
- Frontend → Supervisor (`/query`): `{ query, user_id?, options { debug } }`.
- Supervisor → Worker (request): `{ request_id, agent_name, intent, input { text, metadata }, context { user_id, conversation_id?, timestamp } }`.
- Worker → Supervisor (response): success `{ request_id, agent_name, status: success, output { result, confidence?, details? }, error: null }`; error `{ status: error, output: null, error { type, message } }`.
- Supervisor → Frontend: `{ answer, used_agents[{ name, intent, status }], intermediate_results { step_n: full worker response }, error }`.

## Supervisor Flow
1) Receive user query.
2) Load registry.
3) LLM planner selects agents/steps (plan with `step_id`, `agent`, `intent`, `input_source`).
4) Execute plan: resolve inputs (user_query or prior step output), call agents, collect responses.
5) Compose final answer via LLM (fallback stitching if unavailable).
6) Return structured response; surface errors if planning/execution fails.

## Planner Details
- Prompt includes user query and summarized agents (name/description/intents).
- Expects JSON plan: `steps: [{ step_id, agent, intent, input_source }]` with `input_source` in {`user_query`, `step:X.output.result`}.
- Fallback plan if LLM output invalid or key missing.

## Agent Caller
- Builds handshake request; supports HTTP POST (with timeout/error handling) or simulated/CLI.
- Validates responses; status is `success` or `error` with mutually exclusive output/error.

## Frontend Requirements
- UI: textarea for query, debug checkbox, submit button.
- Sends POST `/query`; displays answer and (when debug) `used_agents` and `intermediate_results`.
- Routes: `GET /` (UI), `POST /query` (full flow), `GET /health` (status ok), optional `/agents` list.

## Implementation Notes
- Keep code clear and well-commented; educational intent.
- Simulate worker responses initially; structure ready for real endpoints.
- Run with `uvicorn main:app --reload` and open http://localhost:8000/.
- Add new worker by updating registry and providing endpoint/command; planner auto-considers via description/intents.

## Running Screenshot
<img width="2058" height="3168" alt="Supervisor Frontend" src="https://github.com/user-attachments/assets/f879d1ba-9cc2-49fe-8825-cbeba037e25c" />

