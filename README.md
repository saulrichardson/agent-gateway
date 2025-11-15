# Gateway

Project scaffolding for a lightweight LLM gateway service. Uses Poetry for dependency management and includes recommended dev tooling (Black, Ruff, MyPy, Pytest, and pre-commit) plus a FastAPI-based controller that can proxy OpenAI, Gemini, Claude, or the built-in echo provider.

## Requirements

- Python 3.11 (recommended via `pyenv`)
- [Poetry 2.x](https://python-poetry.org/docs/#installation)

## Installation

```bash
poetry install
# or the stricter resolver: poetry sync
```

This creates a virtual environment (see `poetry env info`) and installs the pinned dependencies from `poetry.lock`.

Copy `.env.example` to `.env` and populate the provider keys you intend to use:

```bash
cp .env.example .env
# edit the file as needed
```

## Development workflow

| Task        | Command                      | Notes                              |
|-------------|------------------------------|------------------------------------|
| Run tests   | `poetry run pytest` or `make test` | Minimal Pytest config lives in `pyproject.toml`. |
| Lint        | `poetry run ruff check src tests` or `make lint` | Enforces Ruff lint rules. |
| Format      | `make format`                | Runs Black then auto-fixes via Ruff. |
| Type-check  | `make type-check`            | Strict MyPy config checks `src/`. |
| All hooks   | `make pre-commit`            | Runs the configured pre-commit hooks. |

To keep commits clean:

```bash
poetry run pre-commit install
```

## Running the gateway

```bash
poetry run gateway --reload
# or
poetry run uvicorn gateway.app:create_app --factory --host 0.0.0.0 --port 8000
# Docker (from repo root)
make docker-build
make docker-up
```

Key endpoints:

- `GET /healthz` – liveness plus provider list.
- `POST /v1/responses` – Accepts native OpenAI Responses payloads (`model`, `input`, `response_format`, etc.) and streams spec-compliant SSE events.
- `POST /v1/agents/messages` & `GET /v1/agents/{agent_id}/messages` – lightweight in-memory bus for inter-agent messaging/hand-offs.

Attach `X-Request-ID` headers to correlate client requests with gateway traces.

### Provider configuration

- Set `OPENAI_KEY`, `GEMINI_KEY`, `CLAUDE_KEY`, etc. in `.env`.
- Choose a default provider (`DEFAULT_PROVIDER=echo`, `openai`, `gemini`, or `claude`).
- Prefix the `model` field with the provider when calling `/v1/responses` (e.g. `"openai:gpt-5-nano"`, `"gemini:gemini-2.5-pro-preview-03-25"`, or `"echo:test-model"`). If no prefix is supplied, the default provider is used.
- The OpenAI adapter targets the Responses API (`/v1/responses`) so advanced models like `gpt-5-nano` can use reasoning controls by sending `reasoning_effort`. The Gemini adapter defaults to `gemini-2.5-pro-preview-03-25` on the public API unless you override it by passing a different model name.
- The echo provider is always available for local testing and CI.

### Streaming contract

The gateway mirrors the OpenAI Responses streaming format:

- `event: response.created` with metadata about the request.
- `event: response.output_text.delta` chunks containing incremental `output_text`.
- `event: response.completed` with the final payload plus usage info.

Keep the connection open through `response.completed`. Today each provider returns a single delta, but the event order matches the official Responses API so agents can reuse their existing parsers.

### Smoke tests

Use the helper script to call each provider with real credentials (make sure `.env` is populated first):

```bash
poetry run python scripts/smoke_tests.py
```

The script prints a one-line summary per provider/model. Failures (e.g., missing quota) are surfaced inline so you can distinguish gateway bugs from upstream issues quickly.

### Tracing & retries

The gateway seeds a trace ID per request (exposed as `trace_id` in responses and `X-Request-ID` headers) and uses a shared `httpx.AsyncClient` so you can easily add retries, circuit breakers, or observability exporters later.
- `make docker-build` / `make docker-up` – build/run the container via `docker compose` using your `.env`.

### Agent helper

If you're wiring multiple agents to the gateway, use the included `GatewayAgentClient` to avoid rewriting curl snippets. It keeps an `httpx.AsyncClient` alive, parses the Responses SSE stream, and can bundle local images as multimodal input:

```python
import asyncio
from gateway.client import GatewayAgentClient, build_user_message


async def main() -> None:
    input_messages = [
        {"role": "system", "content": "You are a meticulous captioning agent."},
        build_user_message(
            "Describe this chart in one sentence.",
            image_paths=["/path/to/chart.png"],
        ),
    ]

    async with GatewayAgentClient() as client:
        result = await client.complete_response(
            model="openai:gpt-5-nano",
            input_messages=input_messages,
        )

    print(result["text"])


if __name__ == "__main__":
    asyncio.run(main())
```

Call `client.stream_response(...)` if you want the raw Responses SSE events; `client.complete_response(...)` buffers the text for you. Pass `response_format={"type": "json_object"}` or `reasoning={"effort": "high"}` to forward advanced controls directly to OpenAI.
