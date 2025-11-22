from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.models import ChatResponse
from gateway.providers import ProviderError
from gateway.services.gateway import GatewayService
from gateway.settings import Settings


def test_responses_stream_events():
    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "echo:test-model",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "ping"}]},
        ],
    }

    with client.stream("POST", "/v1/responses", json=payload) as response:
        body = "".join(list(response.iter_text()))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [chunk.strip() for chunk in body.split("\n\n") if chunk.strip()]
    assert any("event: response.output_text.delta" in chunk for chunk in events)
    assert any("event: response.completed" in chunk for chunk in events)

    delta_block = next(chunk for chunk in events if "event: response.output_text.delta" in chunk)
    assert "[echo::test-model]" in delta_block


def test_responses_error_mapping(monkeypatch):
    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    async def fake_chat(self, request, trace_id=None):  # noqa: ANN001
        raise ProviderError("rate limited", status_code=429, provider_request_id="req_123")

    monkeypatch.setattr(GatewayService, "chat", fake_chat)

    payload = {
        "model": "echo:test-model",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ],
    }

    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 429
    data = response.json()
    error = data["detail"]["error"]
    assert error["code"] == "upstream_rate_limited"
    assert error["provider_request_id"] == "req_123"


def test_healthz():
    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_requires_openai_key():
    settings = Settings(default_provider="echo", openai_api_key=None)
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/readyz")
    assert response.status_code == 503


def test_request_size_guard(monkeypatch):
    from gateway.api import routes

    monkeypatch.setattr(routes, "MAX_REQUEST_BYTES", 10_000)

    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "echo:test-model",
        "input": [
            {
                "role": "user",
                "content": "x" * 12000,
            }
        ],
    }

    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 413


def test_token_guard(monkeypatch):
    from gateway.api import routes

    monkeypatch.setattr(routes, "MAX_INPUT_TOKENS", 2)

    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    payload = {
        "model": "echo:test-model",
        "input": [
            {
                "role": "user",
                "content": "this will exceed the budget",
            }
        ],
    }

    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 413


def test_responses_keep_structured_messages(monkeypatch):
    settings = Settings(default_provider="echo")
    app = create_app(settings=settings)
    client = TestClient(app)

    captured: dict[str, object] = {}

    async def fake_chat(self, request, trace_id=None):  # noqa: ANN001
        captured["messages"] = request.messages
        return ChatResponse(
            provider="echo",
            model="test-model",
            output_text="ok",
            usage={},
            trace_id=trace_id or "trace",
        )

    monkeypatch.setattr(GatewayService, "chat", fake_chat)

    payload = {
        "model": "echo:test-model",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image."},
                    {
                        "type": "input_image",
                        "image_url": {"url": "https://example.com/test.png"},
                    },
                ],
            }
        ],
    }

    with client.stream("POST", "/v1/responses", json=payload) as response:
        list(response.iter_text())

    assert "messages" in captured
    structured = captured["messages"][0].content
    assert isinstance(structured, list)
    assert structured[1]["type"] == "input_image"
