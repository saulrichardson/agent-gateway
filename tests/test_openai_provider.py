import asyncio

from gateway.models import ChatRequest, Message, Role
from gateway.providers.openai import OPENAI_RESPONSES_URL, OpenAIProvider
from gateway.settings import Settings


class _DummyPostClient:
    def __init__(self, captured: dict):
        self.captured = captured

    async def post(self, url: str, json: dict | None = None, headers=None, timeout=None):  # noqa: ANN001
        self.captured["url"] = url
        self.captured["json"] = json
        request = object()

        class _Resp:
            status_code = 200
            headers = {"x-request-id": "req_post"}

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        return _Resp({"output_text": ["ok"], "usage": {}})


class _DummyStreamResponse:
    def __init__(self):
        self.status_code = 200
        self.headers = {"x-request-id": "req_stream"}
        self._closed = False

    async def aiter_raw(self, chunk_size: int | None = None):  # noqa: ANN001
        yield b"event: response.completed\n\n"

    async def aread(self):  # used only on error path
        return b""

    async def aclose(self):
        self._closed = True


class _DummyStreamClient:
    def __init__(self, captured: dict):
        self.captured = captured

    async def stream(self, method: str, url: str, json=None, headers=None, timeout=None):  # noqa: ANN001
        self.captured.update({"method": method, "url": url, "json": json, "headers": headers})
        return _DummyStreamResponse()


def test_chat_uses_responses_api_and_reasoning():
    captured: dict = {}
    provider = OpenAIProvider(client=_DummyPostClient(captured), settings=Settings(openai_api_key="test"))

    async def _run():
        req = ChatRequest(
            provider="openai",
            model="gpt-5-nano",
            messages=[Message(role=Role.USER, content="hi")],
            metadata={"reasoning": {"effort": "medium"}},
        )

        return await provider.chat(req, trace_id="trace")

    resp = asyncio.run(_run())

    assert captured["url"] == OPENAI_RESPONSES_URL
    assert captured["json"]["reasoning"]["effort"] == "medium"
    assert resp.output_text == "ok"


def test_stream_uses_responses_api_and_does_not_force_max_tokens():
    captured: dict = {}
    provider = OpenAIProvider(client=_DummyStreamClient(captured), settings=Settings(openai_api_key="test"))

    async def _run():
        req = ChatRequest(
            provider="openai",
            model="gpt-5-nano",
            messages=[Message(role=Role.USER, content="hello")],
            metadata={"reasoning": {"effort": "high"}},
        )

        provider_request_id, iterator = await provider.stream(
            req, trace_id="trace", buffer_bytes=1024, max_bytes_out=None
        )

        first_chunk = None
        async for chunk in iterator:
            first_chunk = chunk
            break

        return provider_request_id, first_chunk

    provider_request_id, first_chunk = asyncio.run(_run())

    assert captured["url"] == OPENAI_RESPONSES_URL
    assert "max_output_tokens" not in captured["json"]
    assert captured["json"]["reasoning"]["effort"] == "high"
    assert provider_request_id == "req_stream"
    assert first_chunk is not None
