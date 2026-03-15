"""Gateway service orchestrating providers, retries, agent bus, and response jobs."""

from __future__ import annotations

import asyncio
import uuid

import httpx

from ..logging import bind_trace
from ..models import AgentEnvelope, ChatRequest, ChatResponse
from ..providers import (
    ClaudeProvider,
    EchoProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRegistry,
)
from ..settings import Settings
from .agent_bus import AgentBus
from .response_jobs import ResponseJobRecord, ResponseJobStore, response_meta_from_chat


class GatewayService:
    """Main entry point for executing chat calls and relaying agent messages."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.gateway_timeout_seconds)
        self.providers = self._build_registry(settings)
        self.agent_bus = AgentBus()
        self.response_jobs = ResponseJobStore()

    def _build_registry(self, settings: Settings) -> ProviderRegistry:
        registry = ProviderRegistry()
        registry.register(EchoProvider())
        registry.register(OpenAIProvider(client=self._client, settings=settings))
        registry.register(GeminiProvider(client=self._client, settings=settings))
        registry.register(ClaudeProvider(client=self._client, settings=settings))
        return registry

    async def shutdown(self) -> None:
        await self._client.aclose()

    async def chat(self, request: ChatRequest, trace_id: str | None = None) -> ChatResponse:
        trace_id = trace_id or uuid.uuid4().hex
        attempts = max(1, self._settings.response_job_retry_attempts)

        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._chat_once(request, trace_id=trace_id)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts or not _should_retry_failure(exc):
                    raise
                await asyncio.sleep(self._settings.response_job_retry_backoff_seconds * attempt)

        assert last_error is not None
        raise last_error

    async def _chat_once(self, request: ChatRequest, *, trace_id: str) -> ChatResponse:
        provider_name = request.provider or self._settings.default_provider
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ProviderNotConfiguredError(f"Provider '{provider_name}' is not registered")

        bind_trace(trace_id=trace_id, provider=provider.name)
        return await provider.chat(request, trace_id=trace_id)

    async def submit_response_job(
        self,
        request: ChatRequest,
        *,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> ResponseJobRecord:
        if idempotency_key:
            existing = self.response_jobs.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
        resolved_trace_id = trace_id or uuid.uuid4().hex
        record = ResponseJobRecord(
            job_id=f"job_{uuid.uuid4().hex}",
            request=request,
            trace_id=resolved_trace_id,
            idempotency_key=idempotency_key,
            provider=request.provider or self._settings.default_provider,
            model=request.model,
        )
        self.response_jobs.create(record)
        asyncio.create_task(self._run_response_job(record.job_id))
        return record

    def get_response_job(self, job_id: str) -> ResponseJobRecord | None:
        return self.response_jobs.get(job_id)

    async def _run_response_job(self, job_id: str) -> None:
        record = self.response_jobs.update(job_id, status="running")
        if record is None:
            return

        max_attempts = max(1, self._settings.response_job_retry_attempts)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            self.response_jobs.update(job_id, attempts=attempt)
            try:
                response = await self._chat_once(record.request, trace_id=record.trace_id)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= max_attempts or not _should_retry_failure(exc):
                    self.response_jobs.update(
                        job_id,
                        status="failed",
                        error=_job_error_payload(exc, record.provider),
                    )
                    return
                await asyncio.sleep(self._settings.response_job_retry_backoff_seconds * attempt)
                continue

            self.response_jobs.update(
                job_id,
                status="succeeded",
                provider=response.provider,
                model=response.model,
                text=response.output_text,
                meta=response_meta_from_chat(response),
                error=None,
            )
            return

        if last_error is not None:
            self.response_jobs.update(
                job_id,
                status="failed",
                error=_job_error_payload(last_error, record.provider),
            )

    def publish_agent_message(self, envelope: AgentEnvelope) -> None:
        self.agent_bus.publish(envelope)

    def drain_agent_messages(self, agent_id: str, conversation_id: str) -> list[AgentEnvelope]:
        return self.agent_bus.consume(agent_id, conversation_id)


def _should_retry_failure(exc: Exception) -> bool:
    if isinstance(exc, ProviderNotConfiguredError):
        return False
    if isinstance(exc, ProviderError):
        if exc.status_code in (401, 403):
            return False
        if exc.status_code == 429:
            return True
        if exc.status_code is None:
            return True
        return exc.status_code >= 500
    return isinstance(exc, httpx.HTTPError)


def _job_error_payload(exc: Exception, provider: str | None) -> dict[str, object]:
    if isinstance(exc, ProviderNotConfiguredError):
        return {
            "message": str(exc),
            "code": "provider_not_configured",
            "provider": provider,
        }
    if isinstance(exc, ProviderError):
        code = "provider_error"
        if exc.status_code in (401, 403):
            code = "upstream_auth_error"
        elif exc.status_code == 429:
            code = "upstream_rate_limited"
        elif exc.status_code and exc.status_code >= 500:
            code = "upstream_unavailable"
        return {
            "message": str(exc),
            "code": code,
            "provider": provider,
            "upstream_status": exc.status_code,
            "provider_request_id": exc.provider_request_id,
        }
    return {
        "message": str(exc),
        "code": "internal_error",
        "provider": provider,
    }
