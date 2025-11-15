"""HTTP routes for the gateway."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Request, status
from starlette.responses import StreamingResponse

from ..logging import bind_trace
from ..models import AgentEnvelope, ChatRequest, Message, Role
from ..services.gateway import GatewayService
from ..settings import Settings
from .errors import map_exception
from .schemas import ResponseInputMessage, ResponseRequest
from .sse import format_event, sse_response

logger = structlog.get_logger(__name__)
router = APIRouter()


def get_gateway(request: Request) -> GatewayService:
    gateway: GatewayService = request.app.state.gateway
    return gateway


@router.get("/healthz")
async def health_check(request: Request) -> dict[str, object]:
    gateway = get_gateway(request)
    return {
        "status": "ok",
        "environment": gateway._settings.environment,
        "providers": gateway.providers.available_providers(),
    }


@router.post("/v1/responses")
async def create_response(
    payload: ResponseRequest,
    request: Request,
    gateway: GatewayService = Depends(get_gateway),
) -> StreamingResponse:
    settings: Settings = gateway._settings
    provider_name, upstream_model = _parse_model_identifier(
        payload.model, settings.default_provider
    )
    chat_request = _to_chat_request(payload, provider_name, upstream_model)
    trace_id = uuid.uuid4().hex
    bind_trace(trace_id=trace_id, provider=provider_name, model=upstream_model)

    start = time.perf_counter()
    try:
        response = await gateway.chat(chat_request, trace_id=trace_id)
    except Exception as exc:  # noqa: BLE001
        mapped = map_exception(exc, provider_name)
        logger.warning(
            "response.failed",
            trace_id=trace_id,
            provider=provider_name,
            model=upstream_model,
            error=str(exc),
        )
        raise mapped from exc

    first_delta_at = time.perf_counter()
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    async def event_generator() -> AsyncIterator[str]:
        created_payload = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "model": response.model,
                "created": created_at,
                "usage": response.usage or {},
            },
        }
        yield format_event("response.created", created_payload)

        delta_payload = {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "output_text": [response.output_text],
            "trace_id": trace_id,
        }
        yield format_event("response.output_text.delta", delta_payload)

        completed_payload = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "model": response.model,
                "output_text": [response.output_text],
                "usage": response.usage or {},
            },
            "trace_id": trace_id,
        }
        yield format_event("response.completed", completed_payload)

        duration = time.perf_counter() - start
        ttft = first_delta_at - start
        logger.info(
            "response.stream_complete",
            trace_id=trace_id,
            provider=response.provider,
            model=response.model,
            ttft_ms=round(ttft * 1000, 2),
            duration_ms=round(duration * 1000, 2),
            usage=response.usage,
        )

    return sse_response(event_generator())


@router.post("/v1/agents/messages", status_code=status.HTTP_202_ACCEPTED)
async def publish_agent_message(
    payload: AgentEnvelope,
    gateway: GatewayService = Depends(get_gateway),
) -> dict[str, str]:
    gateway.publish_agent_message(payload)
    return {"status": "queued"}


@router.get("/v1/agents/{agent_id}/messages")
async def drain_agent_messages(
    agent_id: str,
    conversation_id: str,
    gateway: GatewayService = Depends(get_gateway),
) -> dict[str, list[AgentEnvelope]]:
    messages = gateway.drain_agent_messages(agent_id=agent_id, conversation_id=conversation_id)
    return {"messages": messages}


def _parse_model_identifier(model: str, default_provider: str) -> tuple[str, str]:
    if ":" in model:
        provider, upstream = model.split(":", 1)
        return provider.lower(), upstream
    return default_provider, model


def _to_chat_request(payload: ResponseRequest, provider: str, upstream_model: str) -> ChatRequest:
    messages = [_convert_message(entry) for entry in payload.input]
    metadata: dict[str, Any] = dict(payload.metadata or {})
    if payload.reasoning:
        metadata["reasoning"] = payload.reasoning
    if payload.response_format:
        metadata["response_format"] = payload.response_format

    return ChatRequest(
        provider=provider,
        model=upstream_model,
        messages=messages,
        temperature=payload.temperature,
        max_tokens=payload.max_output_tokens,
        metadata=metadata or None,
    )


def _convert_message(message: ResponseInputMessage) -> Message:
    role = Role(message.role)
    content = _normalize_content(message.content)
    return Message(role=role, content=content)


def _normalize_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(_is_structured_chunk(entry) for entry in content):
            return content
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if _is_structured_chunk(content):
            return content
        if "text" in content:
            return str(content["text"])
    return str(content)


def _is_structured_chunk(entry: Any) -> bool:
    return isinstance(entry, dict) and (
        "type" in entry or "image_url" in entry or "image_base64" in entry or "image" in entry
    )
