"""OpenAI provider adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ..models import ChatRequest, ChatResponse, Message, Role
from ..settings import Settings
from .base import BaseProvider, ProviderError, ProviderNotConfiguredError

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__()
        self._client = client
        self._settings = settings

    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        api_key = self._settings.openai_api_key
        if not api_key:
            raise ProviderNotConfiguredError("OPENAI_KEY is not configured")

        metadata = request.metadata or {}
        payload: dict[str, Any] = {
            "model": request.model,
            "input": [_message_to_responses_format(msg) for msg in request.messages],
        }
        if request.max_tokens:
            payload["max_output_tokens"] = request.max_tokens

        reasoning = metadata.get("reasoning")
        if reasoning:
            payload["reasoning"] = reasoning

        response_format = metadata.get("response_format")
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response = await self._client.post(
            OPENAI_RESPONSES_URL,
            json=payload,
            headers=headers,
            timeout=self._settings.gateway_timeout_seconds,
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"OpenAI error {response.status_code}: {response.text}",
                status_code=response.status_code,
                provider_request_id=response.headers.get("x-request-id"),
            )

        data = response.json()
        output_text = _extract_output_text(data)
        usage = data.get("usage", {})

        return ChatResponse(
            provider=self.name,
            model=request.model,
            output_text=output_text,
            usage=usage,
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            agent_id=request.agent_id,
        )


def _message_to_responses_format(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": _coerce_content(message),
    }


def _coerce_content(message: Message) -> list[dict[str, Any]]:
    content = message.content
    if isinstance(content, list):
        return [_normalize_part(part, message.role) for part in content]
    if isinstance(content, dict):
        return [_normalize_part(content, message.role)]
    return [
        {
            "type": _default_chunk_type(message.role),
            "text": str(content),
        }
    ]


def _normalize_part(part: Any, role: Role) -> dict[str, Any]:
    default_type = _default_chunk_type(role)
    if isinstance(part, str):
        return {"type": default_type, "text": part}
    if isinstance(part, dict):
        if "type" not in part and "text" in part:
            return {
                "type": default_type,
                "text": str(part["text"]),
            }
        if "type" in part:
            return part
    return {"type": default_type, "text": str(part)}


def _default_chunk_type(role: Role) -> str:
    return "output_text" if role == Role.ASSISTANT else "input_text"


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text_field = payload.get("output_text")
    if isinstance(output_text_field, list):
        return "".join(str(chunk) for chunk in output_text_field)

    output_items = payload.get("output", [])
    collected: list[str] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "output_text":
            collected.append(item.get("text", ""))
        elif item_type == "message":
            message_payload = item.get("message") or item
            for content in message_payload.get("content", []):
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    collected.append(content.get("text", ""))
    return "".join(collected)
