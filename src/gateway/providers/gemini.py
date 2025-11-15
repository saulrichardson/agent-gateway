"""Gemini provider adapter."""

from __future__ import annotations

from typing import Any

import httpx

from ..models import ChatRequest, ChatResponse, Message
from ..settings import Settings
from .base import BaseProvider, ProviderError, ProviderNotConfiguredError

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        super().__init__()
        self._client = client
        self._settings = settings

    async def chat(self, request: ChatRequest, trace_id: str) -> ChatResponse:
        api_key = self._settings.gemini_api_key
        if not api_key:
            raise ProviderNotConfiguredError("GEMINI_KEY is not configured")

        model_name = _normalize_model(request.model or DEFAULT_GEMINI_MODEL)
        url = f"{GEMINI_BASE_URL}/{model_name}:generateContent?key={api_key}"

        contents = [_message_to_gemini(message) for message in request.messages]
        payload: dict[str, Any] = {"contents": contents}

        response = await self._client.post(url, json=payload)
        if response.status_code >= 400:
            raise ProviderError(
                f"Gemini error {response.status_code}: {response.text}",
                status_code=response.status_code,
                provider_request_id=response.headers.get("x-request-id"),
            )

        data = response.json()
        candidates = data.get("candidates", [])
        output_text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            output_text = "".join(part.get("text", "") for part in parts)

        usage = data.get("usageMetadata", {})

        return ChatResponse(
            provider=self.name,
            model=model_name,
            output_text=output_text,
            usage=usage,
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            agent_id=request.agent_id,
        )


def _message_to_gemini(message: Message) -> dict[str, Any]:
    role = "user" if message.role.value == "user" else "model"
    text = message.as_text()
    return {
        "role": role,
        "parts": [{"text": text}],
    }


def _normalize_model(model_name: str) -> str:
    return model_name.removeprefix("models/")
