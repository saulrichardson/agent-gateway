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

        try:
            response = await self._client.post(
                url,
                json=payload,
                timeout=self._settings.gateway_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Gemini request timed out: {exc}", status_code=504) from exc
        except httpx.RequestError as exc:
            raise ProviderError(f"Gemini request failed: {exc}", status_code=502) from exc
        if response.status_code >= 400:
            body = response.text or ""
            raise ProviderError(
                f"Gemini error {response.status_code}: {_truncate(body)}",
                status_code=response.status_code,
                provider_request_id=_provider_request_id(response),
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

    def _chunk_to_part(chunk: Any) -> dict[str, Any]:
        # Text chunk
        if isinstance(chunk, str):
            return {"text": chunk}

        if isinstance(chunk, dict):
            # Explicit input_text chunk
            if chunk.get("type") == "input_text" and "text" in chunk:
                return {"text": str(chunk.get("text", ""))}

            # Input image provided as data URL
            if chunk.get("type") == "input_image":
                data_url = chunk.get("image_url") or ""
                if data_url.startswith("data:") and ";base64," in data_url:
                    mime, b64 = data_url.split(";base64,", 1)
                    mime = mime.split(":", 1)[1] if ":" in mime else "image/png"
                    return {"inline_data": {"mime_type": mime, "data": b64}}

            # Raw inline data already base64 encoded
            if "image_base64" in chunk:
                return {"inline_data": {"mime_type": "image/png", "data": chunk["image_base64"]}}

            if "text" in chunk:
                return {"text": str(chunk["text"])}

        # Fallback to text-only representation
        return {"text": message.as_text()}

    content = message.content
    parts: list[dict[str, Any]] = []

    if isinstance(content, list):
        parts = [_chunk_to_part(c) for c in content]
    elif isinstance(content, dict):
        parts = [_chunk_to_part(content)]
    else:
        parts = [{"text": message.as_text()}]

    return {"role": role, "parts": parts}


def _normalize_model(model_name: str) -> str:
    return model_name.removeprefix("models/")


def _provider_request_id(response: httpx.Response) -> str | None:
    return (
        response.headers.get("x-request-id")
        or response.headers.get("x-goog-request-id")
        or response.headers.get("x-cloud-trace-context")
    )


def _truncate(text: str, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"
