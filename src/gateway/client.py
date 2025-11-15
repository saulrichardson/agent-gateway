"""Helper utilities for agents that need to call the gateway frequently."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class GatewayAgentClient:
    """Thin async wrapper around the /v1/responses streaming endpoint."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def __aenter__(self) -> "GatewayAgentClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_response(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE events mirroring the OpenAI Responses stream."""

        payload: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "stream": True,
        }
        if response_format:
            payload["response_format"] = response_format
        if reasoning:
            payload["reasoning"] = reasoning
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if metadata:
            payload["metadata"] = metadata

        async with self._client.stream("POST", "/v1/responses", json=payload) as response:
            response.raise_for_status()
            current_event: str | None = None
            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.replace("event:", "", 1).strip()
                    continue
                if line.startswith("data:") and current_event:
                    data = json.loads(line.replace("data:", "", 1).strip())
                    yield {"event": current_event, "data": data}
                    current_event = None

    async def complete_response(
        self,
        *,
        model: str,
        input_messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Collect the streaming response and return the final text + metadata."""

        deltas: list[str] = []
        completed_payload: dict[str, Any] | None = None

        async for event in self.stream_response(
            model=model,
            input_messages=input_messages,
            response_format=response_format,
            reasoning=reasoning,
            metadata=metadata,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ):
            if event["event"] == "response.output_text.delta":
                output_text = event["data"].get("output_text") or []
                deltas.append("".join(str(chunk) for chunk in output_text))
            elif event["event"] == "response.completed":
                completed_payload = event["data"]

        return {
            "text": "".join(deltas),
            "meta": completed_payload or {},
        }


def build_user_message(
    prompt: str,
    *,
    image_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a user message that optionally bundles local images as input_image parts."""

    chunks: list[dict[str, Any]] = []
    if prompt:
        chunks.append({"type": "input_text", "text": prompt})

    for path in image_paths or []:
        chunks.append(_image_chunk_from_path(path))

    if not chunks:
        raise ValueError("A prompt and/or at least one image must be supplied.")

    if len(chunks) == 1 and chunks[0]["type"] == "input_text":
        return {"role": "user", "content": prompt}

    return {"role": "user", "content": chunks}


def _image_chunk_from_path(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(file_path.name)
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    data_url = f"data:{mime_type or 'image/png'};base64,{encoded}"
    chunk: dict[str, Any] = {
        "type": "input_image",
        "image_url": data_url,
    }
    return chunk
