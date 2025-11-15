"""Run live smoke tests against configured providers."""

from __future__ import annotations

import asyncio
import textwrap

from gateway.models import ChatRequest, Message, Role
from gateway.services.gateway import GatewayService
from gateway.settings import Settings

CASES = [
    (
        "openai:gpt-5-nano:low",
        ChatRequest(
            provider="openai",
            model="gpt-5-nano",
            messages=[
                Message(
                    role=Role.USER,
                    content=(
                        "Provide one sentence explaining why reasoning effort 'low' matters "
                        "for planning."
                    ),
                )
            ],
            metadata={"reasoning": {"effort": "low"}},
        ),
    ),
    (
        "openai:gpt-5-nano:medium",
        ChatRequest(
            provider="openai",
            model="gpt-5-nano",
            messages=[
                Message(
                    role=Role.USER,
                    content=(
                        "Provide one sentence explaining why reasoning effort 'medium' matters "
                        "for planning."
                    ),
                )
            ],
            metadata={"reasoning": {"effort": "medium"}},
        ),
    ),
    (
        "openai:gpt-5-nano:high",
        ChatRequest(
            provider="openai",
            model="gpt-5-nano",
            messages=[
                Message(
                    role=Role.USER,
                    content=(
                        "Provide one sentence explaining why reasoning effort 'high' matters "
                        "for planning."
                    ),
                )
            ],
            metadata={"reasoning": {"effort": "high"}},
        ),
    ),
    (
        "gemini:gemini-2.5-pro-preview-03-25",
        ChatRequest(
            provider="gemini",
            model="gemini-2.5-pro-preview-03-25",
            messages=[
                Message(
                    role=Role.USER,
                    content="Give a fun fact about agent systems in exactly 8 words.",
                )
            ],
        ),
    ),
    (
        "claude:claude-3-5-sonnet-20240620",
        ChatRequest(
            provider="claude",
            model="claude-3-5-sonnet-20240620",
            messages=[
                Message(
                    role=Role.USER,
                    content="Give a short haiku about multi-agent coordination.",
                )
            ],
            max_tokens=120,
        ),
    ),
]


async def main() -> None:
    settings = Settings()
    gateway = GatewayService(settings=settings)
    try:
        for label, request in CASES:
            try:
                response = await gateway.chat(request)
            except Exception as exc:  # noqa: BLE001
                print(f"[{label}] ERROR: {exc}")
            else:
                excerpt = textwrap.shorten(response.output_text.strip(), width=160) or "<empty>"
                print(f"[{label}] {excerpt}")
    finally:
        await gateway.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
