"""Application-wide configuration powered by pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    gateway_timeout_seconds: float = Field(default=30.0, alias="GATEWAY_TIMEOUT_SECONDS")
    default_provider: str = Field(default="echo", alias="DEFAULT_PROVIDER")
    max_request_bytes: int = Field(default=256_000, alias="MAX_REQUEST_BYTES")
    default_max_tokens: int = Field(default=2_048, alias="DEFAULT_MAX_TOKENS")
    stream_buffer_bytes: int = Field(default=65_536, alias="STREAM_BUFFER_BYTES")
    max_input_tokens: int = Field(default=6_000, alias="MAX_INPUT_TOKENS")

    # Provider credentials (optional until you supply real keys)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_KEY")
    claude_api_key: str | None = Field(default=None, alias="CLAUDE_KEY")
    census_api_key: str | None = Field(default=None, alias="CENSUS_KEY")


@lru_cache(1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
