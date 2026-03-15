"""Durable response-job records for gateway-managed model calls."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ..models import ChatRequest, ChatResponse

ResponseJobStatus = Literal["pending", "running", "succeeded", "failed"]


@dataclass
class ResponseJobRecord:
    job_id: str
    request: ChatRequest
    trace_id: str
    idempotency_key: str | None = None
    status: ResponseJobStatus = "pending"
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    text: str | None = None
    meta: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "trace_id": self.trace_id,
            "idempotency_key": self.idempotency_key,
            "attempts": self.attempts,
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "meta": self.meta,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ResponseJobStore:
    """Thread-safe in-memory store for response jobs."""

    def __init__(self) -> None:
        self._records: dict[str, ResponseJobRecord] = {}
        self._idempotency_keys: dict[str, str] = {}
        self._lock = threading.RLock()

    def create(self, record: ResponseJobRecord) -> ResponseJobRecord:
        with self._lock:
            if record.idempotency_key:
                self._idempotency_keys[record.idempotency_key] = record.job_id
            self._records[record.job_id] = record
        return record

    def get(self, job_id: str) -> ResponseJobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def get_by_idempotency_key(self, idempotency_key: str) -> ResponseJobRecord | None:
        with self._lock:
            job_id = self._idempotency_keys.get(idempotency_key)
            if job_id is None:
                return None
            return self._records.get(job_id)

    def update(self, job_id: str, **changes: Any) -> ResponseJobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            for key, value in changes.items():
                setattr(record, key, value)
            record.updated_at = datetime.now(timezone.utc)
            return record


def response_meta_from_chat(response: ChatResponse) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "usage": response.usage or {},
        "trace_id": response.trace_id,
        "provider_request_id": response.provider_request_id,
    }
