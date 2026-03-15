from __future__ import annotations

import time

from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.settings import Settings


def test_response_jobs_complete_and_respect_idempotency_keys() -> None:
    app = create_app(settings=Settings())
    payload = {
        "model": "echo:test-model",
        "input": [{"role": "user", "content": "hello job api"}],
        "stream": False,
    }

    with TestClient(app) as client:
        first = client.post(
            "/v1/responses/jobs",
            json=payload,
            headers={"Idempotency-Key": "same-request"},
        )
        second = client.post(
            "/v1/responses/jobs",
            json=payload,
            headers={"Idempotency-Key": "same-request"},
        )

        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]

        job_id = first.json()["job_id"]
        terminal = None
        for _ in range(50):
            terminal = client.get(f"/v1/responses/jobs/{job_id}")
            assert terminal.status_code == 200
            if terminal.json()["status"] == "succeeded":
                break
            time.sleep(0.01)

        assert terminal is not None
        data = terminal.json()
        assert data["status"] == "succeeded"
        assert data["text"] == "[echo::test-model] hello job api"
        assert data["attempts"] >= 1
