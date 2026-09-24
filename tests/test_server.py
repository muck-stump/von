import asyncio
import time

from von.backends.option_marker_backend import VON_MODEL_ID
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from von.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_models(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    ids = [m["id"] for m in data["data"]]
    assert "von-latest" in ids
    assert VON_MODEL_ID in ids
    assert "von-1.1.0" in ids  # previous version stays listed as an alias


def test_system_one_post(client):
    payload = {
        "model": "von-latest",
        "state": "The user clicked the checkout button but received a credit card decline error.",
        "questions": {
            "error_type": {
                "type": "choice",
                "instructions": "What type of error occurred?",
                "criteria": {
                    "payment_error": "Payment or card transaction failure",
                    "ui_bug": "Layout or display bug",
                },
            },
            "is_payment": {
                "type": "noul",
                "instructions": "Is this a payment failure?",
            },
        },
    }
    res = client.post("/v1/systemone", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == VON_MODEL_ID
    assert "error_type" in data["answers"]
    assert data["answers"]["error_type"]["choice"] == "payment_error"
    assert data["answers"]["is_payment"]["noul"] > 0.5


def test_missing_bearer_token_rejected(client, monkeypatch):
    monkeypatch.setenv("VON_API_KEY", "secret-key")
    res = client.post("/v1/systemone", json={"model": "von-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "x?"}}})
    assert res.status_code == 401


def test_wrong_bearer_token_rejected(client, monkeypatch):
    monkeypatch.setenv("VON_API_KEY", "secret-key")
    res = client.post(
        "/v1/systemone",
        headers={"Authorization": "Bearer wrong-key"},
        json={"model": "von-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "x?"}}},
    )
    assert res.status_code == 401


def test_correct_bearer_token_accepted(client, monkeypatch):
    monkeypatch.setenv("VON_API_KEY", "secret-key")
    res = client.post(
        "/v1/systemone",
        headers={"Authorization": "Bearer secret-key"},
        json={"model": "von-latest", "state": "x", "questions": {"q": {"type": "noul", "instructions": "x?"}}},
    )
    assert res.status_code == 200

@pytest.mark.anyio
async def test_health_stays_responsive_during_inference():
    """/v1/systemone must not block the event loop for the whole request.

    Regression test for the endpoint running inference synchronously inside
    an `async def` handler: that blocks every other coroutine on the same
    event loop (including /health) until the PyTorch forward pass returns.
    A concurrent /health request must complete quickly even while a
    /v1/systemone call is in flight.
    """
    payload = {
        "model": "von-latest",
        "state": "The user clicked the checkout button but received a credit card decline error.",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "What type of error occurred?",
                "criteria": {"payment_error": "Payment failure", "ui_bug": "Display bug"},
            }
        },
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        infer_task = asyncio.create_task(ac.post("/v1/systemone", json=payload))
        # Give the inference request a moment to actually start before racing /health.
        await asyncio.sleep(0.01)
        start = time.monotonic()
        health_res = await ac.get("/health")
        health_elapsed = time.monotonic() - start
        infer_res = await infer_task

    assert health_res.status_code == 200
    assert infer_res.status_code == 200
    # A blocked event loop would make /health wait for the whole inference
    # call (tens to low-hundreds of ms on CPU); this bounds it well under that.
    assert health_elapsed < 0.05, f"/health took {health_elapsed:.3f}s -- event loop is blocked"
