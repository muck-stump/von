"""FastAPI server for Von implementing TypeSafe-compatible HTTP endpoints."""

import hmac
import os
import asyncio
from typing import Any, Dict, Optional, Union
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .engine import VonEngine
from .types import Question
from .types import SystemOneResponse

app = FastAPI(
    title="Von Decision Server",
    description="Drop-in open source System One decision engine in homage to John von Neumann and Ludwig von Mises",
    version=__version__,
)

# Origins are configurable; default to open read access for a drop-in local
# server. Credentials are only enabled when an explicit origin allowlist is set,
# since "*" with credentials is both insecure and rejected by browsers anyway.
_cors_origins = [o.strip() for o in os.environ.get("VON_CORS_ORIGINS", "*").split(",") if o.strip()]
_cors_wildcard = _cors_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemOneRequest(BaseModel):
    model: str = Field(default="von-latest")
    state: Any = Field(..., description="State object, string, or array to evaluate")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Dict of question definitions")

    @field_validator("questions")
    @classmethod
    def _at_least_one_question(cls, v: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # OpenAPI declares minProperties: 1 -- an empty dict is a malformed
        # request, not a valid no-op; TypeSafe's own API rejects it too.
        if not v:
            raise ValueError("questions must contain at least one entry")
        return v


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "von-decision-server",
        "version": __version__,
        "engine": "von-1.2",
        "homage": "John von Neumann & Ludwig von Mises",
    }


@app.get("/v1/models")
def list_models():
    model_entries = [
        {"name": "von-latest", "description": "Current Von System One decision model", "release_date": "2026-09-23"},
        {"name": "von-1.2.0", "description": "Von 1.2 stable release (order-invariant option scoring)", "release_date": "2026-09-23"},
        {"name": "von-1.1.0", "description": "Von 1.1 alias (resolves to current model)", "release_date": "2026-09-21"},
        {"name": "jev-latest", "description": "TypeSafe Jev compatibility alias", "release_date": "2026-09-21"},
    ]
    data_entries = [
        {"id": m["name"], "object": "model", "owned_by": "von"}
        for m in model_entries
    ]
    return {
        "models": model_entries,
        "object": "list",
        "data": data_entries,
    }


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def system_one_endpoint(
    req: SystemOneRequest,
    authorization: Optional[str] = Header(None),
):
    expected_key = os.environ.get("VON_API_KEY")
    if expected_key:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
        token = authorization.split("Bearer ", 1)[1].strip()
        # Plain != short-circuits on the first mismatched byte, leaking how
        # many leading characters of the guess were correct via response
        # timing; compare_digest compares in constant time regardless of where
        # (or whether) the strings first differ.
        if not hmac.compare_digest(token.encode("utf-8"), expected_key.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

    try:
        engine = VonEngine.get_instance()
        questions: Dict[str, Union[Question, Dict[str, Any]]] = dict(req.questions)
        # engine.evaluate() runs a synchronous PyTorch forward pass (tens of ms
        # to low-hundreds of ms). Calling it directly here would block this
        # coroutine's event loop thread, stalling every other in-flight request
        # (including /health) for the duration of each inference call. Running
        # it in the default thread pool lets FastAPI keep serving concurrently.
        response = await asyncio.to_thread(
            engine.evaluate,
            state=req.state,
            questions=questions,
            model=req.model,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
