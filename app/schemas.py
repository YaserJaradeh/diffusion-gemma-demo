"""Pydantic (v2) models for the WebSocket protocol and REST responses.

These models mirror the shared contract exactly.  They are used to *validate
inbound* WebSocket messages and to *type* the REST responses.  The streaming
engines themselves yield plain ``dict`` frames (see :mod:`app.engine` /
:mod:`app.mock_engine`) which the server forwards verbatim as JSON, so the
outbound models below double as living documentation of those frames.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.config import GENERATION_DEFAULTS

# ---------------------------------------------------------------------------
# Client -> server messages
# ---------------------------------------------------------------------------


class GenerationParams(BaseModel):
    """Generation parameters sent by the client (defaults from the contract).

    Validation here is intentionally lenient (types only); the server additionally
    runs :func:`app.config.clamp_params` to clamp values to the documented bounds.
    """

    model_config = ConfigDict(extra="ignore")

    max_new_tokens: int = GENERATION_DEFAULTS["max_new_tokens"]
    canvas_length: int = GENERATION_DEFAULTS["canvas_length"]
    max_denoising_steps: int = GENERATION_DEFAULTS["max_denoising_steps"]
    temperature_start: float = GENERATION_DEFAULTS["temperature_start"]
    temperature_end: float = GENERATION_DEFAULTS["temperature_end"]
    entropy_bound: float = GENERATION_DEFAULTS["entropy_bound"]
    adaptive_stop_threshold: float = GENERATION_DEFAULTS["adaptive_stop_threshold"]
    top_k: int = GENERATION_DEFAULTS["top_k"]
    seed: Optional[int] = GENERATION_DEFAULTS["seed"]


class GenerateRequest(BaseModel):
    """The START message: ``{"type": "generate", "prompt": ..., "params": {...}}``."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["generate"]
    prompt: str = ""
    params: GenerationParams = Field(default_factory=GenerationParams)


class CancelRequest(BaseModel):
    """The CANCEL message: ``{"type": "cancel"}``."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["cancel"]


# ---------------------------------------------------------------------------
# Server -> client messages
# ---------------------------------------------------------------------------


class TokenView(BaseModel):
    """One token cell in the current rendering of the active canvas."""

    text: str
    state: Literal["mask", "committed", "tentative"]
    confidence: float = Field(ge=0.0, le=1.0)
    changed: bool


class StartMessage(BaseModel):
    """First frame of a generation run."""

    type: Literal["start"] = "start"
    mode: Literal["real", "mock"]
    canvas_length: int
    max_denoising_steps: int
    prompt: str
    #: Optional metadata flag set by the real engine's fallback path to signal
    #: that the per-step reveal was *reconstructed* from the final text rather
    #: than captured live.  Defaults to ``False`` so the base contract is intact.
    approx: bool = False


class StepMessage(BaseModel):
    """A single denoising-step frame for the active canvas.

    ``tokens`` is the FULL current rendering of the active canvas (mask entries
    included) so the client can render the whole block resolving in place.
    """

    type: Literal["step"] = "step"
    canvas_index: int
    step: int
    total_steps: int
    tokens: List[TokenView]
    committed_text: str
    newly_committed: List[str]


class CanvasCommittedMessage(BaseModel):
    """Emitted once a canvas (block) is fully denoised and committed."""

    type: Literal["canvas_committed"] = "canvas_committed"
    canvas_index: int
    text: str


class GenerationStats(BaseModel):
    """Summary statistics returned with the final ``done`` frame."""

    canvases: int
    steps_total: int
    tokens: int
    gen_seconds: float
    tokens_per_second: float
    #: Estimated *autoregressive* throughput for the same output. An AR model
    #: emits one token per forward pass, so it would need ``tokens`` sequential
    #: passes where this run used ``steps_total``. Assuming equal per-pass
    #: latency, ``ar_tokens_per_second ~= steps_total / gen_seconds`` (the
    #: measured diffusion tok/s divided by tokens-committed-per-step). An
    #: estimate -- see :func:`app.mock_engine.estimate_ar_tokens_per_second`.
    #: Defaults to ``0.0`` so older frames remain valid.
    ar_tokens_per_second: float = 0.0


class DoneMessage(BaseModel):
    """Final frame: the complete decoded text plus stats."""

    type: Literal["done"] = "done"
    text: str
    stats: GenerationStats


class ErrorMessage(BaseModel):
    """Error frame sent on any failure during a run."""

    type: Literal["error"] = "error"
    message: str


# ---------------------------------------------------------------------------
# REST responses
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Body of ``GET /api/health``."""

    status: Literal["ok"] = "ok"
    mode: Literal["real", "mock"]
    model_id: str
    device: str
    precision: str
    #: Set when real mode was requested but the server fell back to mock; a short
    #: human-readable reason + fix hint the UI can surface as a warning.
    detail: Optional[str] = None


class ConfigResponse(BaseModel):
    """Body of ``GET /api/config``."""

    defaults: Dict[str, object]
    bounds: Dict[str, Dict[str, float]]
    display_defaults: Dict[str, object]
