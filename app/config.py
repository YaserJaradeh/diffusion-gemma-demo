"""Configuration, defaults and runtime settings for the DiffusionGemma demo.

This module is intentionally dependency-light: it imports only the standard
library so it can be loaded in *mock* mode without ``torch`` / ``transformers``.

It is the single source of truth for:

* :class:`AppConfig` -- the resolved runtime settings (model id, mode, precision,
  device, host/port, mock-fallback behaviour, plus presentation knobs such as the
  mask glyph).
* :data:`GENERATION_DEFAULTS` / :data:`GENERATION_BOUNDS` -- the generation
  parameters exposed to the playground, matching the shared contract exactly.
* :data:`DISPLAY_DEFAULTS` -- *client-side only* playback/presentation defaults.
* :func:`api_config_payload` -- the JSON body returned by ``GET /api/config``.
* :func:`clamp_params` -- defensive clamping of an incoming params dict to the
  documented bounds (never rejects, always returns something usable).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("diffusiongemma.config")

# ---------------------------------------------------------------------------
# Model / mode constants
# ---------------------------------------------------------------------------

#: HuggingFace model id for the real engine.
DEFAULT_MODEL_ID = "google/diffusiongemma-26B-A4B-it"

#: Supported precisions for the real engine.
SUPPORTED_PRECISIONS = ("bf16", "fp8")

#: Real-engine reveal strategies (see :class:`AppConfig.reveal_mode`).
REVEAL_MODES = ("generate", "custom", "auto")

#: Glyph used to render a not-yet-resolved (MASK) token cell.  Both the mock
#: engine and the real engine's fallback path use this; the frontend may render
#: its own glyph instead (``show_mask_glyphs`` display option).
DEFAULT_MASK_GLYPH = "░"  # light shade block: ░


# ---------------------------------------------------------------------------
# Generation parameters -- defaults & bounds (the shared contract)
# ---------------------------------------------------------------------------

#: Default values for every generation parameter.  ``seed`` is ``None`` meaning
#: "pick a random seed".
GENERATION_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 256,
    "canvas_length": 256,
    "max_denoising_steps": 48,
    "temperature_start": 0.8,
    "temperature_end": 0.4,
    "entropy_bound": 0.1,
    "adaptive_stop_threshold": 0.005,
    "top_k": 0,
    "seed": None,
}

#: Inclusive min/max and UI step for each generation parameter.  ``seed`` is an
#: optional integer; we still publish bounds so the UI can offer a numeric input.
GENERATION_BOUNDS: Dict[str, Dict[str, float]] = {
    "max_new_tokens": {"min": 32, "max": 1024, "step": 32},
    "canvas_length": {"min": 64, "max": 256, "step": 32},
    "max_denoising_steps": {"min": 4, "max": 64, "step": 1},
    "temperature_start": {"min": 0.0, "max": 1.5, "step": 0.05},
    "temperature_end": {"min": 0.0, "max": 1.5, "step": 0.05},
    "entropy_bound": {"min": 0.0, "max": 1.0, "step": 0.01},
    "adaptive_stop_threshold": {"min": 0.0, "max": 0.05, "step": 0.001},
    "top_k": {"min": 0, "max": 200, "step": 1},
    "seed": {"min": 0, "max": 2 ** 32 - 1, "step": 1},
}

#: Integer-valued generation parameters (used by :func:`clamp_params`).
_INT_PARAMS = frozenset(
    {"max_new_tokens", "canvas_length", "max_denoising_steps", "top_k", "seed"}
)


# ---------------------------------------------------------------------------
# Display / playback defaults (CLIENT-SIDE ONLY -- never affect generation)
# ---------------------------------------------------------------------------

#: Defaults for the client-side playback controls.  These do NOT influence the
#: backend; they are returned purely so the UI can initialise its controls.
DISPLAY_DEFAULTS: Dict[str, Any] = {
    "playback_speed_ms": 80,
    "mode": "buffer-then-play",  # vs. "live"
    "show_mask_glyphs": True,
    "color_by_confidence": True,
    "mask_glyph": DEFAULT_MASK_GLYPH,
    # Name of the client-side glyph set used for the mask-cell flicker. This is
    # the key the frontend's mask-glyph-style dropdown reads from /api/config;
    # it must match one of the frontend's GLYPHS keys
    # (shades|blocks|dots|braille|ascii).
    "mask_glyph_style": "shades",
}


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

#: Guard so a ``.env`` is parsed (and logged) at most once per process.
_DOTENV_LOADED = False


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Minimal ``.env`` parser (stdlib fallback when python-dotenv is absent).

    Handles ``KEY=value``, ``export KEY=value``, ``#`` comments, blank lines and
    surrounding single/double quotes.  Deliberately simple -- python-dotenv is
    used when available (see :func:`load_dotenv_files`).
    """
    data: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return data
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            data[key] = value
    return data


def load_dotenv_files() -> List[str]:
    """Load ``.env`` into ``os.environ`` (real exported vars always win).

    Searched, in order: the repo-root ``.env`` (next to ``.env.example``) and a
    ``.env`` in the current working directory.  Existing environment variables are
    **never** overwritten, so a value exported in the shell still takes
    precedence over the file.  Idempotent: only the first call has an
    effect.  Returns the list of files actually loaded (for logging).
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return []
    _DOTENV_LOADED = True

    root_env = Path(__file__).resolve().parent.parent / ".env"
    cwd_env = Path.cwd() / ".env"
    paths: List[Path] = []
    seen = set()
    for candidate in (root_env, cwd_env):
        key = str(candidate)
        if key not in seen and candidate.is_file():
            seen.add(key)
            paths.append(candidate)
    if not paths:
        return []

    loaded: List[str] = []
    try:
        from dotenv import load_dotenv  # python-dotenv (preferred)

        for path in paths:
            load_dotenv(path, override=False)
            loaded.append(str(path))
    except ImportError:
        for path in paths:
            for env_key, env_value in _parse_env_file(path).items():
                os.environ.setdefault(env_key, env_value)
            loaded.append(str(path))
    return loaded


@dataclasses.dataclass
class AppConfig:
    """Resolved runtime settings for the server.

    Built either from CLI flags (see :func:`app.server.main`) or from environment
    variables (:meth:`from_env`, used by the bare ``uvicorn app.server:app``
    entrypoint).
    """

    model_id: str = DEFAULT_MODEL_ID
    #: When ``True`` always use the mock engine.  When ``False`` try the real
    #: engine first (falling back to mock if :attr:`mock_fallback` is set).
    mock: bool = False
    precision: str = "bf16"
    #: ``device_map`` value forwarded to ``from_pretrained`` (e.g. ``"auto"``,
    #: ``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000
    #: If the real model fails to load, silently serve the mock engine instead.
    mock_fallback: bool = True
    #: Glyph used for unresolved token cells.
    mask_glyph: str = DEFAULT_MASK_GLYPH
    #: Optional explicit MASK token id for the real engine when it cannot be
    #: auto-detected from the tokenizer/config (ADAPTATION POINT in engine.py).
    mask_token_id: Optional[int] = None
    #: How the real engine produces the reveal:
    #:   "generate" - documented model.generate() + reconstructed reveal (RELIABLE,
    #:                default; correct text via the official API).
    #:   "custom"   - experimental true per-denoising-step capture (needs the
    #:                model's exact internal API; may be empty/garbled if it
    #:                doesn't match -- falls back to generate() if it can't start).
    #:   "auto"     - try "custom", fall back to "generate".
    reveal_mode: str = "generate"

    def __post_init__(self) -> None:
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                f"precision must be one of {SUPPORTED_PRECISIONS}, got {self.precision!r}"
            )
        if self.reveal_mode not in REVEAL_MODES:
            raise ValueError(
                f"reveal_mode must be one of {REVEAL_MODES}, got {self.reveal_mode!r}"
            )

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build a config from environment variables.

        Used by the ``uvicorn app.server:app`` entrypoint (no CLI parsing).  For
        each setting the ``DG_``-prefixed name takes precedence, and the bare
        name documented in ``.env.example`` (``MODEL_ID``, ``MOCK``,
        ``PRECISION``, ``HOST``, ``PORT``) is accepted as a fallback so that file
        is actually honoured here.  Standard Hugging Face variables (``HF_TOKEN``
        / ``HF_HOME``) are consumed by the ML libraries directly, not here.
        Unknown / empty variables fall back to the dataclass defaults.
        """
        loaded = load_dotenv_files()
        if loaded:
            logger.info("Loaded environment from %s", ", ".join(loaded))

        def _str(*names: str, default: str) -> str:
            for name in names:
                raw = os.environ.get(name)
                if raw is not None and raw.strip() != "":
                    return raw
            return default

        def _bool(*names: str, default: bool) -> bool:
            for name in names:
                raw = os.environ.get(name)
                if raw is not None:
                    return raw.strip().lower() in {"1", "true", "yes", "on"}
            return default

        precision = _str("DG_PRECISION", "PRECISION", default="bf16").strip().lower()
        if precision not in SUPPORTED_PRECISIONS:
            precision = "bf16"

        mask_id_raw = os.environ.get("DG_MASK_TOKEN_ID")
        mask_token_id = int(mask_id_raw) if (mask_id_raw or "").strip().isdigit() else None

        reveal_mode = _str("DG_REVEAL_MODE", "REVEAL_MODE", default="generate").strip().lower()
        if reveal_mode not in REVEAL_MODES:
            reveal_mode = "generate"

        return cls(
            model_id=_str("DG_MODEL_ID", "MODEL_ID", default=DEFAULT_MODEL_ID),
            mock=_bool("DG_MOCK", "MOCK", default=False),
            precision=precision,
            device=_str("DG_DEVICE", default="auto"),
            host=_str("DG_HOST", "HOST", default="0.0.0.0"),
            port=int(_str("DG_PORT", "PORT", default="8000")),
            mock_fallback=_bool("DG_MOCK_FALLBACK", default=True),
            mask_glyph=_str("DG_MASK_GLYPH", default=DEFAULT_MASK_GLYPH),
            mask_token_id=mask_token_id,
            reveal_mode=reveal_mode,
        )

    # -- helpers -----------------------------------------------------------

    @property
    def display_url(self) -> str:
        """A clickable URL for the banner (``0.0.0.0`` shown as ``localhost``)."""
        host = "localhost" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}"


def clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``params`` clamped to :data:`GENERATION_BOUNDS`.

    Missing keys are filled from :data:`GENERATION_DEFAULTS`.  Values are coerced
    to the correct numeric type and clamped to ``[min, max]``.  This is forgiving
    by design -- it never raises, so a slightly malformed request still produces
    a watchable generation rather than an error.
    """
    out: Dict[str, Any] = dict(GENERATION_DEFAULTS)
    out.update({k: v for k, v in (params or {}).items() if k in GENERATION_DEFAULTS})

    for key, bounds in GENERATION_BOUNDS.items():
        value = out.get(key)
        if key == "seed":
            # Seed is optional; keep ``None`` (random) or coerce to a valid int.
            if value is None or value == "":
                out[key] = None
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                out[key] = None
                continue
        if value is None:
            out[key] = GENERATION_DEFAULTS[key]
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            out[key] = GENERATION_DEFAULTS[key]
            continue
        value = max(bounds["min"], min(bounds["max"], value))
        if key in _INT_PARAMS:
            value = int(round(value))
        out[key] = value

    # Keep the start/end temperature schedule sane (start should not undershoot
    # end by a huge margin; we allow either ordering but guard against NaNs).
    for key in ("temperature_start", "temperature_end"):
        if not math.isfinite(float(out[key])):
            out[key] = GENERATION_DEFAULTS[key]

    return out


def api_config_payload() -> Dict[str, Any]:
    """Return the body for ``GET /api/config``.

    Shape matches the contract::

        {"defaults": {...}, "bounds": {param: {min, max, step}}, "display_defaults": {...}}
    """
    return {
        "defaults": dict(GENERATION_DEFAULTS),
        "bounds": {k: dict(v) for k, v in GENERATION_BOUNDS.items()},
        "display_defaults": dict(DISPLAY_DEFAULTS),
    }
