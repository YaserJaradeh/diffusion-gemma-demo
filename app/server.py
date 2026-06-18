"""FastAPI application, routes, WebSocket streaming and CLI entrypoint.

Run it either way:

* ``python -m app.server [flags]``           (CLI -- see :func:`main`)
* ``uvicorn app.server:app``                 (config from ``DG_*`` env vars)

Heavy ML imports are confined to :mod:`app.engine` and only happen when a real
engine is actually constructed, so this module (and the whole server in mock
mode) needs only ``fastapi`` + ``uvicorn`` + ``pydantic``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.config import (
    DEFAULT_MODEL_ID,
    REVEAL_MODES,
    SUPPORTED_PRECISIONS,
    AppConfig,
    api_config_payload,
    clamp_params,
)
from app.mock_engine import MockEngine
from app.schemas import ConfigResponse, GenerateRequest, HealthResponse

logger = logging.getLogger("diffusiongemma.server")

#: Absolute path to the frontend directory served at ``/static`` and ``/``.
#: Resolved relative to this file (``app/server.py`` -> repo root ``/web``) so the
#: app is portable to any checkout path (e.g. a different prefix on the cluster).
#: Override with ``DG_WEB_DIR`` if you relocate the frontend.
WEB_DIR = os.environ.get(
    "DG_WEB_DIR", str(Path(__file__).resolve().parent.parent / "web")
)

# Ensure the directory exists so StaticFiles can mount it even before the
# frontend is built (its files are added later by the frontend tooling).
os.makedirs(WEB_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Engine / application state
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AppState:
    """Resolved runtime state: the active engine and the mode actually serving."""

    config: AppConfig
    engine: Any
    mode: str  # "real" | "mock"
    #: Set when real mode was requested but we fell back to mock; a short,
    #: human-readable reason surfaced in the banner and on ``/api/health``.
    fallback_reason: Optional[str] = None
    #: An actionable hint (e.g. the command to install GPU deps) for the banner.
    fallback_hint: Optional[str] = None


#: Set by :func:`main` before serving so the lifespan picks up CLI config.
_config_override: Optional[AppConfig] = None
#: Lazily-built singleton state (also lets ``uvicorn app.server:app`` work).
_state: Optional[AppState] = None


def _resolve_config() -> AppConfig:
    """Return the CLI override if present, else build one from the environment."""
    return _config_override if _config_override is not None else AppConfig.from_env()


def _diagnose_load_failure(exc: Exception) -> Tuple[str, str]:
    """Return a short ``(reason, fix_hint)`` for a real-engine load failure."""
    text = f"{exc}".lower()
    missing_dep = isinstance(exc, ImportError) or "no module named" in text
    # Check torchvision / PIL FIRST -- "torchvision" contains the substring
    # "torch", so it must not be swallowed by the torch/transformers branch.
    if missing_dep and ("torchvision" in text or "pillow" in text or "'pil'" in text):
        return (
            "The multimodal Gemma4 processor needs torchvision/PIL (the engine "
            "normally falls back to the text-only tokenizer instead).",
            "For the full processor install:  uv sync --extra gpu --extra multimodal",
        )
    if missing_dep and ("torch" in text or "transformers" in text):
        return (
            "GPU dependencies are not installed (torch/transformers missing).",
            "Install them:  uv sync --extra gpu   (then restart).",
        )
    if "diffusiongemmaforblockdiffusion" in text or (
        missing_dep and "transformers" in text
    ):
        return (
            "Your transformers build lacks DiffusionGemmaForBlockDiffusion.",
            "Install a recent transformers:  "
            'uv pip install "git+https://github.com/huggingface/transformers"',
        )
    if "cuda" in text or "out of memory" in text or "device" in text:
        return (
            f"GPU/CUDA error while loading the model: {exc}",
            "Check that a GPU is visible (nvidia-smi) and has enough free memory "
            "(bf16 ~52GB; try --precision fp8).",
        )
    return (
        f"{type(exc).__name__}: {exc}",
        "Run with --no-mock-fallback to see the full traceback.",
    )


def build_engine(config: AppConfig) -> Tuple[Any, str, Optional[str], Optional[str]]:
    """Construct the engine for ``config``.

    Returns ``(engine, mode, fallback_reason, fallback_hint)``.  When
    ``config.mock`` is set, always returns the mock engine.  Otherwise the real
    engine is attempted; if it fails to load and ``config.mock_fallback`` is on, a
    clear warning is logged, a short reason/hint is returned, and the mock engine
    is served instead.  With fallback disabled the load error propagates.
    """
    if config.mock:
        logger.info("Mock mode requested; using the simulated engine.")
        return MockEngine(config), "mock", None, None

    try:
        from app.engine import RealEngine  # lazy: imports torch/transformers

        engine = RealEngine(config)
        return engine, "real", None, None
    except Exception as exc:
        logger.exception("Failed to load the real DiffusionGemma model.")
        reason, hint = _diagnose_load_failure(exc)
        if config.mock_fallback:
            logger.warning(
                "REAL mode was requested (MOCK=0) but the model failed to load: "
                "%s  -> serving the MOCK engine instead (generations are "
                "simulated). %s  Pass --no-mock-fallback to fail loudly.",
                reason,
                hint,
            )
            return MockEngine(config), "mock", reason, hint
        raise


def get_state() -> AppState:
    """Return the singleton :class:`AppState`, building it on first use."""
    global _state
    if _state is None:
        config = _resolve_config()
        engine, mode, reason, hint = build_engine(config)
        _state = AppState(
            config=config,
            engine=engine,
            mode=mode,
            fallback_reason=reason,
            fallback_hint=hint,
        )
    return _state


def _print_banner(state: AppState) -> None:
    """Print a startup banner with the resolved mode, URL and an SSH tunnel hint."""
    config = state.config
    user = os.environ.get("USER", "user")
    device = getattr(state.engine, "device", "?")
    precision = getattr(state.engine, "precision", config.precision)
    bar = "=" * 70
    mode_line = f"  DiffusionGemma demo   |   MODE: {state.mode.upper()}"
    if state.fallback_reason:
        mode_line += "   (requested REAL — fell back!)"
    lines = [
        bar,
        mode_line,
        bar,
        f"  model:     {config.model_id}",
        f"  device:    {device}        precision: {precision}",
    ]
    if state.mode == "real":
        reveal = getattr(config, "reveal_mode", "generate")
        note = {
            "generate": "(documented generate() + reconstructed reveal — reliable)",
            "custom": "(EXPERIMENTAL true per-step — may be empty/garbled for this model!)",
            "auto": "(custom, else generate)",
        }.get(reveal, "")
        lines.append(f"  reveal:    {reveal}   {note}")
    lines += [
        f"  serving:   {config.display_url}   (bind {config.host}:{config.port})",
        bar,
    ]
    if state.fallback_reason:
        lines += [
            "  !! Real mode (MOCK=0) was requested but the model FAILED to load.",
            "     Serving the simulated MOCK engine instead.",
            f"     reason: {state.fallback_reason}",
            f"     fix:    {state.fallback_hint or 'see the traceback above.'}",
            bar,
        ]
    lines += [
        "  Running on a remote compute node? Tunnel from your laptop:",
        f"    ssh -N -L {config.port}:<computenode>:{config.port} {user}@login-node",
        f"  then open {config.display_url}",
        bar,
    ]
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build state and print the banner once, at startup."""
    state = get_state()
    _print_banner(state)
    yield


app = FastAPI(title="DiffusionGemma Demo", version="1.0.0", lifespan=lifespan)

# Serve the frontend bundle. ``check_dir=False`` keeps mounting safe even if the
# directory is (currently) empty.
app.mount("/static", StaticFiles(directory=WEB_DIR, check_dir=False), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Return ``web/index.html`` from disk (or a placeholder if not built yet)."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>DiffusionGemma</title></head><body>"
            "<h1>DiffusionGemma demo</h1>"
            "<p>The frontend has not been built yet. "
            "Expected <code>web/index.html</code>.</p>"
            "<p>API is up: try <a href='/api/health'>/api/health</a> and "
            "<a href='/api/config'>/api/config</a>.</p>"
            "</body></html>",
            status_code=200,
        )
    with open(index_path, "r", encoding="utf-8") as handle:
        return HTMLResponse(handle.read())


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness / mode report."""
    state = get_state()
    detail = None
    if state.fallback_reason:
        detail = (
            f"Requested real mode but fell back to mock: {state.fallback_reason} "
            f"{state.fallback_hint or ''}".strip()
        )
    return HealthResponse(
        status="ok",
        mode=state.mode,  # type: ignore[arg-type]
        model_id=state.config.model_id,
        device=str(getattr(state.engine, "device", "?")),
        precision=str(getattr(state.engine, "precision", state.config.precision)),
        detail=detail,
    )


@app.get("/api/config", response_model=ConfigResponse)
async def config() -> Dict[str, Any]:
    """Generation parameter defaults, bounds and client-side display defaults."""
    return api_config_payload()


@app.get("/api/debug/generate")
async def debug_generate(
    prompt: str = "Why is the sky blue?", max_new_tokens: int = 64
) -> Dict[str, Any]:
    """Diagnostic: run ONE generation and return exactly what the model produces.

    Hit this in a browser / curl to see the raw output shape, token ids, every
    decode variant, the tokenizer's special-token ids, and what an all-MASK
    forward predicts -- so we can tell why text is missing or garbled. Example::

        curl 'http://localhost:8000/api/debug/generate?prompt=Why+is+the+sky+blue%3F'
    """
    state = get_state()
    fn = getattr(state.engine, "debug_generate", None)
    if fn is None:
        return {"mode": state.mode, "note": "debug_generate not available."}
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, fn, prompt, int(max_new_tokens))
        logger.info("debug_generate(%r) -> keys=%s", prompt[:40], list(result.keys()))
        return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("debug_generate failed.")
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------


async def _safe_send(ws: WebSocket, payload: Dict[str, Any]) -> None:
    """Send ``payload`` as JSON, ignoring errors from a closed socket."""
    with suppress(Exception):
        await ws.send_json(payload)


async def _listen_for_cancel(ws: WebSocket, cancel: asyncio.Event) -> None:
    """Background task: set ``cancel`` on a ``{"type":"cancel"}`` or disconnect.

    The handler cancels this task during teardown once the stream is done; that
    arrives here as :class:`asyncio.CancelledError` on the blocked
    ``receive_json()``. We swallow it and return quietly so it is not logged as
    an unhandled ASGI exception (``CancelledError`` is a ``BaseException``, so it
    is NOT caught by ``except Exception``).
    """
    try:
        while not cancel.is_set():
            message = await ws.receive_json()
            if isinstance(message, dict) and message.get("type") == "cancel":
                logger.info("Cancel requested by client.")
                cancel.set()
                return
    except asyncio.CancelledError:
        # Normal teardown: the handler cancelled us after the stream finished.
        return
    except WebSocketDisconnect:
        cancel.set()
    except Exception:
        # Malformed follow-up message or closed socket: treat as a cancel.
        cancel.set()


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket) -> None:
    """Stream a generation run.

    Protocol:
      1. Client sends the START message ``{"type":"generate","prompt":...,"params":...}``.
      2. Server streams ``start`` / ``step`` / ``canvas_committed`` / ``done`` frames.
      3. Client may send ``{"type":"cancel"}`` at any time to stop early.
    """
    await ws.accept()
    state = get_state()
    cancel = asyncio.Event()

    # 1. Receive + validate the START message.
    try:
        raw = await ws.receive_json()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await _safe_send(ws, {"type": "error", "message": f"invalid message: {exc}"})
        with suppress(Exception):
            await ws.close()
        return

    try:
        request = GenerateRequest.model_validate(raw)
    except ValidationError as exc:
        await _safe_send(
            ws, {"type": "error", "message": f"invalid start message: {exc}"}
        )
        with suppress(Exception):
            await ws.close()
        return

    params = clamp_params(request.params.model_dump())
    logger.info(
        "WS generate start: mode=%s reveal=%s prompt=%r max_new=%s canvas=%s steps=%s",
        state.mode,
        getattr(state.config, "reveal_mode", "-") if state.mode == "real" else "-",
        (request.prompt or "")[:80],
        params.get("max_new_tokens"),
        params.get("canvas_length"),
        params.get("max_denoising_steps"),
    )

    # 2. Stream frames while concurrently listening for a cancel.
    listener = asyncio.create_task(_listen_for_cancel(ws, cancel))
    frame_count = 0
    last_type: Optional[str] = None
    try:
        async for frame in state.engine.stream_generate(request.prompt, params, cancel):
            if cancel.is_set():
                break
            await ws.send_json(frame)
            frame_count += 1
            last_type = frame.get("type")
            if last_type in ("done", "error"):
                logger.info(
                    "WS generate %s: streamed %d frames%s",
                    last_type,
                    frame_count,
                    (
                        f" ({frame.get('stats')})"
                        if last_type == "done"
                        else f" ({frame.get('message')})"
                    ),
                )
    except WebSocketDisconnect:
        logger.info("WS client disconnected after %d frames.", frame_count)
        cancel.set()
    except Exception as exc:
        logger.exception("Error while streaming generation (after %d frames).", frame_count)
        await _safe_send(ws, {"type": "error", "message": str(exc)})
    finally:
        cancel.set()
        listener.cancel()
        # Await the cancelled listener WITHOUT propagating its CancelledError.
        # (suppress(Exception) does NOT catch CancelledError, which is a
        # BaseException; gather(return_exceptions=True) collects it cleanly.)
        await asyncio.gather(listener, return_exceptions=True)
        with suppress(Exception):
            await ws.close()
        if frame_count == 0:
            logger.warning("WS generate produced 0 frames (nothing was streamed).")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Every option defaults to ``None`` (a "not passed" sentinel).  :func:`main`
    starts from :meth:`AppConfig.from_env` (which auto-loads ``.env``) and applies
    only the flags that were explicitly given, so precedence is:
    **CLI flag > environment / .env > built-in default**.
    """
    parser = argparse.ArgumentParser(
        prog="dlm",
        description="Serve the DiffusionGemma block-diffusion demo. Settings fall "
        "back to environment variables / a .env file when a flag is omitted.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"HF model id (env MODEL_ID/DG_MODEL_ID; default {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force the simulated engine (--no-mock to require the real model; "
        "env MOCK/DG_MOCK).",
    )
    parser.add_argument(
        "--precision",
        choices=list(SUPPORTED_PRECISIONS),
        default=None,
        help="Real-engine precision (bf16 ~52GB, fp8 ~18GB; env PRECISION/DG_PRECISION).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device_map (auto/cuda/cpu; env DG_DEVICE; default auto).",
    )
    parser.add_argument(
        "--host", default=None, help="Bind host (env HOST/DG_HOST; default 0.0.0.0)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (env PORT/DG_PORT; default 8000).",
    )
    parser.add_argument(
        "--no-mock-fallback",
        dest="mock_fallback",
        action="store_false",
        default=None,
        help="Do not fall back to mock if the real model fails to load.",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Explicit MASK token id for the real engine if auto-detection fails "
        "(env DG_MASK_TOKEN_ID).",
    )
    parser.add_argument(
        "--reveal-mode",
        choices=list(REVEAL_MODES),
        default=None,
        help="Real-engine reveal: 'generate' (default, reliable: documented "
        "model.generate() + reconstructed reveal) | 'custom' (experimental "
        "per-denoising-step capture) | 'auto' (custom, else generate). "
        "Env REVEAL_MODE/DG_REVEAL_MODE.",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    """CLI entrypoint: parse flags, build config, then serve with uvicorn."""
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # Start from the environment (this also auto-loads a .env file), then apply
    # only the flags the user actually passed. Precedence: CLI > env/.env > default.
    config = AppConfig.from_env()
    overrides: Dict[str, Any] = {}
    if args.model_id is not None:
        overrides["model_id"] = args.model_id
    if args.mock is not None:
        overrides["mock"] = args.mock
    if args.precision is not None:
        overrides["precision"] = args.precision
    if args.device is not None:
        overrides["device"] = args.device
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.mock_fallback is not None:
        overrides["mock_fallback"] = args.mock_fallback
    if args.mask_token_id is not None:
        overrides["mask_token_id"] = args.mask_token_id
    if args.reveal_mode is not None:
        overrides["reveal_mode"] = args.reveal_mode
    # dataclasses.replace re-runs __post_init__ so precision is re-validated.
    config = dataclasses.replace(config, **overrides)

    global _config_override
    _config_override = config

    # Build the engine now so load errors surface before we start serving (and so
    # the heavy model load is not deferred to the first request).
    get_state()

    import uvicorn  # local import: only needed for the CLI runner

    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
