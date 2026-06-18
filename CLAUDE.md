# Diffusion Gemma Demo

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

An interactive web demo for Google's **DiffusionGemma `26B-A4B-it`** — a discrete *block-diffusion* (MoE) language model that generates text by iteratively denoising a fixed token "canvas" rather than left-to-right. The app has two views (a scrollable **How-it-works** explainer and a **Playground**) and its centerpiece is visualizing text *resolving from noise*, with a client-side **playback speed** control. It runs on a single H100; the UI is reached over an SSH tunnel.

## Commands

`uv` manages everything. Run from the repo root.

```bash
# Mock mode — laptop/CPU preview, no GPU, no model download (light deps only):
uv sync
uv run dlm --mock            # → http://localhost:8000

# Real mode — on the H100 (loads the 26B model):
uv sync --extra gpu                          # torch, transformers, accelerate, ...
export HF_TOKEN=...                          # model is gated
uv run dlm                   # reveal-mode defaults to "generate"

# uv sync --extra gpu --extra multimodal     # ONLY if you want the full Gemma4
                                             # image processor (torchvision+PIL);
                                             # NOT needed — see "Tokenizer fallback".

# ASGI form (config from env / .env instead of CLI flags):
uvicorn app.server:app
```

Useful flags: `--mock/--no-mock`, `--precision {bf16,fp8}`, `--reveal-mode {generate,custom,auto}`, `--port`, `--no-mock-fallback`, `--mask-token-id`. All settable via env too (precedence: **CLI flag > env / `.env` > default**; `.env` is auto-loaded from the repo root).

### Checks & smoke tests (there is no pytest suite)

```bash
uv run python -m py_compile app/*.py         # Python syntax
node --check web/js/*.js                      # JS syntax (ES modules)
# Boot mock mode, then:
curl localhost:8000/api/health
curl localhost:8000/api/config
curl 'localhost:8000/api/debug/generate?prompt=Why+is+the+sky+blue%3F'   # real mode: dumps raw model output
```

The WebSocket (`/ws/generate`) cannot be exercised by Starlette's `TestClient` (it deadlocks on the concurrent cancel-listener). Use a real `uvicorn` server + a `websockets` client. Engine logic is verified with a **fake model object** (`object.__new__(RealEngine)` + stub `model`/`processor`) under CPU-only torch — no GPU or weights needed; see the conversation history for the pattern.

## Architecture (the big picture)

**Two engines, one streaming interface.** `app/engine.py:RealEngine` (real model) and `app/mock_engine.py:MockEngine` (GPU-free simulation) both expose `async def stream_generate(prompt, params, cancel) -> AsyncIterator[dict]`, yielding frames that the server forwards verbatim as JSON over the WebSocket. `app/server.py` picks the engine at startup and is engine-agnostic.

**Lazy heavy imports are load-bearing.** `torch`/`transformers` are imported **only inside `RealEngine` methods**, never at module top level. This is what lets mock mode (and the whole server) run with just `fastapi`+`uvicorn`+`pydantic`. Do **not** add top-level ML imports to `config.py`, `server.py`, `schemas.py`, or `mock_engine.py`, or you break laptop preview.

**The shared contract is the integration seam.** The backend (`app/schemas.py` + the dict frames the engines yield) and the frontend WebSocket client (`web/js/playground.js`) must agree exactly. The protocol:
- REST: `GET /api/health` (mode/model/device/precision/`detail`), `GET /api/config` (param defaults+bounds + display defaults), `GET /api/debug/generate` (diagnostics). Static assets mounted at `/static`; `GET /` serves `web/index.html`.
- WS `/ws/generate`: client sends `{"type":"generate","prompt","params"}` then optionally `{"type":"cancel"}`. Server streams `start` → `step`* → `canvas_committed` → `done` (or `error`). Each `step` frame carries the **full current canvas** as `tokens[]` (each `{text,state:mask|committed|tentative,confidence,changed}`).

**Generation speed is decoupled from display speed.** The server streams every denoising frame as fast as it's produced; the client (`playground.js`) buffers them all and a separate playback loop renders at a user-controlled `ms/frame`. This is why a 1000+ tok/s model is still watchable, and why stats (`gen_seconds`, `tokens/s`) are measured **server-side around `generate()`** and are unaffected by the playback slider.

**Frontend is fully self-contained — no CDNs, no build step.** Remote GPU compute nodes are usually offline, so `web/` uses only vanilla HTML/CSS + ES modules (system/monospace fonts). `web/js/app.js` boots, fetches `/api/health`+`/api/config`, and routes between the two views; `howto.js` and `playground.js` are the views.

## Critical, non-obvious knowledge

- **`reveal_mode=generate` is the only reliable path; do NOT chase `custom`.** DiffusionGemma's true per-denoising-step states are **not** recoverable via the public API — a naive all-mask forward predicts gibberish (confirmed via `/api/debug/generate`: `custom_argmax_decoded` ≈ `". the the the"`). The default `generate` mode calls the documented `model.generate()` for **correct** text and *reconstructs* the garbage→text reveal (start frame is flagged `approx:true`). The `custom` per-step loop exists but is experimental and empty/garbled for this model — don't make it the default or spend effort "fixing" its output.
- **Call `model.generate()` minimally:** just `generate(**inputs, max_new_tokens=N)`. Passing speculative diffusion sampler kwargs (`max_denoising_steps`, `entropy_bound`, …) changed the output tensor's *shape* on this build. Token extraction (`RealEngine._to_flat_ids`) is deliberately **shape-agnostic** (handles `[batch,seq]`, `[batch,blocks,canvas]`, `GenerateOutput.sequences`).
- **The model post-dates the training cutoff** — its internal denoising API is *plausible but unconfirmed*. Everything model-specific is isolated behind **`ADAPTATION POINTS`** comments in `app/engine.py`. Verify against the real H100; use `/api/debug/generate` to see actual output rather than guessing.
- **Tokenizer fallback:** the processor is a multimodal `Gemma4Processor` whose image path needs `torchvision`/`PIL`. The engine falls back to `AutoTokenizer` for this text-only demo, so the heavy `multimodal` extra is normally unnecessary.
- **Precision:** bf16 default (~52 GB, fits one 80 GB H100); `--fp8` (~18 GB) is an opt-in best-guess path that retries in bf16 on failure.
- **Deployment reality:** the running copy lives on a remote cluster (e.g. `/nfs/.../Gemma`) at a *different path* than this repo and is updated by copying a tarball. `WEB_DIR` resolves relative to `app/server.py` (override with `DG_WEB_DIR`) so it's path-portable. Reach the UI via `ssh -N -L <port>:<computenode>:<port> <user>@<login-node>`.

See `README.md` for the user-facing guide (run modes, all params, remote/tunnel workflow, troubleshooting).
