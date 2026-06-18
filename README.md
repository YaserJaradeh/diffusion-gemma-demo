# DiffusionGemma Demo

An interactive web app that lets you **watch DiffusionGemma think**. Unlike a
normal left-to-right language model, [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
generates text by **iteratively denoising a fixed-size token "canvas"** — a block
of placeholder/`MASK` tokens that gets refined over a handful of passes until the
whole block resolves at once. This demo streams every denoising step to the
browser and lets you replay it at a comfortable, human-watchable speed.

It runs in two modes:

- **Mock mode** — laptop / no-GPU preview. Synthetic denoising, zero model
  weights, instant startup. Perfect for trying the UI and developing the
  frontend.
- **Real mode** — loads the actual model on a single H100 and streams real
  per-step denoising.

---

## What is DiffusionGemma? (the short version)

- **Discrete BLOCK-diffusion** language model — text is produced one *block*
  (canvas) at a time; within a block, tokens are filled in by **denoising**, not
  one-by-one.
- **Mixture-of-Experts**: 26B total parameters, ~3.8B active per token (8 of 128
  experts).
- **How a canvas resolves**: a canvas starts as all `MASK` tokens and is refined
  over up to **48 denoising steps**. Each pass commits the **highest-confidence
  (lowest-entropy) tokens first**; those committed tokens become context for the
  remaining masked positions. Roughly **15–20 tokens resolve per pass**. When a
  canvas is fully denoised it is committed to the KV cache and the next canvas
  begins — so generation is *block-autoregressive*.
- **Attention**: **causal** while ingesting the prompt (prefill), **bidirectional**
  while denoising a canvas.
- **Footprint / speed**: bf16 ≈ **52 GB** (fits one 80 GB H100); fp8 ≈ **18 GB**.
  Reported at **~1000+ tok/s on an H100** and **~700 tok/s on an RTX 5090** —
  about **4× faster than autoregressive** on GPUs. The tradeoff is overall
  quality a touch below standard Gemma, with particular strength at **infilling,
  in-line editing, code, markdown, and structured / non-linear text**.

> Because real generation can exceed **1000 tok/s**, it is *too fast to watch*.
> The demo buffers every frame and plays it back at a speed you control (see
> [Playback controls](#playback--display-controls)).

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
   Your laptop           │            Browser (web/)                 │
   ┌───────────┐         │  index.html                               │
   │  Browser  │◀───────▶│   ├─ /static/css/styles.css               │
   └───────────┘  HTTP   │   ├─ /static/js/app.js   (router/shell)   │
        ▲          + WS   │   ├─ /static/js/howto.js (explainer)      │
        │                 │   └─ /static/js/playground.js (the viewer)│
        │                 │       • buffers every step frame          │
        │                 │       • play/pause/scrub at your speed    │
        │                 └──────────────────────────────────────────┘
        │ ssh -N -L tunnel (real mode on a cluster)
        ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │  FastAPI server (app/server.py)                                     │
 │   GET  /                  -> web/index.html                         │
 │   GET  /static/*          -> StaticFiles(web/)                      │
 │   GET  /api/health        -> mode / model / device / precision      │
 │   GET  /api/config        -> param defaults + bounds                │
 │   WS   /ws/generate       -> streams start/step/canvas_committed/   │
 │                              done/error frames                      │
 │                                                                     │
 │  Engine (app/engine.py)                                             │
 │   ┌─────────────────────┐        ┌──────────────────────────────┐  │
 │   │ MockEngine          │   or   │ RealEngine                    │  │
 │   │ synthetic denoising │        │ DiffusionGemmaForBlockDiffusion│ │
 │   │ no GPU, no weights  │        │ on H100 (bf16 / fp8)          │  │
 │   └─────────────────────┘        └──────────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────┘
```

The engine emits one **frame per denoising step**. Each frame carries the *full*
current rendering of the active canvas (mask entries included), so the browser
can show the whole block resolving in place.

---

## Quickstart

You need [`uv`](https://docs.astral.sh/uv/). Run everything from the repo root.

### A) Mock mode — laptop / no GPU (preview)

```bash
uv sync                          # installs the light core deps only
uv run dlm --mock
# open http://localhost:8000
```

That's it — no token, no weights, no GPU. The viewer will animate a synthetic
canvas denoising so you can explore the UI and playback controls.

### B) Real mode — on an H100

```bash
uv sync --extra gpu              # adds torch / transformers / accelerate / ...
export HF_TOKEN=hf_xxx           # see "HF token & gating" below
uv run dlm --precision bf16
# open http://localhost:8000  (use the SSH tunnel if on a cluster)
```

> **transformers version**: DiffusionGemma is new. If
> `DiffusionGemmaForBlockDiffusion` cannot be imported, install transformers
> from source — `uv pip install "git+https://github.com/huggingface/transformers"`
> — and check the [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
> for the exact required version. fp8 additionally needs appropriate fp8 kernels.

Both entry points work from the project root:

```bash
uv run dlm --mock      # via the [project.scripts] entry point
uvicorn app.server:app                 # direct ASGI app (reads env vars / .env)
```

---

## Slides (presentation deck)

The app ships a built-in **slide deck** — a self-contained presentation view
(canned animations + a bundled video). It needs **no model, no GPU, and no
network**, so you don't have to load anything on the H100 to show it: just start
the app in **mock mode** and open the **Slides** tab.

```bash
uv sync                  # light deps only
uv run dlm --mock        # → http://localhost:8000
```

Then click **Slides** in the top navigation (or jump straight to it at
<http://localhost:8000/#slides>). Because the deck is entirely client-side, mock
mode is all you need for a talk or walkthrough — there's no reason to spin up the
real model just to present.

---

## Configuration

Settings come from (lowest → highest priority): built-in defaults → `.env` →
environment variables → CLI flags. Copy `.env.example` to `.env` to set them
persistently.

| Env var     | Default                              | Meaning                                                  |
|-------------|--------------------------------------|----------------------------------------------------------|
| `HF_TOKEN`  | *(empty)*                            | Hugging Face access token (real mode).                   |
| `HF_HOME`   | *(HF default)*                       | Weight cache location (point at fast/large storage).     |
| `MODEL_ID`  | `google/diffusiongemma-26B-A4B-it`   | Model repo to load.                                      |
| `HOST`      | `0.0.0.0`                            | Bind address.                                            |
| `PORT`      | `8000`                               | Bind port.                                               |
| `PRECISION` | `bf16`                               | `bf16` (~52 GB) or `fp8` (~18 GB).                       |
| `MOCK`      | `0`                                  | `1` forces mock mode.                                    |

### CLI flags

`dlm` (and `python -m app.server`) accept flags that **override**
the env vars above:

| Flag                  | Default            | Description                                                        |
|-----------------------|--------------------|-------------------------------------------------------------------|
| `--mock`              | off                | Force mock mode (no GPU / no weights). Same as `MOCK=1`.          |
| `--host HOST`         | `0.0.0.0`          | Bind address.                                                     |
| `--port PORT`         | `8000`             | Bind port.                                                        |
| `--precision {bf16,fp8}` | `bf16`          | Compute precision in real mode.                                  |
| `--model-id ID`       | `google/diffusiongemma-26B-A4B-it` | Model repo id to load.                          |

> The frontend never sets server flags — these are deployment-time choices.
> Generation parameters (below) are sent per-request over the WebSocket.

---

## Generation parameters (the EB sampler)

DiffusionGemma denoises with **Entropy-Bounded Denoising with Adaptive Stopping
(EB)**. Every parameter below is exposed in the playground and returned from
`GET /api/config`. These **control generation** (they change what the model
produces).

| Parameter                 | Default | Bounds (min/max/step) | What it does                                                                                                   |
|---------------------------|---------|-----------------------|---------------------------------------------------------------------------------------------------------------|
| `prompt`                  | —       | textarea              | Your instruction / context. Ingested causally during prefill.                                                 |
| `max_new_tokens`          | 256     | 32 / 1024 / 32        | Total tokens to generate (across however many canvases that takes).                                            |
| `canvas_length`           | 256     | 64 / 256 / 32         | **Block size** — how many tokens are denoised together in one canvas.                                          |
| `max_denoising_steps`     | 48      | 4 / 64 / 1            | Max refinement passes per canvas. More steps = more chances to resolve tokens (and more frames to watch).     |
| `temperature_start`       | 0.8     | 0.0 / 1.5 / 0.05      | Sampling temperature at the **first** denoising step. EB linearly decays temperature across the steps.        |
| `temperature_end`         | 0.4     | 0.0 / 1.5 / 0.05      | Sampling temperature at the **last** step. Lower-by-the-end → confident tokens lock in cleanly.               |
| `entropy_bound`           | 0.1     | 0.0 / 1.0 / 0.01      | Only tokens whose per-position entropy is **under this bound** get committed (lowest-entropy first).          |
| `adaptive_stop_threshold` | 0.005   | 0.0 / 0.05 / 0.001    | Stop denoising a canvas **early** once the remaining entropy falls below this — saves passes on easy blocks.   |
| `top_k`                   | 0       | 0 / 200 / 1           | Optional top-k truncation of the per-position distribution. `0` = off.                                         |
| `seed`                    | *(null)*| integer (empty = random) | RNG seed for reproducible runs. Empty → random each time.                                                  |

**How it fits together**: a canvas of `canvas_length` `MASK` tokens is refined
for up to `max_denoising_steps`. At each step the model scores every masked
position; positions with entropy below `entropy_bound` are committed (the most
confident first, ~15–20 per pass) and become context for the rest. Temperature
decays from `temperature_start` to `temperature_end` across the steps. If the
canvas's remaining entropy drops below `adaptive_stop_threshold`, it stops early,
commits to the KV cache, and the next canvas begins — repeating until
`max_new_tokens` is reached.

---

## Playback / display controls

These are **client-side only**. They change *how buffered frames are shown* — they
**never** touch generation.

| Control                | Default            | Purpose                                                              |
|------------------------|--------------------|---------------------------------------------------------------------|
| `playback_speed_ms`    | 80 (0–600, step 10)| Milliseconds shown per denoising frame. `0` = as fast as possible.   |
| mode                   | `buffer-then-play` | `buffer-then-play` (default) collects all frames then replays; `live` shows them as they arrive. |
| play / pause           | —                  | Pause and resume the replay.                                         |
| step-back / step-forward | —                | Move one frame at a time.                                            |
| restart                | —                  | Jump to the first frame.                                             |
| scrub-timeline         | —                  | Drag to any frame.                                                   |
| `show_mask_glyphs`     | `true`             | Render `MASK` placeholders as glyphs so you see the canvas shape.    |
| `color_by_confidence`  | `true`             | Tint tokens by commit confidence (low-entropy → solid).             |
| mask glyph style       | —                  | Cosmetic style of the mask placeholder.                             |

**Why this exists**: real generation can exceed **1000 tok/s** — far too fast to
watch a canvas resolve. The client **buffers every frame as it arrives** and
plays them back at a user-controlled rate, fully **decoupling generation speed
from viewing speed**. (In mock mode the same machinery animates the synthetic
denoising.)

---

## HF token & gating

DiffusionGemma is **Apache-2.0** per its model card, but downloading the weights
still typically requires Hugging Face authentication (and accepting the model's
terms on the model page). Provide a token one of these ways:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx     # env var
# or
huggingface-cli login                        # writes to the HF cache
```

Set `HF_HOME` to a fast/large location so the ~52 GB (bf16) download is cached
and reused — especially important on shared clusters. Mock mode needs no token.

---

## Running on a remote H100 + SSH tunnel

Real mode is meant for a GPU box — one 80 GB H100 fits the bf16 model. When the
server runs on a remote machine, reach the UI from your laptop with an SSH
tunnel.

1. Install the GPU deps once on the host: `uv sync --extra gpu`.
2. Set `HF_TOKEN` (gated model) and point `HF_HOME` at fast/large storage so the
   ~52 GB (bf16) download is cached and reused.
3. Start the server on the GPU host (bind all interfaces):

   ```bash
   uv run dlm --host 0.0.0.0 --port 8000
   ```

   The startup banner prints a ready-to-copy tunnel line.
4. From your **laptop**, open the tunnel, then browse to <http://localhost:8000>:

   ```bash
   ssh -N -L 8000:<gpu-host>:8000 <user>@<remote>
   ```

---

## Troubleshooting: it says `MODE: MOCK` but I set `MOCK=0`

Real mode was requested, the model failed to load, and the server **fell back to
mock** (the default; disable with `--no-mock-fallback` to fail loudly). The
startup banner and `GET /api/health` (`detail` field) now tell you exactly why.
Most common causes, in order:

1. **GPU deps not installed** — `ModuleNotFoundError: No module named 'torch'`.
   Fix: `uv sync --extra gpu`, then restart. (`uv sync` alone installs only the
   light mock-mode stack.)
2. **`transformers` too old** — `cannot import name 'DiffusionGemmaForBlockDiffusion'`.
   Install a recent build:
   `uv pip install "git+https://github.com/huggingface/transformers"`
   (see the [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
   for the exact required version).
2b. **The multimodal processor wants `torchvision` / `PIL`** — e.g.
   `Gemma4Processor requires the PIL library`, or
   `No module named 'torchvision'`. DiffusionGemma's processor is the multimodal
   Gemma 4 processor whose *image* path imports `torchvision`/`PIL`. **This demo
   is text-only and the engine automatically falls back to the plain tokenizer**
   (it logs `AutoProcessor unavailable … Falling back to AutoTokenizer`), so you
   normally need **nothing** here. Only if you specifically want the full
   processor: `uv sync --extra gpu --extra multimodal` (pulls a torch-matched
   `torchvision` + `pillow`).
3. **No GPU visible / out of memory** — you're on a login node, or bf16 (~52 GB)
   doesn't fit. Run on the H100 compute node (check `nvidia-smi`); try
   `--precision fp8` (~18 GB).
4. **Gated model** — set `HF_TOKEN` (or `huggingface-cli login`) and accept the
   model terms.

To see the full traceback instead of a silent fallback:

```bash
uv run dlm --no-mock-fallback
```

> Note: loading a 26B model on **CPU** is not practical — real mode is meant for
> the H100. Use mock mode for laptop/CPU previews.

---

## Reveal mode (how the denoising animation is produced)

The real engine can build the garbage→text reveal two ways, chosen with
`--reveal-mode` (or `REVEAL_MODE` in `.env`):

| Mode | What it does | Use it when |
|------|--------------|-------------|
| `generate` **(default)** | Calls the documented `model.generate()` for the **correct** text, then reconstructs a multi-step garbage→text reveal from the real output. | You want a reliable, correct demo. |
| `custom` | Drives the model **one denoising step at a time** and streams the *true* intermediate canvas states. | You've confirmed the model's internal denoising API (see ADAPTATION POINTS) and want the genuine per-step animation. |
| `auto` | Try `custom`, fall back to `generate` if it can't start. | Experimenting. |

**Why `generate` is the default:** DiffusionGemma shipped after this demo's
reference docs, so the exact internal hooks the `custom` loop needs (which logits
correspond to canvas positions, the mask id, causal-vs-bidirectional alignment,
the real sampler) are **plausible but unconfirmed**. A generic guess can produce
empty/garbled output. `generate` relies only on the official API, so it's correct
today; the reveal is reconstructed (still shows tokens resolving from noise) and
the `start` frame carries `approx: true`.

## Known caveat / adaptation (the `custom` path)

> The **true per-denoising-step capture** (`--reveal-mode custom`) may need a
> small tweak to match the model's internals.

- Everything model-specific is **isolated in `app/engine.py`** behind the engine
  interface and marked with **`ADAPTATION POINTS`** comments — the one place to
  adjust. The server, WebSocket contract, and frontend stay unchanged.
- It is written **defensively**: it validates canvas↔logits alignment up front and
  falls back to `generate()` if it can't start, rather than hard-crashing.
- **Mock mode** (`--mock`) and **`generate` mode** both exercise the exact same
  WebSocket contract and frontend, so you can demo the full experience regardless
  of the `custom`-path internals. The bf16/fp8 footprints and EB parameter
  defaults match the published model facts; only the per-step *capture* is the
  open item.

Always cross-check the current
[model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) and the
latest `transformers` for the authoritative API.
