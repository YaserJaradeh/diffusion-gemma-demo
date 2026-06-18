"""GPU-free block-diffusion *simulation* with the real streaming contract.

The :class:`MockEngine` lets the entire UI be developed and demoed on a laptop
with no GPU, ``torch`` or ``transformers`` installed.  It emits exactly the same
WebSocket frames as :class:`app.engine.RealEngine` (``start`` / ``step`` /
``canvas_committed`` / ``done``), reproducing the look of the reference video:
text emerging from garbage as a fixed-size canvas is progressively denoised.

Design of the simulation
------------------------
* A canned but prompt-relevant answer is generated (:func:`build_answer`) and
  tokenised into pieces whose concatenation reproduces the text exactly.
* Each canvas (block) starts as all-MASK and resolves over several passes.  The
  number of passes follows the documented behaviour ("~15-20 tokens resolve per
  pass" with adaptive early-stopping), capped by ``max_denoising_steps``.
* Tokens resolve roughly lowest-index-first with jitter (a stand-in for
  lowest-entropy-first), short/punctuation tokens resolve slightly earlier, and
  not-yet-committed cells near the resolving frontier flicker through garbage so
  the block visibly "emerges" rather than appearing all at once.
* Temperature follows the linear ``temperature_start -> temperature_end``
  schedule; confidence noise shrinks as steps progress.
* Deterministic when ``seed`` is set (a per-request :class:`random.Random`).

The module-level helpers :func:`tokenize_text`, :func:`simulate_canvas_steps` and
:func:`estimate_ar_tokens_per_second` are pure-Python (no heavy deps) and are
reused by the real engine's fallback path.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

logger = logging.getLogger("diffusiongemma.mock")

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

#: Splits text into word / whitespace / punctuation pieces such that
#: ``"".join(tokenize_text(s)) == s`` for any input (every character matches
#: exactly one alternative: whitespace, a word run, or a single other char).
#: This keeps committed-text reconstruction exact while giving realistic
#: per-"token" granularity for the animation.
_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

#: Garbage fragments used to render flickering, not-yet-resolved cells.
_GARBAGE_GLYPHS = list("▓▒░◌#§¤@%&≈±∆◊¬‡†·•")
_GARBAGE_LETTERS = "etaoinshrdlucmfwypvbgkjqxz"


def tokenize_text(text: str) -> List[str]:
    """Tokenise ``text`` into pieces whose concatenation equals ``text``."""
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def _garbage_token(rng: random.Random) -> str:
    """Produce a short, garbage-looking fragment for a flickering cell."""
    roll = rng.random()
    if roll < 0.4:
        return rng.choice(_GARBAGE_GLYPHS)
    length = rng.randint(1, 4)
    return "".join(rng.choice(_GARBAGE_LETTERS) for _ in range(length))


def build_answer(prompt: str) -> str:
    """Return a canned but prompt-relevant markdown answer for the demo.

    The aim is to look plausible (and to show off the model's documented strength
    at markdown / structured text), while clearly echoing the user's prompt so a
    live demo feels responsive.
    """
    p = (prompt or "").strip()
    if not p:
        p = "your prompt"
    # A short, human-readable topic derived from the prompt's first line.
    topic = p.splitlines()[0].strip().rstrip("?.! ")
    if len(topic) > 80:
        topic = topic[:77].rstrip() + "..."

    wants_code = bool(
        re.search(r"\b(code|function|python|script|implement|algorithm|bug)\b", p, re.I)
    )

    parts: List[str] = []
    parts.append(f"## {topic.capitalize() if topic else 'Response'}\n\n")
    parts.append(
        "Here is a concise, block-diffusion answer. Unlike left-to-right "
        "generation, the whole block is denoised in place: confident tokens "
        "settle first and then guide the rest.\n\n"
    )
    parts.append("**Key points**\n\n")
    parts.append(f"- The request was about *{topic}*.\n")
    parts.append(
        "- High-confidence (low-entropy) tokens are committed early and act as "
        "context for their neighbours.\n"
    )
    parts.append(
        "- Remaining positions are refined over successive passes until the "
        "canvas stabilises (adaptive stopping).\n\n"
    )
    if wants_code:
        parts.append("```python\n")
        parts.append("def denoise(canvas, steps):\n")
        parts.append('    """Resolve a token canvas via entropy-bounded denoising."""\n')
        parts.append("    for _ in range(steps):\n")
        parts.append("        logits = model(canvas)\n")
        parts.append("        canvas = commit_low_entropy(canvas, logits)\n")
        parts.append("        if stable(canvas):\n")
        parts.append("            break\n")
        parts.append("    return canvas\n")
        parts.append("```\n\n")
    parts.append(
        "In short, the answer materialises as a whole rather than one word at a "
        "time, which is why long, structured outputs render so quickly.\n"
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Core per-canvas simulation (shared with the real engine's fallback path)
# ---------------------------------------------------------------------------


def _reveal_order(canvas_tokens: List[str], rng: random.Random) -> List[int]:
    """Return canvas positions ordered by when they resolve.

    Approximates lowest-entropy-first: roughly lowest-index-first, with jitter,
    nudging short/whitespace/punctuation tokens slightly earlier (they tend to be
    low-entropy in practice).
    """
    n = len(canvas_tokens)
    keyed: List[tuple] = []
    for i, tok in enumerate(canvas_tokens):
        jitter = rng.uniform(-0.18, 0.18)
        ease = -0.06 if (tok.strip() == "" or len(tok.strip()) <= 2) else 0.0
        key = i / max(1, n) + jitter + ease
        keyed.append((key, i))
    keyed.sort(key=lambda kv: kv[0])
    return [i for _, i in keyed]


def simulate_canvas_steps(
    canvas_tokens: List[str],
    params: Dict[str, Any],
    rng: random.Random,
    prefix_text: str,
    canvas_index: int,
    mask_glyph: str,
    show_garbage: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield ``step`` frames for a single canvas resolving from MASK to text.

    Parameters
    ----------
    canvas_tokens:
        The final token strings for this canvas (their concatenation is the
        committed canvas text).
    params:
        Clamped generation parameters (``max_denoising_steps``,
        ``temperature_start/end``, ``entropy_bound``, ``adaptive_stop_threshold``).
    rng:
        Seeded RNG for deterministic playback.
    prefix_text:
        Text already committed by previous canvases (prefix of ``committed_text``).
    canvas_index:
        Index of this canvas/block.
    mask_glyph:
        Glyph used for unresolved cells.
    show_garbage:
        When ``True``, cells near the resolving frontier flicker through garbage.

    Yields
    ------
    dict
        ``step`` frames exactly matching the contract.
    """
    n = len(canvas_tokens)
    if n == 0:
        return

    max_steps = max(1, int(params.get("max_denoising_steps", 48)))
    t_start = float(params.get("temperature_start", 0.8))
    t_end = float(params.get("temperature_end", 0.4))
    entropy_bound = float(params.get("entropy_bound", 0.1))
    adaptive_stop = float(params.get("adaptive_stop_threshold", 0.005))

    # Tokens committed per pass: ~18 by default, looser entropy bound -> more per
    # pass (fewer steps); higher adaptive-stop threshold -> stop a touch earlier.
    tokens_per_pass = max(4, round(18 * (0.5 + 5.0 * entropy_bound)))
    needed = math.ceil(n / tokens_per_pass)
    effective_steps = min(max_steps, max(2, needed))
    effective_steps = max(2, round(effective_steps * (1.0 - min(0.6, adaptive_stop * 20.0))))
    effective_steps = min(effective_steps, max_steps)

    order = _reveal_order(canvas_tokens, rng)
    prev_display: List[Optional[str]] = [None] * n
    prev_target = 0

    for s in range(effective_steps):
        # Cumulative number of committed cells by the end of this step. Ease so a
        # few resolve immediately and the last step commits everything.
        progress = (s + 1) / effective_steps
        target = min(n, max(prev_target, round(n * progress)))
        if s == effective_steps - 1:
            target = n  # guarantee full commitment on the final step

        committed_ranks = set(order[:target])
        # Cells in the "frontier" preview as tentative garbage before committing.
        frontier = set(order[target : min(n, target + tokens_per_pass)])

        newly_positions = sorted(order[prev_target:target])
        prev_target = target

        # Temperature for this step (linear schedule), used only to flavour the
        # confidence noise in the simulation.
        if effective_steps > 1:
            temperature = t_start + (t_end - t_start) * (s / (effective_steps - 1))
        else:
            temperature = t_start
        conf_noise = 0.18 * (1.0 - s / max(1, effective_steps - 1))

        tokens: List[Dict[str, Any]] = []
        for i in range(n):
            if i in committed_ranks:
                conf = 0.80 + 0.18 * progress + rng.uniform(-conf_noise, conf_noise)
                conf = max(0.0, min(1.0, conf))
                text = canvas_tokens[i]
                state = "committed"
            elif show_garbage and i in frontier:
                # Flickering preview -- garbage tinted by temperature.
                conf = max(0.0, min(0.55, rng.uniform(0.05, 0.45) * (0.5 + temperature)))
                text = _garbage_token(rng)
                state = "tentative"
            else:
                conf = 0.0
                text = mask_glyph
                state = "mask"

            changed = text != prev_display[i]
            prev_display[i] = text
            tokens.append(
                {
                    "text": text,
                    "state": state,
                    "confidence": round(conf, 4),
                    "changed": changed,
                }
            )

        committed_text = prefix_text + "".join(
            canvas_tokens[i] for i in range(n) if i in committed_ranks
        )
        newly_committed = [canvas_tokens[i] for i in newly_positions]

        yield {
            "type": "step",
            "canvas_index": canvas_index,
            "step": s,
            "total_steps": effective_steps,
            "tokens": tokens,
            "committed_text": committed_text,
            "newly_committed": newly_committed,
        }


# ---------------------------------------------------------------------------
# Throughput estimate (shared with the real engine)
# ---------------------------------------------------------------------------


def estimate_ar_tokens_per_second(
    tokens: int, steps_total: int, gen_seconds: float
) -> float:
    """Estimate the *autoregressive* (AR) token throughput for the same output.

    Block diffusion commits **many** tokens per sequential model forward pass; a
    classic autoregressive model emits exactly **one** token per forward pass. So
    to produce the same ``tokens`` output, the diffusion run needed ``steps_total``
    sequential passes where an AR model would have needed ``tokens`` of them.

    Holding the per-forward-pass latency equal between the two (a deliberate
    simplifying assumption -- a diffusion pass scores a whole block while an AR
    decode step scores one position, so this is only an *estimate*), the
    AR-equivalent throughput is the diffusion model's pass-rate::

        AR tok/s  ~=  steps_total / gen_seconds
                   =  (tokens / gen_seconds) / (tokens / steps_total)
                   =  measured diffusion tok/s  /  tokens-committed-per-step

    Returned rounded for display.  ``0.0`` when it cannot be computed (no steps or
    no elapsed time).
    """
    if gen_seconds <= 0 or steps_total <= 0:
        return 0.0
    return round(steps_total / gen_seconds, 2)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MockEngine:
    """A GPU-free engine implementing the shared streaming interface."""

    mode = "mock"

    def __init__(self, config: Any) -> None:
        """Store presentation config.  ``config`` is an :class:`app.config.AppConfig`."""
        self.config = config
        self.model_id = getattr(config, "model_id", "mock")
        self.device = "cpu"
        self.precision = getattr(config, "precision", "bf16")
        self.mask_glyph = getattr(config, "mask_glyph", "░")

    def debug_generate(self, prompt: str, max_new_tokens: int = 64) -> Dict[str, Any]:
        """Mock counterpart of :meth:`RealEngine.debug_generate`.

        There is no model here, so this just shows the canned answer the mock
        would reveal (useful to confirm the debug endpoint itself works).
        """
        answer = build_answer(prompt)
        tokens = tokenize_text(answer)[: max(1, int(max_new_tokens))]
        return {
            "mode": "mock",
            "prompt": prompt,
            "note": "Mock engine: no real model. This is the canned answer that "
            "would be revealed. Run in real mode (MODE: REAL) to debug the model.",
            "answer_preview": answer[:600],
            "n_pieces": len(tokens),
        }

    async def stream_generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        cancel: "asyncio.Event",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream simulated denoising frames for ``prompt``.

        Yields dicts that are exactly the server->client messages from the
        contract.  Honours ``cancel`` between steps.
        """
        start_t = time.perf_counter()
        seed = params.get("seed")
        rng = random.Random(seed)

        answer = build_answer(prompt)
        all_tokens = tokenize_text(answer)

        max_new = int(params.get("max_new_tokens", 256))
        if max_new > 0:
            all_tokens = all_tokens[:max_new]

        canvas_len = max(1, int(params.get("canvas_length", 256)))
        canvases = [
            all_tokens[i : i + canvas_len] for i in range(0, len(all_tokens), canvas_len)
        ] or [[]]

        yield {
            "type": "start",
            "mode": "mock",
            "canvas_length": canvas_len,
            "max_denoising_steps": int(params.get("max_denoising_steps", 48)),
            "prompt": prompt,
        }

        prefix_text = ""
        steps_total = 0
        total_tokens = 0
        committed_canvases = 0

        for ci, canvas_tokens in enumerate(canvases):
            if not canvas_tokens:
                continue
            for frame in simulate_canvas_steps(
                canvas_tokens, params, rng, prefix_text, ci, self.mask_glyph
            ):
                if cancel.is_set():
                    logger.info("mock generation cancelled during canvas %d", ci)
                    return
                steps_total += 1
                yield frame
                # Yield control so the server can observe a cancel message.
                await asyncio.sleep(0)

            canvas_text = "".join(canvas_tokens)
            prefix_text += canvas_text
            total_tokens += len(canvas_tokens)
            committed_canvases += 1
            yield {
                "type": "canvas_committed",
                "canvas_index": ci,
                "text": canvas_text,
            }
            await asyncio.sleep(0)

        gen_seconds = max(1e-6, time.perf_counter() - start_t)
        tps = total_tokens / gen_seconds
        yield {
            "type": "done",
            "text": prefix_text,
            "stats": {
                "canvases": committed_canvases,
                "steps_total": steps_total,
                "tokens": total_tokens,
                "gen_seconds": round(gen_seconds, 4),
                "tokens_per_second": round(tps, 2),
                "ar_tokens_per_second": estimate_ar_tokens_per_second(
                    total_tokens, steps_total, gen_seconds
                ),
            },
        }
