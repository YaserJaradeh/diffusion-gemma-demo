"""Real ``transformers`` block-diffusion engine for DiffusionGemma.

This module drives ``google/diffusiongemma-26B-A4B-it`` and streams per-denoising
-step canvas states over the shared contract.  ``torch`` and ``transformers`` are
imported **lazily, only inside the real code paths**, so importing this module
costs nothing and the server still runs in mock mode with just
``fastapi`` + ``uvicorn`` + ``pydantic`` installed.

================================================================================
ADAPTATION POINTS  (confirm / adjust these on the H100 once the model is loaded)
================================================================================
The DiffusionGemma transformers API was released after this code was written, so
the exact internal denoising API is *plausible but unconfirmed*.  Everything that
touches model internals is isolated below and wrapped defensively; the engine
degrades gracefully (custom step loop -> documented ``generate()`` -> error frame)
and must never hard-crash the UI.  Verify the following:

1. CLASS / LOADING (``RealEngine.__init__``)
   - ``DiffusionGemmaForBlockDiffusion`` and ``AutoProcessor`` import names.
   - ``from_pretrained`` accepts ``dtype`` + ``device_map``.  bf16 is the default;
     the fp8 path (quantization config vs. ``dtype="float8_e4m3fn"``) is a guess.
   - HF auth: loading may require ``HF_TOKEN`` / ``huggingface-cli login`` even
     though the card lists Apache-2.0.

2. MASK TOKEN ID (``RealEngine._detect_mask_id``)
   - Where the placeholder/MASK id lives (tokenizer ``mask_token_id``, a config
     attribute, or a special token like ``<mask>``).  Set ``--mask-token-id`` /
     ``DG_MASK_TOKEN_ID`` if auto-detection fails.

3. STEP CAPTURE (``RealEngine._denoise_canvas`` / ``_forward_logits``)
   - Whether ``model(input_ids=...)`` returns ``.logits`` of shape
     ``[batch, seq, vocab]`` and the off-by-one mapping between sequence position
     and the token it predicts (causal prefill vs. bidirectional denoising).
   - Whether the model exposes a documented per-step hook / scheduler / callback
     that yields intermediate canvas token ids.  If so, prefer it over the
     generic masked-diffusion loop implemented here and read confidences from the
     model's own per-token probabilities/entropy.

4. ``generate()`` KWARGS (``RealEngine._run_generate`` -- the fallback)
   - Which sampler kwargs ``generate`` accepts (``max_denoising_steps``,
     ``temperature``, ``entropy_bound``, ``top_k`` ...).  Unknown kwargs are
     dropped automatically via a TypeError retry.

If step capture is unavailable, the fallback synthesises a plausible per-step
reveal from the final text and flags it with ``approx=True`` in the ``start``
message (and logs that the reveal is reconstructed).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger("diffusiongemma.engine")


class _StepCaptureUnavailable(RuntimeError):
    """Raised during setup when the custom step loop cannot run.

    Raised *before* any frame is emitted so the caller can cleanly fall back to
    the documented ``generate()`` path without producing duplicate frames.
    """


class RealEngine:
    """Drives DiffusionGemma and streams per-step canvas states.

    The heavy imports (``torch`` / ``transformers``) happen in :meth:`__init__`,
    i.e. only when a ``RealEngine`` is actually constructed (real mode).  If
    construction fails, the server may fall back to the mock engine.
    """

    mode = "real"

    def __init__(self, config: Any) -> None:
        """Load the processor and model.

        ``config`` is an :class:`app.config.AppConfig`.  Raises on any load
        failure so the server can decide whether to fall back to mock.
        """
        # --- ADAPTATION POINT 1: import names & loading -------------------
        import torch  # noqa: F401  (kept local; validates torch is importable)
        from transformers import (  # type: ignore[import-not-found]
            AutoProcessor,
            AutoTokenizer,
            DiffusionGemmaForBlockDiffusion,
        )

        self.config = config
        self.model_id: str = config.model_id
        self.precision: str = config.precision
        self.mask_glyph: str = getattr(config, "mask_glyph", "░")
        self._torch = torch

        # DiffusionGemma ships a *multimodal* Gemma4Processor whose image path
        # imports optional, torch-version-coupled deps (torchvision, PIL). This is
        # a TEXT-ONLY demo, so we only need tokenization + the chat template -- both
        # carried by the plain tokenizer. Try the full processor first; on ANY
        # failure (e.g. "No module named 'torchvision'") fall back to AutoTokenizer
        # so real mode works without the multimodal stack. Install it explicitly
        # (uv sync --extra gpu --extra multimodal) if you want the full processor.
        logger.info("Loading tokenizer/processor for %s ...", self.model_id)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            logger.info("Loaded full processor: %s", type(self.processor).__name__)
        except Exception as proc_exc:
            logger.warning(
                "AutoProcessor unavailable (%s). Falling back to AutoTokenizer "
                "(text-only path). For the full multimodal processor, install: "
                "uv sync --extra gpu --extra multimodal",
                proc_exc,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Keep a single handle: downstream code uses self.processor for both
            # apply_chat_template and decode, which the tokenizer also provides.
            self.processor = self.tokenizer
            logger.info("Loaded tokenizer: %s", type(self.tokenizer).__name__)

        load_kwargs: Dict[str, Any] = {"device_map": config.device or "auto"}
        if self.precision == "fp8":
            # ADAPTATION POINT 1b: fp8 path is a best guess. Prefer an explicit
            # quantization config if the build supports it; otherwise try an fp8
            # dtype string; otherwise degrade to bf16 with a warning.
            try:
                from transformers import (  # type: ignore[import-not-found]
                    FineGrainedFP8Config,
                )

                load_kwargs["quantization_config"] = FineGrainedFP8Config()
                logger.info("fp8 requested: using FineGrainedFP8Config().")
            except Exception:  # pragma: no cover - depends on transformers build
                load_kwargs["dtype"] = "float8_e4m3fn"
                logger.warning(
                    "fp8 requested but FineGrainedFP8Config unavailable; "
                    "trying dtype='float8_e4m3fn' (verify on H100)."
                )
        else:
            load_kwargs["dtype"] = torch.bfloat16  # bf16 default (~52GB on H100)

        logger.info("Loading model %s with %s ...", self.model_id, load_kwargs)
        try:
            self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                self.model_id, **load_kwargs
            )
        except Exception:
            if self.precision == "fp8":
                # Retry once in bf16 so an fp8 misconfiguration is recoverable.
                logger.exception("fp8 load failed; retrying in bf16.")
                load_kwargs.pop("quantization_config", None)
                load_kwargs["dtype"] = torch.bfloat16
                self.precision = "bf16"
                self.model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                    self.model_id, **load_kwargs
                )
            else:
                raise

        self.model.eval()
        self.device = str(getattr(self.model, "device", config.device or "cpu"))
        self._mask_id = self._detect_mask_id()
        logger.info(
            "Real engine ready: device=%s precision=%s mask_id=%s",
            self.device,
            self.precision,
            self._mask_id,
        )

    # ------------------------------------------------------------------
    # Detection / decoding helpers (ADAPTATION POINT 2)
    # ------------------------------------------------------------------

    def _detect_mask_id(self) -> Optional[int]:
        """Best-effort detection of the placeholder/MASK token id."""
        if self.config.mask_token_id is not None:
            return int(self.config.mask_token_id)

        tok = self.tokenizer
        for attr in ("mask_token_id",):
            value = getattr(tok, attr, None)
            if isinstance(value, int):
                return value

        cfg = getattr(self.model, "config", None)
        for attr in (
            "mask_token_id",
            "diffusion_mask_token_id",
            "placeholder_token_id",
            "pad_token_id",  # last-resort: some diffusion LMs reuse pad as mask
        ):
            value = getattr(cfg, attr, None)
            if isinstance(value, int):
                if attr == "pad_token_id":
                    logger.warning("Falling back to pad_token_id as MASK id.")
                return value

        for glyph in ("<mask>", "[MASK]", "<|mask|>", "<unused0>"):
            try:
                ids = tok.convert_tokens_to_ids(glyph)
            except Exception:
                continue
            unk = getattr(tok, "unk_token_id", None)
            if isinstance(ids, int) and ids >= 0 and ids != unk:
                return ids

        logger.warning("Could not detect a MASK token id; custom loop disabled.")
        return None

    def _decode_ids(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of ids to text (skipping special tokens by default)."""
        if not ids:
            return ""
        try:
            return self.processor.decode(ids, skip_special_tokens=skip_special)
        except Exception:
            try:
                return self.tokenizer.decode(ids, skip_special_tokens=skip_special)
            except Exception:  # pragma: no cover - defensive
                return ""

    def _decode_one(self, token_id: int) -> str:
        """Decode a single token id to its display text (MASK -> glyph)."""
        if self._mask_id is not None and token_id == self._mask_id:
            return self.mask_glyph
        try:
            return self.processor.decode([token_id], skip_special_tokens=False)
        except Exception:
            try:
                return self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:  # pragma: no cover - defensive
                return self.mask_glyph

    # ------------------------------------------------------------------
    # Input building / forward (ADAPTATION POINT 3)
    # ------------------------------------------------------------------

    def _build_inputs(self, prompt: str) -> Dict[str, Any]:
        """Apply the chat template and move tensors to the model device."""
        message = [{"role": "user", "content": prompt}]
        inputs = self.processor.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _probe_forward(self, inputs: Dict[str, Any]) -> bool:
        """Return True iff a forward pass yields usable ``[B, T, V]`` logits."""
        torch = self._torch
        try:
            with torch.no_grad():
                out = self.model(input_ids=inputs["input_ids"])
            logits = getattr(out, "logits", None)
            return logits is not None and logits.dim() == 3
        except Exception:
            logger.exception("Probe forward failed.")
            return False

    def _probe_canvas_alignment(self, inputs: Dict[str, Any], clen: int) -> bool:
        """Verify a forward yields exactly ``clen`` aligned canvas logit rows.

        Runs the same shape of forward the denoise loop uses (prefix + a block of
        ``clen`` MASK tokens) *before* any frame is streamed, so a shape/alignment
        mismatch triggers a clean fallback to ``generate()`` instead of failing
        mid-stream (which is what caused the earlier IndexError).
        """
        torch = self._torch
        try:
            prefix_ids = inputs["input_ids"]
            prefix_len = prefix_ids.shape[1]
            canvas_t = torch.full(
                (1, clen),
                int(self._mask_id),
                dtype=prefix_ids.dtype,
                device=prefix_ids.device,
            )
            full = torch.cat([prefix_ids, canvas_t], dim=1)
            canvas_logits = self._forward_canvas_logits(full, prefix_len, clen)
            return canvas_logits.dim() == 2 and canvas_logits.shape[0] == clen
        except Exception as exc:
            logger.warning("Canvas alignment probe failed: %s", exc)
            return False

    def _forward_canvas_logits(self, full_ids: Any, prefix_len: int, clen: int) -> Any:
        """Forward ``full_ids`` and return ``[clen, vocab]`` logits for the canvas.

        The canvas is the LAST block appended to ``full_ids`` (prefix + committed +
        canvas), so its per-position distributions are the **last ``clen`` rows** of
        the model output.  Taking the tail is robust whether the model returns
        logits for the whole sequence *or* only the canvas region -- DiffusionGemma
        returns roughly the canvas region, which is why the previous fixed
        ``[prefix_len : prefix_len + clen]`` slice came up short and raised an
        IndexError downstream.

        ADAPTATION POINT 3: this assumes *bidirectional* denoising (canvas position
        ``i`` is scored at its own position).  If this model is causal here, shift
        by one: return ``logits[0, seq - clen - 1 : seq - 1, :]``.  ``prefix_len``
        is kept in the signature for that alternative mapping.
        """
        torch = self._torch
        with torch.no_grad():
            out = self.model(input_ids=full_ids)
        logits = getattr(out, "logits", None)
        if logits is None or logits.dim() != 3:
            raise _StepCaptureUnavailable(
                f"forward did not return 3-D logits (got {type(out).__name__})"
            )
        seq = logits.shape[1]
        if seq < clen:
            raise _StepCaptureUnavailable(
                f"forward returned {seq} logit positions for a {clen}-token canvas"
            )
        return logits[0, seq - clen : seq, :]

    # ------------------------------------------------------------------
    # Public streaming interface
    # ------------------------------------------------------------------

    async def stream_generate(
        self,
        prompt: str,
        params: Dict[str, Any],
        cancel: "asyncio.Event",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream denoising frames for ``prompt``.

        The reveal strategy is chosen by ``config.reveal_mode``:

        * ``"generate"`` (default): use the documented ``model.generate()`` for
          correct text, then reconstruct the garbage->text reveal from the real
          output. Reliable -- it relies only on the official API.
        * ``"custom"``: the experimental true per-denoising-step capture loop. It
          requires the model's exact internal denoising semantics (ADAPTATION
          POINTS); if those don't match it can yield empty/garbled output, so it
          is opt-in. Falls back to ``generate()`` if it cannot even start.
        * ``"auto"``: try ``"custom"``, fall back to ``"generate"`` on failure.
        """
        start_t = time.perf_counter()
        loop = asyncio.get_event_loop()
        mode = getattr(self.config, "reveal_mode", "generate")

        try:
            inputs = await loop.run_in_executor(None, self._build_inputs, prompt)
        except Exception as exc:
            logger.exception("Failed to build inputs.")
            yield {"type": "error", "message": f"input preparation failed: {exc}"}
            return

        # Strategy (a): experimental custom step-by-step denoising loop.
        if mode in ("custom", "auto"):
            try:
                async for frame in self._stream_custom(prompt, params, cancel, inputs, start_t):
                    yield frame
                return
            except _StepCaptureUnavailable as exc:
                logger.warning(
                    "Per-step capture unavailable (%s); using generate() instead.",
                    exc,
                )
            except Exception:  # pragma: no cover - any unexpected setup failure
                logger.exception(
                    "Custom denoising loop failed before streaming; using generate()."
                )

        # Strategy (b): documented generate() + reconstructed reveal (the default).
        try:
            async for frame in self._stream_fallback(prompt, params, cancel, inputs, start_t):
                yield frame
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("generate() path failed.")
            yield {"type": "error", "message": f"generation failed: {exc}"}

    # ------------------------------------------------------------------
    # Strategy (a): custom denoising loop
    # ------------------------------------------------------------------

    async def _stream_custom(
        self,
        prompt: str,
        params: Dict[str, Any],
        cancel: "asyncio.Event",
        inputs: Dict[str, Any],
        start_t: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Drive the model one denoising step at a time, reading intermediate ids.

        Implements a *generic* masked-diffusion loop: append a block of MASK
        tokens, forward, commit the lowest-entropy masked positions, repeat.  All
        risky setup happens before the first ``yield`` so failures raise
        :class:`_StepCaptureUnavailable` and the caller can fall back cleanly.
        """
        loop = asyncio.get_event_loop()

        if self._mask_id is None:
            raise _StepCaptureUnavailable("no MASK token id detected")

        probe_ok = await loop.run_in_executor(None, self._probe_forward, inputs)
        if not probe_ok:
            raise _StepCaptureUnavailable("model.forward did not return [B, T, V] logits")

        canvas_len = int(params["canvas_length"])
        max_steps = int(params["max_denoising_steps"])
        max_new = int(params["max_new_tokens"])
        n_canvases = max(1, math.ceil(max_new / max(1, canvas_len)))

        # Validate canvas<->logits alignment BEFORE emitting `start`, so a shape
        # mismatch falls back to generate() cleanly instead of crashing mid-stream.
        first_clen = min(canvas_len, max_new)
        align_ok = await loop.run_in_executor(
            None, self._probe_canvas_alignment, inputs, first_clen
        )
        if not align_ok:
            raise _StepCaptureUnavailable(
                "could not align model logits to the canvas length"
            )

        seed = params.get("seed")
        if seed is not None:
            self._torch.manual_seed(int(seed))

        # Emit start only after setup succeeded.
        yield {
            "type": "start",
            "mode": "real",
            "canvas_length": canvas_len,
            "max_denoising_steps": max_steps,
            "prompt": prompt,
        }

        committed_ids: List[int] = []  # canvas ids committed across all canvases
        prefix_text = ""
        steps_total = 0
        total_tokens = 0
        committed_canvases = 0

        for ci in range(n_canvases):
            remaining = max_new - total_tokens
            if remaining <= 0:
                break
            clen = min(canvas_len, remaining)

            last_canvas_text: Optional[str] = None
            try:
                async for frame, canvas_text in self._denoise_canvas(
                    inputs, committed_ids, clen, params, ci, prefix_text, cancel
                ):
                    if frame is None:
                        # Sentinel carrying the final text for *this* block.
                        last_canvas_text = canvas_text
                        prefix_text += canvas_text
                        total_tokens += clen
                        committed_canvases += 1
                        break
                    steps_total += 1
                    yield frame
                    if cancel.is_set():
                        logger.info("Real generation cancelled during canvas %d", ci)
                        break
            except Exception:
                # Mid-stream internal failure: log and finalise with what we have
                # rather than crashing the UI (we have already emitted ``start``).
                logger.exception("Denoising canvas %d failed; finalising early.", ci)
                break

            if cancel.is_set() or last_canvas_text is None:
                break

            yield {
                "type": "canvas_committed",
                "canvas_index": ci,
                "text": last_canvas_text,
            }
            await asyncio.sleep(0)

        gen_seconds = max(1e-6, time.perf_counter() - start_t)
        final_text = self._decode_ids(committed_ids)
        # ``steps_total`` here is a count of REAL per-denoising-step forward passes,
        # so the AR estimate is on its firmest footing in this (custom) path.
        from app.mock_engine import estimate_ar_tokens_per_second

        yield {
            "type": "done",
            "text": final_text,
            "stats": {
                "canvases": committed_canvases,
                "steps_total": steps_total,
                "tokens": total_tokens,
                "gen_seconds": round(gen_seconds, 4),
                "tokens_per_second": round(total_tokens / gen_seconds, 2),
                "ar_tokens_per_second": estimate_ar_tokens_per_second(
                    total_tokens, steps_total, gen_seconds
                ),
            },
        }

    async def _denoise_canvas(
        self,
        inputs: Dict[str, Any],
        committed_ids: List[int],
        clen: int,
        params: Dict[str, Any],
        canvas_index: int,
        prefix_text: str,
        cancel: "asyncio.Event",
    ):
        """Yield ``(step_frame, None)`` per step, then ``(None, canvas_text)``.

        Runs each forward pass in a thread executor so the event loop stays
        responsive and ``cancel`` can be observed between steps.
        """
        torch = self._torch
        loop = asyncio.get_event_loop()

        prefix_ids = inputs["input_ids"]  # [1, L_prompt]
        prefix_len = prefix_ids.shape[1] + len(committed_ids)

        # Canvas state: all MASK to start.
        canvas = [self._mask_id] * clen
        committed_mask = [False] * clen
        prev_display: List[Optional[str]] = [None] * clen

        max_steps = int(params["max_denoising_steps"])
        t_start = float(params["temperature_start"])
        t_end = float(params["temperature_end"])
        top_k = int(params["top_k"])

        device = self.model.device

        def _build_full_ids() -> Any:
            committed_t = torch.tensor([committed_ids], dtype=prefix_ids.dtype, device=device)
            canvas_t = torch.tensor([canvas], dtype=prefix_ids.dtype, device=device)
            return torch.cat([prefix_ids, committed_t, canvas_t], dim=1)

        for s in range(max_steps):
            if cancel.is_set():
                break

            full_ids = _build_full_ids()
            canvas_logits = await loop.run_in_executor(
                None, self._forward_canvas_logits, full_ids, prefix_len, clen
            )

            # Temperature schedule.
            if max_steps > 1:
                temp = t_start + (t_end - t_start) * (s / (max_steps - 1))
            else:
                temp = t_start
            temp = max(1e-4, temp)

            logits = canvas_logits.float() / temp
            if top_k and top_k > 0:
                logits = self._apply_top_k(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = -(probs * log_probs).sum(dim=-1)  # [clen]
            max_prob, argmax = probs.max(dim=-1)  # [clen]

            # Choose uncommitted positions to commit this step: lowest entropy
            # first; always commit those under the (scaled) entropy bound, but at
            # least ``per_pass_target`` to guarantee progress.
            entropy_list = entropy.tolist()
            if len(entropy_list) < clen:
                # Defensive: alignment was probed up front, but never index past
                # the rows we actually got. Finalise this canvas via force-commit.
                logger.warning(
                    "Canvas logits returned %d rows for a %d-token canvas; "
                    "finalising this canvas early.",
                    len(entropy_list),
                    clen,
                )
                break
            uncommitted = [i for i in range(clen) if not committed_mask[i]]
            if not uncommitted:
                # Everything already committed: stop stepping and finalise below.
                break

            uncommitted.sort(key=lambda i: entropy_list[i])  # most confident first
            # Commit a LIMITED number of the most-confident positions per step so
            # the canvas resolves GRADUALLY over ~max_steps. A single all-mask
            # forward is often confident everywhere, so an uncapped (entropy-bound)
            # commit would resolve the whole block in step 0 -- nothing to watch.
            # Linear schedule: cumulative target committed by the end of step s.
            if s >= max_steps - 1:
                target_total = clen
            else:
                target_total = int(round(clen * (s + 1) / max_steps))
            already = clen - len(uncommitted)
            to_commit = max(1, target_total - already)
            commit_now = uncommitted[:to_commit]

            newly_positions: List[int] = []
            for i in commit_now:
                if self._mask_id is not None and not committed_mask[i]:
                    canvas[i] = int(argmax[i].item())
                    committed_mask[i] = True
                    newly_positions.append(i)
            newly_positions.sort()

            # Build the full canvas rendering for this step.
            tokens: List[Dict[str, Any]] = []
            for i in range(clen):
                if committed_mask[i]:
                    text = self._decode_one(canvas[i])
                    conf = float(max_prob[i].item())
                    state = "committed"
                else:
                    # Tentative preview: the current argmax guess + its prob.
                    conf = float(max_prob[i].item())
                    if conf >= 0.15:
                        text = self._decode_one(int(argmax[i].item()))
                        state = "tentative"
                    else:
                        text = self.mask_glyph
                        state = "mask"
                changed = text != prev_display[i]
                prev_display[i] = text
                tokens.append(
                    {
                        "text": text,
                        "state": state,
                        "confidence": round(max(0.0, min(1.0, conf)), 4),
                        "changed": changed,
                    }
                )

            committed_canvas_ids = [canvas[i] for i in range(clen) if committed_mask[i]]
            committed_text = prefix_text + self._decode_ids(committed_canvas_ids)
            newly_committed = [self._decode_one(canvas[i]) for i in newly_positions]

            yield (
                {
                    "type": "step",
                    "canvas_index": canvas_index,
                    "step": s,
                    "total_steps": max_steps,
                    "tokens": tokens,
                    "committed_text": committed_text,
                    "newly_committed": newly_committed,
                },
                None,
            )
            await asyncio.sleep(0)

            if all(committed_mask):
                break

        # Force-commit any stragglers (e.g. on cancel/loop exhaustion).
        if not all(committed_mask):
            # One final forward to fill remaining positions deterministically.
            try:
                full_ids = _build_full_ids()
                canvas_logits = await loop.run_in_executor(
                    None, self._forward_canvas_logits, full_ids, prefix_len, clen
                )
                argmax = canvas_logits.float().argmax(dim=-1)
                m = int(argmax.shape[0])
                for i in range(clen):
                    if not committed_mask[i]:
                        # Only use argmax where we actually have a row; otherwise
                        # leave the existing MASK id (rendered as a glyph).
                        if i < m:
                            canvas[i] = int(argmax[i].item())
                        committed_mask[i] = True
            except Exception:  # pragma: no cover - defensive
                logger.exception("Final fill forward failed; leaving MASKs decoded.")

        committed_ids.extend(canvas)
        yield None, self._decode_ids(canvas)

    @staticmethod
    def _apply_top_k(logits: Any, top_k: int) -> Any:
        """Mask all but the ``top_k`` highest logits per position (in-place safe)."""
        import torch

        k = min(top_k, logits.shape[-1])
        if k <= 0:
            return logits
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < kth, float("-inf"))

    # ------------------------------------------------------------------
    # Strategy (b): documented generate() + synthesised per-step reveal
    # ------------------------------------------------------------------

    def _to_flat_ids(self, obj: Any) -> List[int]:
        """Reduce a generate() output to a flat 1-D list of token ids (batch 0).

        Robust to the several shapes ``generate`` can return: a ``[batch, seq]``
        tensor, a higher-rank ``[batch, blocks, canvas]`` tensor (block diffusion),
        a ``GenerateOutput`` with ``.sequences``, or nested lists.
        """
        torch = self._torch
        if hasattr(obj, "sequences"):  # GenerateOutput-like
            obj = obj.sequences
        if isinstance(obj, torch.Tensor):
            t = obj[0] if obj.dim() >= 2 else obj  # drop the batch dim
            return [int(x) for x in t.reshape(-1).tolist()]
        # Fallback: recursively flatten lists/tuples/tensors of batch element 0.
        flat: List[int] = []

        def _rec(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                flat.extend(int(y) for y in x.reshape(-1).tolist())
            elif isinstance(x, (list, tuple)):
                for y in x:
                    _rec(y)
            else:
                flat.append(int(x))

        try:
            first = obj[0]
        except Exception:  # pragma: no cover - defensive
            first = obj
        _rec(first)
        return flat

    def _run_generate(self, inputs: Dict[str, Any], params: Dict[str, Any]) -> List[int]:
        """Call ``model.generate`` (documented usage) and return the new token ids."""
        torch = self._torch
        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        max_new = int(params["max_new_tokens"])
        # Mirror the DOCUMENTED call exactly: just ``max_new_tokens``. Passing the
        # speculative diffusion sampler kwargs (max_denoising_steps/entropy_bound/
        # etc.) was unconfirmed and on this build changed the output tensor shape,
        # so we don't. Reveal pacing is handled client-side anyway.
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new)

        seq = getattr(out, "sequences", out)
        flat_ids = self._to_flat_ids(out)
        input_list = self._to_flat_ids(inputs["input_ids"])
        input_len = len(input_list)

        # ``generate()`` may return ``[prompt + response]`` OR only ``[response]``.
        # Slice off the prompt ONLY when the output actually starts with it.
        if len(flat_ids) > input_len and flat_ids[:input_len] == input_list:
            new_ids = flat_ids[input_len:]
        else:
            new_ids = flat_ids
        try:
            shape: Any = tuple(seq.shape)
        except Exception:  # pragma: no cover - defensive
            shape = type(seq).__name__
        logger.info(
            "generate(): out shape=%s -> %d ids (input %d) -> %d new",
            shape,
            len(flat_ids),
            input_len,
            len(new_ids),
        )
        return new_ids

    # ------------------------------------------------------------------
    # Debug helper (GET /api/debug/generate)
    # ------------------------------------------------------------------

    def debug_generate(self, prompt: str, max_new_tokens: int = 64) -> Dict[str, Any]:
        """Run the real model and return a rich diagnostic of what it produces.

        Surfaces the raw ``generate()`` output shape, token ids, every decode
        variant, the tokenizer's special-token ids, and what an all-MASK forward
        predicts -- everything needed to see why text might be missing/garbled.
        Blocking; call via ``run_in_executor``.
        """
        torch = self._torch
        info: Dict[str, Any] = {"mode": "real", "prompt": prompt}

        def _safe_decode(ids: Any, skip: bool) -> str:
            try:
                return self._decode_ids(ids, skip_special=skip)
            except Exception as exc:  # pragma: no cover - defensive
                return f"<decode error: {exc}>"

        # Tokenizer / special tokens (a common cause of empty decodes).
        tok = self.tokenizer
        info["tokenizer_class"] = type(tok).__name__
        info["processor_class"] = type(self.processor).__name__
        for attr in (
            "mask_token_id",
            "pad_token_id",
            "eos_token_id",
            "bos_token_id",
            "unk_token_id",
        ):
            info[attr] = getattr(tok, attr, None)
        info["detected_mask_id"] = self._mask_id

        try:
            inputs = self._build_inputs(prompt)
        except Exception as exc:
            info["error"] = f"build_inputs failed: {exc}"
            return info

        # --- model.generate() ground truth ---
        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
            seq = getattr(out, "sequences", out)
            info["generate_raw_type"] = type(out).__name__
            try:
                info["generate_raw_shape"] = list(seq.shape)
            except Exception:
                info["generate_raw_shape"] = None
            flat = self._to_flat_ids(out)
            inp = self._to_flat_ids(inputs["input_ids"])
            starts = len(flat) > len(inp) and flat[: len(inp)] == inp
            new_ids = flat[len(inp):] if starts else flat
            info["input_len"] = len(inp)
            info["generate_flat_len"] = len(flat)
            info["output_starts_with_prompt"] = starts
            info["new_len"] = len(new_ids)
            info["new_ids_head"] = new_ids[:48]
            info["new_ids_tail"] = new_ids[-16:]
            info["decode_new_skip_special"] = _safe_decode(new_ids, True)[:600]
            info["decode_new_keep_special"] = _safe_decode(new_ids, False)[:600]
            info["decode_full_keep_special"] = _safe_decode(flat, False)[:600]
            info["sample_pieces"] = [
                {"id": int(i), "piece": _safe_decode([int(i)], False)}
                for i in new_ids[:12]
            ]
        except Exception as exc:
            info["generate_error"] = f"{type(exc).__name__}: {exc}"

        # --- custom-loop probe: what does an all-MASK forward predict? ---
        try:
            clen = 16
            prefix_ids = inputs["input_ids"]
            prefix_len = prefix_ids.shape[1]
            mask_id = self._mask_id if self._mask_id is not None else 0
            canvas_t = torch.full(
                (1, clen), int(mask_id), dtype=prefix_ids.dtype, device=prefix_ids.device
            )
            full = torch.cat([prefix_ids, canvas_t], dim=1)
            with torch.no_grad():
                o = self.model(input_ids=full)
            lg = getattr(o, "logits", None)
            info["custom_forward_logits_shape"] = (
                list(lg.shape) if lg is not None else None
            )
            if lg is not None and lg.dim() == 3:
                seqd = lg.shape[1]
                k = min(clen, seqd)
                canvas_lg = lg[0, seqd - k : seqd, :]
                am = [int(x) for x in canvas_lg.argmax(dim=-1).tolist()]
                info["custom_argmax_ids"] = am
                info["custom_argmax_decoded"] = _safe_decode(am, False)
        except Exception as exc:
            info["custom_forward_error"] = f"{type(exc).__name__}: {exc}"

        return info

    async def _stream_fallback(
        self,
        prompt: str,
        params: Dict[str, Any],
        cancel: "asyncio.Event",
        inputs: Dict[str, Any],
        start_t: float,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate the final text, then synthesise a per-step reveal (approx)."""
        import random

        # Pure-Python reveal helpers reused from the mock engine (no heavy deps).
        from app.mock_engine import (
            estimate_ar_tokens_per_second,
            simulate_canvas_steps,
            tokenize_text,
        )

        loop = asyncio.get_event_loop()
        # Time ONLY the model.generate() call -- this is the true generation cost.
        # (The reconstructed reveal below is pure-Python and streamed with
        # sleep(0); it, and the client's playback speed, must NOT affect tokens/s.)
        gen_t0 = time.perf_counter()
        new_ids = await loop.run_in_executor(None, self._run_generate, inputs, params)
        generate_seconds = max(1e-6, time.perf_counter() - gen_t0)
        model_tokens = len(new_ids)
        final_text = self._decode_ids(new_ids)
        if not final_text.strip():
            # Clean decode stripped everything (e.g. all special tokens). Retry
            # keeping specials so the user sees the real output rather than blank.
            logger.warning(
                "Decoded text empty with skip_special_tokens=True; retrying without."
            )
            final_text = self._decode_ids(new_ids, skip_special=False)
        logger.info(
            "generate() reveal: %d tokens, %d chars, preview=%r",
            len(new_ids),
            len(final_text),
            final_text[:120],
        )

        all_tokens = tokenize_text(final_text)
        max_new = int(params["max_new_tokens"])
        if max_new > 0:
            all_tokens = all_tokens[:max_new]

        canvas_len = max(1, int(params["canvas_length"]))
        canvases = [
            all_tokens[i : i + canvas_len] for i in range(0, len(all_tokens), canvas_len)
        ] or [[]]

        rng = random.Random(params.get("seed"))

        # ``approx=True`` marks the reveal as reconstructed (see schema/contract).
        yield {
            "type": "start",
            "mode": "real",
            "canvas_length": canvas_len,
            "max_denoising_steps": int(params["max_denoising_steps"]),
            "prompt": prompt,
            "approx": True,
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
                    logger.info("Fallback generation cancelled during canvas %d", ci)
                    return
                steps_total += 1
                yield frame
                await asyncio.sleep(0)

            canvas_text = "".join(canvas_tokens)
            prefix_text += canvas_text
            total_tokens += len(canvas_tokens)
            committed_canvases += 1
            yield {"type": "canvas_committed", "canvas_index": ci, "text": canvas_text}
            await asyncio.sleep(0)

        # Report the REAL model throughput: actual generated token count over the
        # pure generate() wall-clock. ``total_tokens`` (word-pieces revealed) and
        # ``steps_total`` describe only the cosmetic reveal, not the model's work.
        yield {
            "type": "done",
            "text": prefix_text,
            "stats": {
                "canvases": committed_canvases,
                "steps_total": steps_total,
                "tokens": model_tokens,
                "gen_seconds": round(generate_seconds, 4),
                "tokens_per_second": round(model_tokens / generate_seconds, 2),
                # AR baseline is an ESTIMATE built on ``steps_total`` -- which in
                # this fallback path is the *reconstructed* reveal's step count,
                # not a captured count of real denoising passes (the start frame
                # is flagged approx). It illustrates the diffusion-vs-AR gap.
                "ar_tokens_per_second": estimate_ar_tokens_per_second(
                    model_tokens, steps_total, generate_seconds
                ),
            },
        }
