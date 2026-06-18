/* =====================================================================
   playground.js — the real DiffusionGemma playground
   ---------------------------------------------------------------------
   Pieces:
     1. CONTROLS  — built from /api/config defaults+bounds so they stay in
                    sync with the backend (sliders + number inputs).
     2. WS CLIENT — opens ws(s)://<host>/ws/generate, sends the START
                    message, pushes EVERY incoming frame into an in-memory
                    recording. Cancel sends {"type":"cancel"}.
     3. PLAYBACK  — a frame buffer + a playback loop that renders frames at
                    a user-controlled rate, fully DECOUPLED from arrival
                    speed (so it works whether generation is slow or
                    1000+ tok/s). Buffer-then-play vs live.
     4. VISUALIZE — a monospace canvas where MASK tokens flicker, committed
                    tokens are crisp, tentative/low-confidence tokens are
                    dim/blue-tinted, resolving smoothly in place.
   The display/playback controls are CLIENT-ONLY and never touch generation.
   ===================================================================== */

const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Order + plain-language metadata for every generation parameter.
   Bounds/defaults come from /api/config; this only adds labels/hints. */
const PARAM_GROUPS = [
  { label: "Canvas & length", params: ["max_new_tokens", "canvas_length", "max_denoising_steps"] },
  { label: "EB sampler", params: ["temperature_start", "temperature_end", "entropy_bound", "adaptive_stop_threshold", "top_k"] },
];
const PARAM_META = {
  max_new_tokens: { label: "Max new tokens", hint: "Total tokens to generate, spread across canvases.", int: true },
  canvas_length: { label: "Canvas length", hint: "Block size — tokens denoised together per canvas.", int: true },
  max_denoising_steps: { label: "Max denoising steps", hint: "Ceiling on refinement passes per canvas.", int: true },
  temperature_start: { label: "Temperature · start", hint: "Exploratory sampling temperature on early steps." },
  temperature_end: { label: "Temperature · end", hint: "Decisive temperature on late steps (decays from start)." },
  entropy_bound: { label: "Entropy bound", hint: "Confidence gate — only tokens below this entropy commit." },
  adaptive_stop_threshold: { label: "Adaptive stop", hint: "Stop a canvas early once remaining entropy drops below this." },
  top_k: { label: "Top-k", hint: "Restrict each pick to the k likeliest tokens. 0 = off.", int: true },
};
const NUMERIC_PARAMS = ["max_new_tokens", "canvas_length", "max_denoising_steps", "temperature_start", "temperature_end", "entropy_bound", "adaptive_stop_threshold", "top_k"];

const PRESETS = [
  { label: "Why is the sky blue?", text: "Why is the sky blue?" },
  { label: "Fibonacci (code)", text: "Write a Python function that returns the nth Fibonacci number. Include a docstring and a couple of examples." },
  { label: "Markdown table", text: "Create a markdown table comparing autoregressive and diffusion text generation across speed, revision, and infilling." },
  { label: "Infilling", text: "Fill in the blanks: The capital of France is ____, a city on the river ____ famous for the ____." },
];

const GLYPHS = {
  shades: "░▒▓",
  blocks: "█▉▊▋",
  dots: "·•∙",
  braille: "⠁⠂⠄⡀⢀⠠⠐⠈",
  ascii: "#%@&$?*",
};

/* ----------------------------- pure helpers ----------------------------- */
function glyphRun(style, len = 2) {
  const set = GLYPHS[style] || GLYPHS.shades;
  let s = "";
  for (let i = 0; i < len; i++) s += set[(Math.random() * set.length) | 0];
  return s;
}
function decimalsOf(step) {
  const s = String(step);
  const i = s.indexOf(".");
  return i < 0 ? 0 : s.length - i - 1;
}
function fmtVal(v, dec) {
  return Number(v).toFixed(dec);
}
function lerpColor(a, b, t) {
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r}, ${g}, ${bl})`;
}

/* ======================================================================= */
export function initPlayground(root, ctx) {
  if (!root) return;
  const { el, clamp, fmtNum, config } = ctx;

  const dd = config.display_defaults || {};
  const SPEED_BOUNDS = (config.bounds && config.bounds.playback_speed_ms) || { min: 0, max: 600, step: 10 };

  /* ----------------------------- state ----------------------------- */
  const display = {
    playback_speed_ms: dd.playback_speed_ms ?? 80,
    mode: dd.mode ?? "buffer-then-play",
    show_mask_glyphs: dd.show_mask_glyphs ?? true,
    color_by_confidence: dd.color_by_confidence ?? true,
    mask_glyph_style: dd.mask_glyph_style ?? "shades",
  };

  const pg = {
    inputs: {}, // name -> { get, set }
    ws: null,
    generating: false,
    done: false,
    cancelled: false,
    recording: [], // every incoming "step" frame, in arrival order
    playIndex: 0,
    playing: false,
    playTimer: null,
    rafPending: false,
    autoFollow: display.mode === "live",
    viewMode: display.mode,
    mode: ctx.mode === "real" ? "real" : "mock",
    totalSteps: config.defaults.max_denoising_steps || 48,
    canvasLen: config.defaults.canvas_length || 256,
    finalText: null,
    stats: null,
    committedText: "",
    flickerTimer: null,
  };

  /* ============================ BUILD UI ============================ */

  /* -- a slider + number-input pair bound to a config param -- */
  function buildParam(name) {
    const b = (config.bounds && config.bounds[name]) || { min: 0, max: 100, step: 1 };
    const def = config.defaults[name];
    const meta = PARAM_META[name];
    const dec = decimalsOf(b.step);
    const id = "pg-" + name;

    const num = el("input", { type: "number", class: "ctl-num", id, min: b.min, max: b.max, step: b.step, value: fmtVal(def, dec) });
    const range = el("input", { type: "range", min: b.min, max: b.max, step: b.step, value: String(def), "aria-label": meta.label });

    const refreshFill = () => {
      const pct = b.max === b.min ? 0 : ((Number(range.value) - b.min) / (b.max - b.min)) * 100;
      range.style.setProperty("--fill", pct + "%");
    };
    range.addEventListener("input", () => {
      num.value = fmtVal(Number(range.value), dec);
      refreshFill();
    });
    num.addEventListener("input", () => {
      const v = Number(num.value);
      if (Number.isNaN(v)) return;
      range.value = String(clamp(v, b.min, b.max));
      refreshFill();
    });
    num.addEventListener("change", () => {
      let v = Number(num.value);
      if (Number.isNaN(v)) v = def;
      v = clamp(v, b.min, b.max);
      num.value = fmtVal(v, dec);
      range.value = String(v);
      refreshFill();
    });
    refreshFill();

    pg.inputs[name] = {
      get() {
        let v = Number(num.value);
        if (Number.isNaN(v)) v = def;
        v = clamp(v, b.min, b.max);
        return meta.int ? Math.round(v) : v;
      },
      set(v) {
        num.value = fmtVal(v, dec);
        range.value = String(v);
        refreshFill();
      },
    };

    return el(
      "div",
      { class: "field" },
      el("div", { class: "control-row" }, el("label", { class: "ctl-label", for: id }, meta.label), num),
      range,
      el("span", { class: "hint" }, meta.hint)
    );
  }

  /* -- prompt + presets -- */
  const promptEl = el("textarea", { id: "pg-prompt", rows: 3, placeholder: "Ask DiffusionGemma anything…" });
  promptEl.value = PRESETS[0].text;
  const presetRow = el(
    "div",
    { class: "presets" },
    ...PRESETS.map((p) => el("button", { type: "button", class: "preset", title: "Use this prompt", onClick: () => (promptEl.value = p.text) }, p.label))
  );

  /* -- seed -- */
  const seedInput = el("input", { type: "number", id: "pg-seed", placeholder: "random", step: 1, min: 0 });
  pg.inputs.seed = {
    get() {
      const raw = seedInput.value.trim();
      if (raw === "") return null;
      const v = parseInt(raw, 10);
      return Number.isNaN(v) ? null : v;
    },
    set(v) {
      seedInput.value = v == null ? "" : String(v);
    },
  };

  /* -- generate / cancel -- */
  const genBtn = el("button", { class: "btn btn-primary", type: "button", onClick: startGenerate }, "Generate");
  const cancelBtn = el("button", { class: "btn btn-danger", type: "button", disabled: true, onClick: cancelGenerate }, "Cancel");

  const controlPanel = el(
    "aside",
    { class: "panel", "aria-label": "Generation controls" },
    // Sticky top region: prompt + the primary action stay pinned and visible
    // while the advanced parameters scroll below — no scrolling to reach Generate.
    el(
      "div",
      { class: "panel-head pg-top" },
      el("h2", { class: "panel-title" }, "Generation"),
      el("div", { class: "field pg-prompt-field" }, el("label", { for: "pg-prompt" }, "Prompt"), promptEl, presetRow),
      el("div", { class: "btn-row" }, genBtn, cancelBtn)
    ),
    el(
      "div",
      { class: "panel-pad" },
      el("p", { class: "panel-sub pg-params-note" }, "Parameters — sent to the model, synced with /api/config."),
      ...PARAM_GROUPS.flatMap((grp) => [el("div", { class: "section-label" }, grp.label), ...grp.params.map(buildParam)]),
      el("div", { class: "section-label" }, "Reproducibility"),
      el("div", { class: "field" }, el("label", { for: "pg-seed" }, "Seed"), seedInput, el("span", { class: "hint" }, "Empty = random. Set a value for reproducible runs."))
    )
  );

  /* ----------------- center visualization ----------------- */
  const modeBadge = el("span", { class: "vh-mode", "data-mode": pg.mode }, pg.mode === "real" ? "REAL" : "MOCK");
  const headCanvas = el("span", { class: "vh-item" }, "canvas 0");
  const headStep = el("span", { class: "vh-item" }, "step 0 / " + pg.totalSteps);
  const slowBadge = el("span", { class: "badge-slow", hidden: true }, "Slowed down output");
  const liveBadge = el("span", { class: "badge-live", hidden: true }, "LIVE");
  const runStatus = el("span", { class: "vh-item" }, "idle");

  const vizText = el("div", { id: "viz-text" }, el("span", { class: "viz-empty" }, "Enter a prompt and press Generate — the canvas will resolve from noise here."));

  // results
  const statsGrid = el("div", { class: "stats-grid" });
  const finalText = el("pre", { class: "final-text" });
  const copyBtn = el("button", { class: "btn btn-ghost copy-btn", type: "button", onClick: copyFinal }, "Copy");
  const results = el(
    "div",
    { class: "results" },
    el("h3", {}, "Result"),
    statsGrid,
    el("div", { class: "final-wrap" }, copyBtn, finalText)
  );

  const vizCard = el(
    "div",
    { class: "viz-card" },
    el(
      "div",
      { class: "viz-head" },
      modeBadge,
      headCanvas,
      headStep,
      el("span", { class: "vh-spacer" }),
      slowBadge,
      liveBadge,
      runStatus
    ),
    el("div", { class: "viz-body" }, vizText),
    results
  );

  /* ----------------- playback & display panel ----------------- */
  // speed
  const speedRange = el("input", { type: "range", min: SPEED_BOUNDS.min, max: SPEED_BOUNDS.max, step: SPEED_BOUNDS.step, value: String(display.playback_speed_ms), "aria-label": "Playback speed (ms per frame)" });
  const speedReadout = el("span", { class: "hint" });
  const refreshSpeedFill = () => {
    const pct = SPEED_BOUNDS.max === SPEED_BOUNDS.min ? 0 : ((Number(speedRange.value) - SPEED_BOUNDS.min) / (SPEED_BOUNDS.max - SPEED_BOUNDS.min)) * 100;
    speedRange.style.setProperty("--fill", pct + "%");
  };
  function updateSpeedReadout() {
    const ms = Number(display.playback_speed_ms) || 0;
    speedReadout.textContent = ms <= 0 ? "0 ms/frame · as fast as possible" : `${ms} ms/frame · ~${Math.round(1000 / ms)} fps`;
  }
  speedRange.addEventListener("input", () => {
    display.playback_speed_ms = Number(speedRange.value);
    refreshSpeedFill();
    updateSpeedReadout();
  });
  refreshSpeedFill();
  updateSpeedReadout();

  // source mode (buffer vs live)
  const sourceSeg = buildSegmented(
    [
      { value: "buffer-then-play", label: "Buffer → play" },
      { value: "live", label: "Live" },
    ],
    display.mode,
    (v) => {
      display.mode = v;
      pg.viewMode = v;
      if (v === "live") {
        // start following the newest frame
        pg.autoFollow = true;
        if (pg.recording.length) {
          pg.playIndex = pg.recording.length - 1;
          renderFrame(pg.recording[pg.playIndex]);
          setScrub(pg.playIndex);
        }
      } else {
        pg.autoFollow = false;
      }
      updateBadges();
    }
  );

  // transport
  const btnBack = el("button", { class: "tbtn", type: "button", "aria-label": "Step back one frame", title: "Step back", onClick: stepBack }, "⏮");
  const btnPlay = el("button", { class: "tbtn primary", type: "button", "aria-label": "Play", title: "Play / pause", onClick: togglePlay }, "▶");
  const btnFwd = el("button", { class: "tbtn", type: "button", "aria-label": "Step forward one frame", title: "Step forward", onClick: stepForward }, "⏭");
  const btnRestart = el("button", { class: "tbtn", type: "button", "aria-label": "Restart playback", title: "Restart", onClick: restart }, "↻");
  const transport = el("div", { class: "transport" }, btnBack, btnPlay, btnFwd, btnRestart);

  // scrubber
  const scrub = el("input", { type: "range", min: 0, max: 0, step: 1, value: 0, "aria-label": "Scrub recorded frames" });
  scrub.addEventListener("input", () => scrubTo(Number(scrub.value)));
  const scrubReadout = el("div", { class: "scrub-readout" }, (pg._scrubL = el("span", {}, "frame 0 / 0")), (pg._scrubR = el("span", {}, "0 buffered")));

  // toggles
  const maskSwitch = buildSwitch("pg-mask", "Show mask glyphs", display.show_mask_glyphs, (v) => {
    display.show_mask_glyphs = v;
    rerenderCurrent();
  });
  const confSwitch = buildSwitch("pg-conf", "Color by confidence", display.color_by_confidence, (v) => {
    display.color_by_confidence = v;
    rerenderCurrent();
  });

  // glyph style
  const glyphSelect = el(
    "select",
    { id: "pg-glyph", onChange: (e) => ((display.mask_glyph_style = e.target.value), rerenderCurrent()) },
    ...["shades", "blocks", "dots", "braille", "ascii"].map((g) => el("option", { value: g, selected: g === display.mask_glyph_style }, g[0].toUpperCase() + g.slice(1)))
  );

  const playbackPanel = el(
    "aside",
    { class: "panel", "aria-label": "Playback and display controls" },
    el("div", { class: "panel-head" }, el("h2", { class: "panel-title" }, "Playback & display"), el("p", { class: "panel-sub" }, "View only — never changes generation.")),
    el(
      "div",
      { class: "panel-pad" },
      el("div", { class: "note-banner" }, el("strong", {}, "Decoupled from generation. "), "Every frame is recorded as it arrives and replayed here at your chosen speed — so 1000+ tok/s is still watchable."),
      el("div", { class: "field" }, el("label", { for: "pg-glyph" }, "Playback speed"), speedRange, speedReadout),
      el("div", { class: "field" }, el("label", {}, "Source"), sourceSeg.node),
      transport,
      el("div", { class: "scrub-wrap" }, scrub, scrubReadout),
      el("div", { class: "section-label" }, "Display"),
      maskSwitch,
      confSwitch,
      el("div", { class: "field", style: { marginTop: "10px" } }, el("label", { for: "pg-glyph" }, "Mask glyph style"), glyphSelect)
    )
  );

  /* assemble */
  root.replaceChildren(el("div", { class: "pg-layout" }, controlPanel, el("div", { class: "pg-center" }, vizCard), playbackPanel));

  updateTransport();
  startFlicker();

  /* ====================== small UI builders ====================== */
  function buildSwitch(id, label, checked, onChange) {
    const input = el("input", { type: "checkbox", id });
    input.checked = checked;
    input.addEventListener("change", () => onChange(input.checked));
    return el("div", { class: "toggle-row" }, el("label", { for: id }, label), el("label", { class: "switch" }, input, el("span", { class: "slider" })));
  }
  function buildSegmented(options, value, onChange) {
    let cur = value;
    const btns = options.map((o) =>
      el(
        "button",
        {
          type: "button",
          "aria-pressed": String(o.value === cur),
          onClick: () => {
            cur = o.value;
            btns.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === cur)));
            onChange(cur);
          },
        },
        o.label
      )
    );
    return {
      node: el("div", { class: "segmented", role: "group", "aria-label": "Playback source" }, ...btns),
      set(v) {
        cur = v;
        btns.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i].value === v)));
      },
    };
  }

  /* ====================== visualization ====================== */
  function applyToken(span, t) {
    const state = t.state;
    let cls = "tok";
    let text;

    span.style.color = "";
    span.style.opacity = "";

    if (state === "mask") {
      if (display.show_mask_glyphs) {
        cls += " tok-mask";
        text = glyphRun(display.mask_glyph_style);
      } else {
        text = " "; // keep a faint gap but no glyph
      }
    } else if (state === "committed") {
      cls += " tok-committed";
      text = t.text;
    } else {
      cls += " tok-tentative";
      text = t.text;
    }

    if (t.changed) cls += " tok-changed";
    span.className = cls;

    if (display.color_by_confidence && state !== "mask") {
      const c = clamp(typeof t.confidence === "number" ? t.confidence : 1, 0, 1);
      // low confidence -> accent blue, high -> near-white
      span.style.color = lerpColor([138, 180, 248], [232, 235, 245], c);
      if (state === "tentative") span.style.opacity = (0.42 + 0.55 * c).toFixed(2);
    }

    span.textContent = text;
  }

  function renderFrame(frame) {
    if (!frame) return;
    const toks = frame.tokens || [];
    if (vizText.childElementCount !== toks.length || vizText.querySelector(".viz-empty")) {
      const spans = new Array(toks.length);
      for (let i = 0; i < toks.length; i++) spans[i] = document.createElement("span");
      vizText.replaceChildren(...spans);
    }
    const spans = vizText.children;
    for (let i = 0; i < toks.length; i++) applyToken(spans[i], toks[i]);
    updateVizHead(frame);
  }

  function rerenderCurrent() {
    const f = pg.recording[pg.playIndex];
    if (f) renderFrame(f);
  }

  function updateVizHead(frame) {
    modeBadge.dataset.mode = pg.mode;
    modeBadge.textContent = pg.mode === "real" ? "REAL" : "MOCK";
    if (frame) {
      headCanvas.textContent = "canvas " + (frame.canvas_index ?? 0);
      headStep.textContent = "step " + (frame.step ?? 0) + " / " + (frame.total_steps ?? pg.totalSteps);
    }
    updateBadges();
  }

  function updateBadges() {
    // "Slowed down output": timed playback is behind the buffered frames.
    const backlog = pg.recording.length - 1 - pg.playIndex;
    slowBadge.hidden = !(pg.playing && backlog > 1);
    liveBadge.hidden = !(pg.viewMode === "live" && pg.generating);
  }

  // Keep mask glyphs flickering so the noise field looks alive.
  function startFlicker() {
    if (REDUCED_MOTION) return;
    pg.flickerTimer = setInterval(() => {
      if (!display.show_mask_glyphs) return;
      const masks = vizText.getElementsByClassName("tok-mask");
      for (let i = 0; i < masks.length; i++) masks[i].textContent = glyphRun(display.mask_glyph_style);
    }, 95);
  }

  /* ====================== playback engine ====================== */
  function setScrub(i) {
    scrub.value = String(i);
    const max = Math.max(0, pg.recording.length - 1);
    pg._scrubL.textContent = `frame ${i} / ${max}`;
    pg._scrubR.textContent = `${pg.recording.length} buffered`;
  }
  function setScrubMax(max) {
    scrub.max = String(Math.max(0, max));
    if (!pg.playing && pg.viewMode !== "live") {
      pg._scrubR.textContent = `${pg.recording.length} buffered`;
    }
  }

  function clearPlayTimer() {
    if (pg.playTimer) {
      clearTimeout(pg.playTimer);
      pg.playTimer = null;
    }
  }

  function playStep() {
    if (!pg.playing) return;
    const n = pg.recording.length;
    if (n === 0) {
      if (pg.done) {
        stopPlayback();
        return;
      }
      pg.playTimer = setTimeout(playStep, 80);
      return;
    }
    if (pg.playIndex > n - 1) pg.playIndex = n - 1;
    renderFrame(pg.recording[pg.playIndex]);
    setScrub(pg.playIndex);
    updateBadges();

    if (pg.playIndex >= n - 1) {
      if (pg.done) {
        stopPlayback();
        onReachedEnd();
        return;
      }
      // caught up to generation — wait briefly for more frames
      pg.playTimer = setTimeout(playStep, 80);
      return;
    }
    pg.playIndex++;
    pg.playTimer = setTimeout(playStep, Math.max(0, Number(display.playback_speed_ms) || 0));
  }

  function play() {
    pg.autoFollow = false; // manual playback overrides live auto-follow
    if (pg.playing) return;
    if (pg.recording.length === 0 && pg.done) return;
    if (pg.done && pg.playIndex >= pg.recording.length - 1) pg.playIndex = 0; // replay from start
    pg.playing = true;
    updateTransport();
    playStep();
  }
  function pause() {
    pg.playing = false;
    clearPlayTimer();
    updateTransport();
    updateBadges();
  }
  function stopPlayback() {
    pg.playing = false;
    clearPlayTimer();
    updateTransport();
    updateBadges();
  }
  function togglePlay() {
    if (pg.playing) pause();
    else play();
  }
  function stepForward() {
    pause();
    pg.playIndex = clamp(pg.playIndex + 1, 0, Math.max(0, pg.recording.length - 1));
    rerenderCurrent();
    setScrub(pg.playIndex);
  }
  function stepBack() {
    pause();
    pg.playIndex = clamp(pg.playIndex - 1, 0, Math.max(0, pg.recording.length - 1));
    rerenderCurrent();
    setScrub(pg.playIndex);
  }
  function restart() {
    pg.playIndex = 0;
    rerenderCurrent();
    setScrub(0);
    play();
  }
  function scrubTo(v) {
    pause();
    pg.autoFollow = false;
    pg.playIndex = clamp(v, 0, Math.max(0, pg.recording.length - 1));
    rerenderCurrent();
    setScrub(pg.playIndex);
  }
  function onReachedEnd() {
    // final clean frame stays on screen; results already shown on `done`.
    updateBadges();
  }

  // live mode: coalesce rapid arrivals into one render per animation frame
  function scheduleLiveRender() {
    if (pg.rafPending) return;
    pg.rafPending = true;
    requestAnimationFrame(() => {
      pg.rafPending = false;
      const i = pg.recording.length - 1;
      if (i < 0) return;
      pg.playIndex = i;
      renderFrame(pg.recording[i]);
      setScrub(i);
      updateBadges();
    });
  }

  function updateTransport() {
    const has = pg.recording.length > 0;
    btnBack.disabled = !has;
    btnFwd.disabled = !has;
    btnRestart.disabled = !has;
    btnPlay.disabled = !has;
    btnPlay.textContent = pg.playing ? "⏸" : "▶";
    btnPlay.setAttribute("aria-label", pg.playing ? "Pause" : "Play");
  }

  /* ====================== websocket / generation ====================== */
  function setRunStatus(text) {
    runStatus.textContent = text;
  }
  function updateButtons() {
    genBtn.disabled = pg.generating;
    cancelBtn.disabled = !pg.generating;
    genBtn.textContent = pg.generating ? "Generating…" : "Generate";
  }

  function collectParams() {
    const p = {};
    for (const name of NUMERIC_PARAMS) p[name] = pg.inputs[name].get();
    p.seed = pg.inputs.seed.get();
    return p;
  }

  function resetRun() {
    stopPlayback();
    clearPlayTimer();
    pg.recording = [];
    pg.playIndex = 0;
    pg.done = false;
    pg.cancelled = false;
    pg.finalText = null;
    pg.stats = null;
    pg.committedText = "";
    pg.viewMode = display.mode;
    pg.autoFollow = display.mode === "live";
    vizText.replaceChildren(el("span", { class: "viz-empty" }, "Connecting to the model…"));
    hideResults();
    setScrubMax(0);
    setScrub(0);
    updateTransport();
  }

  function showError(msg) {
    pg.playing = false;
    clearPlayTimer();
    vizText.replaceChildren(el("div", { class: "error-frame" }, el("strong", {}, "Error: "), String(msg)));
    setRunStatus("error");
    updateTransport();
  }

  function startGenerate() {
    if (pg.generating) return;
    const prompt = promptEl.value.trim();
    if (!prompt) {
      showError("Please enter a prompt before generating.");
      return;
    }
    resetRun();

    const params = collectParams();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/generate`;

    let ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      showError("Could not open WebSocket: " + (e && e.message ? e.message : e));
      return;
    }
    pg.ws = ws;
    pg.generating = true;
    pg.done = false;
    updateButtons();
    setRunStatus("connecting…");

    ws.onopen = () => {
      setRunStatus("generating");
      try {
        ws.send(JSON.stringify({ type: "generate", prompt, params }));
      } catch (e) {
        showError("Failed to send request: " + (e && e.message ? e.message : e));
      }
    };
    ws.onmessage = (ev) => {
      let m;
      try {
        m = JSON.parse(ev.data);
      } catch {
        return; // ignore non-JSON frames
      }
      handleMessage(m);
    };
    ws.onerror = () => {
      // onclose fires next; surface a message only if we never finished
    };
    ws.onclose = (ev) => {
      const wasRunning = pg.generating && !pg.done;
      pg.generating = false;
      updateButtons();
      updateBadges();
      if (wasRunning && !pg.cancelled) {
        showError("Connection closed before completion" + (ev && ev.reason ? `: ${ev.reason}` : "") + ". You can press Generate to retry.");
      } else if (wasRunning && pg.cancelled) {
        setRunStatus("cancelled");
      }
    };
  }

  function cancelGenerate() {
    pg.cancelled = true;
    if (pg.ws && pg.ws.readyState === WebSocket.OPEN) {
      try {
        pg.ws.send(JSON.stringify({ type: "cancel" }));
      } catch {
        /* ignore */
      }
    }
    setRunStatus("cancelling…");
    // The server should respond with `done` or close the socket; we keep
    // whatever frames were already buffered so they remain scrubbable.
  }

  function handleMessage(m) {
    switch (m && m.type) {
      case "start":
        pg.mode = m.mode === "real" ? "real" : "mock";
        if (m.canvas_length) pg.canvasLen = m.canvas_length;
        if (m.max_denoising_steps) pg.totalSteps = m.max_denoising_steps;
        updateVizHead(null);
        setRunStatus(pg.viewMode === "live" ? "live" : "buffering…");
        break;

      case "step":
        onStepFrame(m);
        break;

      case "canvas_committed":
        pg.committedText = m.text != null ? m.text : pg.committedText;
        break;

      case "done":
        onDone(m);
        break;

      case "error":
        showError(m.message || "Unknown error from server.");
        pg.generating = false;
        pg.done = true;
        updateButtons();
        break;

      default:
        // ignore unknown message types defensively
        break;
    }
  }

  function onStepFrame(m) {
    const wasEmpty = pg.recording.length === 0;
    pg.recording.push(m);
    setScrubMax(pg.recording.length - 1);
    if (wasEmpty) updateTransport(); // transport becomes usable once we have frames

    if (pg.viewMode === "live" && pg.autoFollow) {
      scheduleLiveRender();
    } else if (!pg.playing && pg.recording.length === 1) {
      // buffer mode: render the first frame as a static preview, then wait
      renderFrame(pg.recording[0]);
      setScrub(0);
    } else if (!pg.playing) {
      // keep the buffered-count readout fresh while buffering
      setScrub(pg.playIndex);
    }
    updateBadges();
  }

  function onDone(m) {
    pg.done = true;
    pg.generating = false;
    pg.finalText = m.text || "";
    pg.stats = m.stats || null;
    updateButtons();
    showResults();
    setRunStatus("done");

    if (pg.viewMode === "buffer-then-play" && !pg.playing) {
      // auto-play the buffered run from the start (unless the user is
      // already watching it fill, in which case let it run to completion)
      pg.playIndex = 0;
      play();
    } else if (pg.viewMode === "live") {
      // live: ensure the final, fully-resolved frame is on screen
      pg.playIndex = Math.max(0, pg.recording.length - 1);
      if (pg.recording.length) renderFrame(pg.recording[pg.playIndex]);
      setScrub(pg.playIndex);
    }
    updateTransport();
    updateBadges();
  }

  /* ====================== results ====================== */
  function showResults() {
    const s = pg.stats || {};
    // Estimated autoregressive throughput, shown next to the measured diffusion
    // Tokens/s so the speedup is legible at a glance. See the backend helper
    // (estimate_ar_tokens_per_second) for the formula behind this estimate.
    const arTip =
      "Estimated autoregressive throughput. An AR model emits 1 token per forward pass, " +
      "so the same output needs ~" +
      (s.tokens != null ? fmtNum(s.tokens, 0) : "tokens") +
      " sequential passes vs. this run's " +
      (s.steps_total != null ? fmtNum(s.steps_total, 0) : "—") +
      " denoising steps. ≈ steps ÷ time (assumes equal per-pass latency); an estimate.";
    const boxes = [
      ["Tokens/s", s.tokens_per_second != null ? fmtNum(s.tokens_per_second, 1) : "—"],
      ["AR tok/s (est.)", s.ar_tokens_per_second != null ? fmtNum(s.ar_tokens_per_second, 1) : "—", arTip],
      ["Tokens", s.tokens != null ? fmtNum(s.tokens, 0) : "—"],
      ["Canvases", s.canvases != null ? fmtNum(s.canvases, 0) : "—"],
      ["Steps", s.steps_total != null ? fmtNum(s.steps_total, 0) : "—"],
      ["Time (s)", s.gen_seconds != null ? fmtNum(s.gen_seconds, 2) : "—"],
    ];
    statsGrid.replaceChildren(
      ...boxes.map(([lbl, val, title]) =>
        el(
          "div",
          { class: "stat-box", title, style: title ? { cursor: "help" } : null },
          el("div", { class: "sb-num" }, val),
          el("div", { class: "sb-lbl" }, lbl)
        )
      )
    );
    finalText.textContent = pg.finalText || "(empty)";
    results.classList.add("show");
  }
  function hideResults() {
    results.classList.remove("show");
    statsGrid.replaceChildren();
    finalText.textContent = "";
    copyBtn.classList.remove("copied");
    copyBtn.textContent = "Copy";
  }

  async function copyFinal() {
    const text = pg.finalText || finalText.textContent || "";
    let ok = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch {
      ok = false;
    }
    if (!ok) {
      // fallback for non-secure contexts
      try {
        const ta = el("textarea", { style: { position: "fixed", opacity: "0" } });
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        ta.remove();
      } catch {
        ok = false;
      }
    }
    copyBtn.textContent = ok ? "Copied!" : "Copy failed";
    copyBtn.classList.toggle("copied", ok);
    setTimeout(() => {
      copyBtn.textContent = "Copy";
      copyBtn.classList.remove("copied");
    }, 1600);
  }
}
