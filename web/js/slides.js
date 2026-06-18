/* =====================================================================
   slides.js — a self-contained horizontal slide deck (the "Slides" tab)
   ---------------------------------------------------------------------
   A presentation about diffusion language models / DiffusionGemma.
   - 10 slides on a fixed 1280×720 logical "stage" scaled-to-fit, so the
     layout is pixel-stable for presenting and screen-recording.
   - Every animation is canned + offline (no server, no CDN). The two video
     clips are bundled under /static/assets/ and loaded locally.
   - Only the *active* slide animates; switching tabs pauses everything
     (driven by the document "viewchange" event dispatched from app.js).
   This module is intentionally independent of app.js (no imports) so the
   deck keeps working even if the rest of the app fails — same posture as
   howto.js.
   ===================================================================== */

const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const STAGE_W = 1280, STAGE_H = 720;

/* ----------------------------- tiny DOM helper ---------------------------- */
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith("--")) node.style.setProperty(sk, sv);
        else node.style[sk] = sv;
      }
    } else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === true) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const rnd = (a, b) => a + Math.random() * (b - a);
const pad2 = (n) => String(n).padStart(2, "0");

/* --------------------------- animation lifecycle -------------------------- *
 * A per-enter bag of stoppable things. leave() calls stopAll() so a slide
 * leaves no timers/rAFs running while it's off-screen.
 * ------------------------------------------------------------------------- */
function lifecycle() {
  const stops = [];
  return {
    timeout(fn, ms) { const id = setTimeout(fn, ms); stops.push(() => clearTimeout(id)); return id; },
    interval(fn, ms) { const id = setInterval(fn, ms); stops.push(() => clearInterval(id)); return id; },
    raf(fn) {
      let id, start = null, last = null, live = true;
      const step = (t) => {
        if (!live) return;
        if (start == null) { start = t; last = t; }
        const el = t - start, dt = t - last; last = t;
        fn(el, dt);
        id = requestAnimationFrame(step);
      };
      id = requestAnimationFrame(step);
      stops.push(() => { live = false; cancelAnimationFrame(id); });
    },
    add(stopFn) { stops.push(stopFn); },
    stopAll() { while (stops.length) { try { stops.pop()(); } catch (e) { /* ignore */ } } },
  };
}

/** Wrap a built element + an enter "runner" into the slide contract. */
function slide(id, title, el, runner) {
  let life = null, onLeave = null;
  return {
    id, title, el,
    enter() {
      life = lifecycle();
      try { onLeave = runner ? runner(life) : null; } catch (e) { console.warn("slide enter", id, e); }
    },
    leave() {
      if (onLeave) { try { onLeave(); } catch (e) { /* ignore */ } }
      if (life) life.stopAll();
      life = null; onLeave = null;
    },
  };
}

/* ------------------------------- text resolve ----------------------------- *
 * The shared "words emerge from noise" primitive (same visual language as the
 * Playground/how-to view). order: "diffusion" (whole block, random commit
 * order) or "linear" (typewriter, strictly left→right).
 * ------------------------------------------------------------------------- */
const GLYPHS = "░▒▓";
const noiseStr = (n) => { let s = ""; for (let i = 0; i < n; i++) s += GLYPHS[(Math.random() * GLYPHS.length) | 0]; return s; };
const tokenize = (t) => t.match(/\S+\s*|\s+/g) || [t];

function playResolve(life, host, opts = {}) {
  const order = opts.order || "diffusion";
  const steps = opts.steps || 22;
  const frameMs = opts.frameMs || 90;
  const loop = opts.loop !== false;
  const holdMs = opts.holdMs == null ? 1800 : opts.holdMs;
  const getText = opts.nextText || (() => opts.text || "");
  const onStep = opts.onStep;
  const onComplete = opts.onComplete;

  let tokens = [], n = 0, commitStep = [], spans = null, step = 0, total = 0;

  function setup(text) {
    tokens = tokenize(text); n = tokens.length;
    commitStep = new Array(n);
    if (order === "linear") { for (let i = 0; i < n; i++) commitStep[i] = i; }
    else {
      const idx = [...Array(n).keys()];
      for (let i = idx.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
      idx.forEach((ti, rank) => { commitStep[ti] = Math.floor((rank / n) * (steps - 2)) + 1; });
    }
    total = order === "linear" ? n : steps;
    host.replaceChildren(...tokens.map(() => document.createElement("span")));
    spans = host.children;
    step = 0;
  }

  function render(s) {
    for (let i = 0; i < n; i++) {
      const sp = spans[i];
      if (order === "linear") {
        if (i < s) { sp.className = "tok tok-committed"; sp.textContent = tokens[i]; }
        else if (i === s) { sp.className = "tok tok-tentative"; sp.textContent = tokens[i]; }
        else { sp.className = ""; sp.textContent = ""; }
      } else if (s >= commitStep[i]) {
        if (!sp.dataset.done) { sp.dataset.done = "1"; sp.className = "tok tok-committed tok-changed"; sp.textContent = tokens[i]; }
      } else {
        delete sp.dataset.done;
        sp.className = "tok tok-mask"; sp.dataset.noise = "1";
        sp.textContent = /^\s+$/.test(tokens[i]) ? tokens[i] : noiseStr(Math.min(3, Math.max(1, tokens[i].trim().length)));
      }
    }
    if (onStep) onStep(Math.min(s, total), total);
  }

  if (REDUCED_MOTION) {
    setup(getText());
    for (let i = 0; i < n; i++) { spans[i].className = "tok tok-committed"; spans[i].textContent = tokens[i]; }
    if (onStep) onStep(total, total);
    if (onComplete) onComplete();
    return;
  }

  function tick() {
    render(step);
    if (step >= total) {
      if (onComplete) onComplete();
      if (loop) life.timeout(() => { setup(getText()); tick(); }, holdMs);
      return;
    }
    step++;
    life.timeout(tick, frameMs);
  }

  setup(getText());
  tick();

  if (order !== "linear") {
    life.interval(() => {
      for (const sp of host.querySelectorAll('.tok-mask[data-noise="1"]')) {
        if (/^\s+$/.test(sp.textContent)) continue;
        sp.textContent = noiseStr(sp.textContent.length);
      }
    }, 110);
  }
}

/* --------------------------- small UI components -------------------------- */
function meter(label, klass) {
  const fill = h("i", { class: "meter-fill" });
  const val = h("b", {}, "0%");
  const node = h("div", { class: "meter " + (klass || "") },
    h("div", { class: "meter-top" }, h("span", {}, label), val),
    h("div", { class: "meter-track" }, fill));
  return { node, set(pct, text) { fill.style.width = clamp(pct, 0, 100) + "%"; val.textContent = text != null ? text : Math.round(pct) + "%"; } };
}
function bignum(unit) {
  const n = h("span", { class: "n" }, "0");
  const node = h("div", { class: "bignum" }, n, h("span", { class: "u" }, unit));
  return { node, set(v) { n.textContent = v; } };
}

/* ============================================================================
   THE SLIDES
   ========================================================================= */

/* ---- 1 · Title --------------------------------------------------------- */
function slideTitle() {
  const field = h("div", { class: "glyphfield", "aria-hidden": "true" });
  const resolveLine = h("div", { class: "mono-stage anim", style: { "--d": "0.5s", fontSize: "1.5rem", color: "var(--accent-2)" } });
  const el = h("div", { class: "slide", id: "s-title" },
    field,
    h("div", {},
      h("p", { class: "slide-kicker anim", style: { "--d": "0s" } }, "Research share · 2026"),
      h("h1", { class: "slide-title anim-zoom", style: { "--d": "0.12s", fontSize: "4.4rem" } }, "Diffusion ", h("span", { class: "grad" }, "Language Models")),
      h("p", { class: "slide-sub anim", style: { "--d": "0.32s", marginBottom: "26px" } }, "Featuring ", h("strong", { style: { color: "var(--text)" } }, "DiffusionGemma"), " — Google's 26B-A4B-it, released ~June 10, 2026."),
      resolveLine));
  function fillField() {
    const cols = 96, rows = 17;
    let s = "";
    for (let r = 0; r < rows; r++) { for (let c = 0; c < cols; c++) s += GLYPHS[(Math.random() * 3) | 0]; s += "\n"; }
    field.textContent = s;
  }
  fillField();
  return slide("s-title", "Diffusion Language Models", el, (life) => {
    playResolve(life, resolveLine, { text: "text that resolves from noise — not left to right.", order: "diffusion", steps: 24, frameMs: 95, holdMs: 2600 });
    if (!REDUCED_MOTION) life.interval(fillField, 240);
  });
}

/* ---- 2 · Typewriter vs Canvas ----------------------------------------- */
function slideVersus() {
  const arLine = h("div", { class: "mono-stage caret", id: "vs-ar" });
  const diffLine = h("div", { class: "mono-stage", id: "vs-diff" });
  const el = h("div", { class: "slide", id: "s-versus" },
    h("div", { class: "slide-num-ghost" }, "01"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "01 — Two ways to write"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "Typewriter ", h("span", { style: { color: "var(--muted)" } }, "vs."), " Canvas"),
      h("div", { class: "two-col anim", style: { "--d": "0.2s", marginTop: "30px" } },
        h("div", { class: "panel-lite" },
          h("header", {}, h("span", { class: "lbl-pill ar" }, "Autoregressive"), h("span", { class: "tag-ar" }, "Typewriter")),
          h("div", { style: { minHeight: "3.6em", display: "flex", alignItems: "center" } }, arLine),
          h("p", { class: "passnote", style: { color: "var(--muted)" } }, "one token at a time · left → right · can't revise")),
        h("div", { class: "panel-lite" },
          h("header", {}, h("span", { class: "lbl-pill diff" }, "Diffusion"), h("span", { class: "tag-diff" }, "Canvas")),
          h("div", { style: { minHeight: "3.6em", display: "flex", alignItems: "center" } }, diffLine),
          h("p", { class: "passnote", style: { color: "var(--muted)" } }, "whole canvas at once · commits the confident words first · self-corrects")))));
  const LINES = [
    "May the Force be with you.",
    "I'm sorry, Dave. I'm afraid I can't do that.",
    "It's dangerous to go alone! Take this.",
    "Winter is coming.",
    "The cake is a lie.",
  ];
  return slide("s-versus", "Typewriter vs. Canvas", el, (life) => {
    let i = 0;
    function run() {
      const line = LINES[i % LINES.length]; i++;
      let arDone = false, diffDone = false;
      const next = () => { if (arDone && diffDone) life.timeout(run, 1700); };
      playResolve(life, arLine, { text: line, order: "linear", frameMs: 95, loop: false, onComplete: () => { arDone = true; next(); } });
      playResolve(life, diffLine, { text: line, order: "diffusion", steps: 18, frameMs: 100, loop: false, onComplete: () => { diffDone = true; next(); } });
    }
    run();
  });
}

/* ---- 3 · Image diffusion (canvas) ------------------------------------- */
function slideImageDiffusion() {
  const RW = 240, RH = 150;
  const canvas = h("canvas", { class: "diffuse-canvas", width: RW, height: RH, style: { width: "440px", height: "275px" }, "aria-label": "an image emerging from noise" });
  const stepLbl = h("div", { class: "mono", style: { fontFamily: "var(--mono)", fontSize: ".95rem", color: "var(--text-dim)", marginTop: "14px" } }, "step 0 / 25");
  const bar = h("i", {});
  const el = h("div", { class: "slide", id: "s-imgdiff" },
    h("div", { class: "slide-num-ghost" }, "02"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "02 — Where the idea comes from"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "Diffusion, borrowed from images"),
      h("div", { class: "diffuse-wrap anim", style: { "--d": "0.2s", marginTop: "26px" } },
        canvas,
        h("div", { class: "noisebar-wrap" },
          h("p", { class: "slide-sub", style: { fontSize: "1.2rem" } }, "Start from pure static. Each step removes a little noise — until a picture remains."),
          stepLbl,
          h("div", { class: "noisebar" }, bar),
          h("p", { class: "passnote", style: { marginTop: "10px" } }, "DiffusionGemma does the same — but the ", h("span", { class: "tag-diff" }, "canvas is text tokens"), ", not pixels.")))));

  const g = canvas.getContext("2d");
  // Render the clean target once into an offscreen buffer.
  const off = document.createElement("canvas"); off.width = RW; off.height = RH;
  const og = off.getContext("2d");
  drawScene(og, RW, RH);
  const target = og.getImageData(0, 0, RW, RH).data;
  const frame = g.createImageData(RW, RH);
  const fd = frame.data;
  const STEPS = 30;

  function renderSigma(sigma) {
    // sigma=1 -> pure static (no scene); sigma=0 -> clean image. Crossfade the
    // target with independent per-channel noise so it STARTS fully noised.
    const t = 1 - sigma;
    for (let i = 0; i < target.length; i += 4) {
      fd[i] = target[i] * t + Math.random() * 255 * sigma;
      fd[i + 1] = target[i + 1] * t + Math.random() * 255 * sigma;
      fd[i + 2] = target[i + 2] * t + Math.random() * 255 * sigma;
      fd[i + 3] = 255;
    }
    g.putImageData(frame, 0, 0);
  }

  return slide("s-imgdiff", "What is diffusion?", el, (life) => {
    if (REDUCED_MOTION) { renderSigma(0); bar.style.width = "100%"; stepLbl.textContent = "denoised"; return; }
    let k = 0;
    function step() {
      const sigma = 1 - k / (STEPS - 1);
      renderSigma(sigma);
      bar.style.width = Math.round((1 - sigma) * 100) + "%";
      stepLbl.textContent = `step ${k} / ${STEPS - 1}  ·  noise ${Math.round(sigma * 100)}%`;
      if (k >= STEPS - 1) { life.timeout(() => { k = 0; step(); }, 2200); return; }
      k++;
      life.timeout(step, 200);
    }
    step();
  });
}

function drawScene(g, W, H) {
  const sky = g.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, "#0a0f2c"); sky.addColorStop(.5, "#241248"); sky.addColorStop(1, "#3a1a4a");
  g.fillStyle = sky; g.fillRect(0, 0, W, H);
  g.fillStyle = "#dfe6ff";
  for (let i = 0; i < 70; i++) { g.globalAlpha = rnd(.25, 1); g.beginPath(); g.arc(rnd(0, W), rnd(0, H * .72), rnd(.3, 1.4), 0, 7); g.fill(); }
  g.globalAlpha = 1;
  const px = W * .68, py = H * .42, pr = H * .26;
  const pg = g.createRadialGradient(px - pr * .3, py - pr * .3, pr * .1, px, py, pr);
  pg.addColorStop(0, "#ffe1b0"); pg.addColorStop(.5, "#ff9d5c"); pg.addColorStop(1, "#bf3f1d");
  g.fillStyle = pg; g.beginPath(); g.arc(px, py, pr, 0, 7); g.fill();
  g.strokeStyle = "rgba(255,220,180,.65)"; g.lineWidth = 3;
  g.save(); g.translate(px, py); g.rotate(-.4); g.scale(1, .32); g.beginPath(); g.arc(0, 0, pr * 1.7, 0, 7); g.stroke(); g.restore();
  const hz = H * .8; g.fillStyle = "#0b0712"; g.beginPath(); g.moveTo(0, H); g.lineTo(0, hz);
  for (let x = 0; x <= W; x += W / 16) g.lineTo(x, hz - Math.sin(x * .06) * 7 - (x > W * .4 && x < W * .72 ? 24 : 0));
  g.lineTo(W, H); g.closePath(); g.fill();
}

/* ---- 4 · How AR LLMs work + hardware ---------------------------------- */
function slideAR() {
  const tokrow = h("div", { class: "tokrow", id: "ar-tokrow" });
  const gpu = meter("GPU compute used", "amber");
  const mem = meter("Memory bandwidth", "amber");
  const passes = bignum("forward passes");
  const el = h("div", { class: "slide", id: "s-ar" },
    h("div", { class: "slide-num-ghost" }, "03"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "03 — Autoregressive LLMs"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "One token per forward pass"),
      h("div", { class: "two-col anim", style: { "--d": "0.2s", marginTop: "26px", alignItems: "start" } },
        h("div", { class: "panel-lite" },
          h("header", {}, h("span", { class: "lbl-pill ar" }, "Generating"), h("span", { class: "tag-ar" }, "left → right")),
          tokrow,
          h("p", { class: "passnote" }, h("b", {}, "1"), " token / forward pass · strictly sequential")),
        h("div", { class: "panel-lite" },
          h("div", { class: "meters" }, gpu.node, mem.node),
          passes.node,
          h("p", { class: "passnote" }, "GPU sits ", h("span", { class: "tag-ar" }, "starved"), ", waiting on memory")))),
    h("div", { class: "slide-foot" }, h("span", {}, "Latency grows with every token — but it ", h("strong", { style: { color: "var(--text)" } }, "batches"), " beautifully across many users.")));
  const SENT = "Paris is the capital of France .".split(" ");
  return slide("s-ar", "How AR models work", el, (life) => {
    if (REDUCED_MOTION) {
      tokrow.replaceChildren(...SENT.map((w) => h("span", { class: "tokchip in" }, w)));
      gpu.set(38, "38%"); mem.set(86, "86%"); passes.set(SENT.length); return;
    }
    let i = 0;
    function addTok() {
      if (i < SENT.length) {
        const chip = h("span", { class: "tokchip hot" }, SENT[i]);
        tokrow.appendChild(chip);
        requestAnimationFrame(() => chip.classList.add("in"));
        life.timeout(() => chip.classList.remove("hot"), 280);
        passes.set(i + 1);
        i++;
        life.timeout(addTok, 360);
      } else {
        life.timeout(() => { tokrow.replaceChildren(); i = 0; passes.set(0); addTok(); }, 2200);
      }
    }
    addTok();
    // meters: GPU low + spiky (memory-bandwidth bound), memory near saturation.
    mem.set(86, "86%");
    life.interval(() => {
      gpu.set(rnd(26, 46), null);
      mem.set(rnd(80, 92), null);
    }, 130);
  });
}

/* ---- 5 · How Diffusion LMs work + hardware ---------------------------- */
function slideDiffusion() {
  const COLS = 24, ROWS = 5, TOTAL = COLS * ROWS;
  const block = h("div", { class: "canvasblock", id: "df-block", style: { "--cols": COLS } },
    ...Array.from({ length: TOTAL }, () => h("div", { class: "cvtok mask" })));
  const cells = block.children;
  const gpu = meter("GPU compute used", "good");
  const mem = meter("Memory bandwidth", "");
  const passes = bignum("denoising passes");
  const el = h("div", { class: "slide", id: "s-diff" },
    h("div", { class: "slide-num-ghost" }, "04"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "04 — Diffusion LMs"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "A whole canvas per pass"),
      h("div", { class: "two-col anim", style: { "--d": "0.2s", marginTop: "26px", alignItems: "start" } },
        h("div", { class: "panel-lite" },
          h("header", {}, h("span", { class: "lbl-pill diff" }, "Denoising"), h("span", { class: "tag-diff" }, "all positions at once")),
          block,
          h("p", { class: "passnote" }, h("b", {}, "~15–20"), " tokens / pass · ~4× fewer steps")),
        h("div", { class: "panel-lite" },
          h("div", { class: "meters" }, gpu.node, mem.node),
          passes.node,
          h("p", { class: "passnote" }, "GPU runs ", h("span", { class: "tag-diff" }, "hot"), " — compute-bound, not memory-starved")))),
    h("div", { class: "slide-foot" }, h("span", {}, "One stream resolves fast → ", h("strong", { style: { color: "var(--text)" } }, "low latency"), " on a single local GPU.")));
  return slide("s-diff", "How diffusion LMs work", el, (life) => {
    const reset = () => { for (const c of cells) c.className = "cvtok mask"; };
    if (REDUCED_MOTION) { for (const c of cells) c.className = "cvtok commit"; gpu.set(88, "88%"); mem.set(54, "54%"); passes.set(12); return; }
    let committed, pass;
    function start() {
      reset(); committed = 0; pass = 0; passes.set(0);
    }
    function tickPass() {
      if (committed >= TOTAL) { life.timeout(() => { start(); tickPass(); }, 2100); return; }
      // commit a batch of random still-masked cells (low-entropy-first, illustratively).
      const open = [];
      for (let i = 0; i < TOTAL; i++) if (cells[i].classList.contains("mask")) open.push(i);
      for (let k = open.length - 1; k > 0; k--) { const j = (Math.random() * (k + 1)) | 0; [open[k], open[j]] = [open[j], open[k]]; }
      const take = open.slice(0, Math.min(open.length, 14 + ((Math.random() * 8) | 0)));
      for (const i of take) { const c = cells[i]; c.className = "cvtok commit fresh"; life.timeout(() => c.classList.remove("fresh"), 340); }
      committed += take.length; pass++; passes.set(pass);
      life.timeout(tickPass, 430);
    }
    start(); tickPass();
    gpu.set(88, "88%"); mem.set(54, "54%");
    life.interval(() => { gpu.set(rnd(82, 94), null); mem.set(rnd(48, 62), null); }, 150);
  });
}

/* ---- 6 · Cloud vs Local ----------------------------------------------- */
function slideCloudLocal() {
  // AR: many user streams packed into ONE GPU (1 token / user / batched pass)
  const USERS = 5, AR_STEPS = 9;
  const arChips = [], lanes = [];
  for (let u = 0; u < USERS; u++) {
    const row = [], toks = h("div", { class: "toks" });
    for (let c = 0; c < AR_STEPS; c++) { const tk = h("span", { class: "tk" }); row.push(tk); toks.appendChild(tk); }
    arChips.push(row);
    lanes.push(h("div", { class: "lane" }, h("span", { class: "uid" }, "u" + (u + 1)), toks));
  }
  const arThru = h("b", {}, "0");
  const arGpu = h("div", { class: "gpu" },
    h("div", { class: "gpu-head" }, h("span", { class: "lbl-pill ar" }, "AR \u00b7 cloud"), h("span", { class: "chip-gpu" }, "1 GPU")),
    h("div", { class: "gpu-body" }, h("div", { class: "lanes" }, ...lanes)),
    h("div", { class: "gpu-foot" },
      h("div", { class: "batch-stat" }, "1 pass \u2192 +1 token \u00d7 ", h("b", {}, USERS + " users"), "  \u00b7  served: ", arThru),
      h("div", { class: "verdict ar" }, "\u2192 high throughput \u00b7 scales in the cloud")));

  // Diffusion: ONE user's whole canvas fills the GPU (many tokens / pass)
  const DCOLS = 12, DROWS = 4, DTOTAL = DCOLS * DROWS;
  const dCells = [];
  const dGrid = h("div", { class: "fill-canvas", style: { "--cols": DCOLS } });
  for (let i = 0; i < DTOTAL; i++) { const c = h("span", { class: "fc" }); dCells.push(c); dGrid.appendChild(c); }
  const dPass = h("b", {}, "0");
  const diffGpu = h("div", { class: "gpu" },
    h("div", { class: "gpu-head" }, h("span", { class: "lbl-pill diff" }, "Diffusion \u00b7 local"), h("span", { class: "chip-gpu" }, "1 GPU")),
    h("div", { class: "gpu-body" }, h("div", { class: "one-user" }, h("span", { class: "uid" }, "u1"), dGrid)),
    h("div", { class: "gpu-foot" },
      h("div", { class: "batch-stat" }, "1 pass \u2192 whole canvas \u00b7 ", h("b", {}, "1 user"), "  \u00b7  passes: ", dPass),
      h("div", { class: "verdict diff" }, "\u2192 low latency \u00b7 great on your own box")));

  const el = h("div", { class: "slide", id: "s-cloudlocal" },
    h("div", { class: "slide-num-ghost" }, "05"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "05 \u2014 Where each one wins"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "One GPU, two ways to fill it"),
      h("div", { class: "batch-grid anim", style: { "--d": "0.2s", marginTop: "28px" } }, arGpu, diffGpu)),
    h("div", { class: "slide-foot" }, h("span", {}, "Same hardware, opposite sweet spot \u2014 AR packs the GPU with ", h("strong", { style: { color: "var(--amber)" } }, "many users"), "; diffusion gives it all to ", h("strong", { style: { color: "var(--accent-2)" } }, "one"), ".")));

  const shuffle = (a) => { for (let k = a.length - 1; k > 0; k--) { const j = (Math.random() * (k + 1)) | 0; [a[k], a[j]] = [a[j], a[k]]; } return a; };

  return slide("s-cloudlocal", "Cloud vs. local", el, (life) => {
    function resetAR() { for (const row of arChips) for (const tk of row) tk.className = "tk"; }
    function resetDiff() { for (const c of dCells) c.className = "fc"; }
    if (REDUCED_MOTION) {
      for (const row of arChips) row.forEach((tk) => { tk.className = "tk on"; });
      arThru.textContent = USERS * AR_STEPS;
      for (const c of dCells) c.className = "fc on";
      dPass.textContent = "3";
      return;
    }
    let col = 0, served = 0;
    function arStep() {
      if (col >= AR_STEPS) { life.timeout(() => { resetAR(); col = 0; served = 0; arThru.textContent = "0"; arStep(); }, 1600); return; }
      for (const row of arChips) { if (col > 0) row[col - 1].classList.remove("head"); row[col].className = "tk on head"; }
      served += USERS; arThru.textContent = served; col++;
      life.timeout(arStep, 460);
    }
    arStep();
    let dCommitted = 0, pass = 0;
    function diffStep() {
      if (dCommitted >= DTOTAL) { life.timeout(() => { resetDiff(); dCommitted = 0; pass = 0; dPass.textContent = "0"; diffStep(); }, 1900); return; }
      const open = [];
      for (let i = 0; i < DTOTAL; i++) if (!dCells[i].classList.contains("on")) open.push(i);
      shuffle(open);
      const take = open.slice(0, Math.ceil(DTOTAL / 3.2));
      for (const i of take) { const c = dCells[i]; c.className = "fc on fresh"; life.timeout(() => c.classList.remove("fresh"), 360); }
      dCommitted += take.length; pass++; dPass.textContent = pass;
      life.timeout(diffStep, 620);
    }
    diffStep();
  });
}

/* ---- 7 · The catch (reddit video) ------------------------------------- */
function slideTradeoff() {
  const video = h("video", { src: "/static/assets/diffusion-vs-ar.mp4", loop: true, muted: true, playsinline: true, controls: true, preload: "auto" });
  video.muted = true; video.playsInline = true;
  const el = h("div", { class: "slide", id: "s-tradeoff" },
    h("div", { class: "slide-num-ghost" }, "06"),
    h("div", { class: "video-row" },
      h("div", { class: "video-frame anim-zoom" }, video),
      h("div", { class: "anim", style: { "--d": "0.2s" } },
        h("p", { class: "slide-kicker", style: { marginBottom: "10px" } }, "06 — The catch"),
        h("h2", { class: "slide-title", style: { fontSize: "2.8rem" } }, "Faster — but ", h("span", { style: { color: "var(--bad)" } }, "noisier")),
        h("div", { class: "kpi" },
          h("div", {}, h("div", { class: "kpi-1 up" }, "~4×"), h("small", {}, "faster generation")),
          h("div", {}, h("div", { class: "kpi-1 down" }, "~6×"), h("small", {}, "more errors vs AR"))),
        h("p", { class: "slide-sub", style: { fontSize: "1.15rem" } }, "Speed isn't free. Side-by-side, diffusion races ahead — and trips more often."))),
    h("div", { class: "slide-foot" }, h("span", {}, "Watch it race the typewriter."), h("span", { class: "src" }, "source: r/LocalLLaMA")));
  return slide("s-tradeoff", "Faster, but noisier", el, () => {
    video.currentTime = 0; const p = video.play(); if (p && p.catch) p.catch(() => {});
    return () => video.pause();
  });
}

/* ---- 7b - DiffusionGemma at a glance ---------------------------------- */
function slideModel() {
  const INPUTS = [
    { ic: "\u2261", label: "Text" },
    { ic: "\u25a6", label: "Image" },
    { ic: "\u25ba", label: "Video" },
  ];
  const inChips = INPUTS.map((x) => h("div", { class: "mm-chip" }, h("span", { class: "mm-ico" }, x.ic), x.label));
  const outHost = h("div", { class: "mono mm-out-text" });
  const flow = h("div", { class: "mm-flow" },
    h("div", { class: "mm-inputs" }, ...inChips),
    h("div", { class: "mm-beam" }),
    h("div", { class: "mm-core" }, h("span", { class: "mm-gem" }, "\u25c6"), h("b", {}, "DiffusionGemma")),
    h("div", { class: "mm-beam" }),
    h("div", { class: "mm-out" }, h("div", { class: "mm-out-label" }, "one model, any input"), outHost));

  const tiles = [
    h("div", { class: "spec-tile" },
      h("div", { class: "spec-h" }, "Encoder\u2013decoder"),
      h("div", { class: "spec-d" }, "not decoder-only"),
      h("div", { class: "arch-mini" }, h("span", { class: "ab e" }, "E"), h("span", { class: "ar2" }, "\u2192"), h("span", { class: "ab d" }, "D"))),
    h("div", { class: "spec-tile" },
      h("div", { class: "spec-h" }, "Thinking mode"),
      h("div", { class: "spec-d" }, "reasons first"),
      h("div", { class: "think" }, h("i", {}), h("i", {}), h("i", {}))),
    h("div", { class: "spec-tile" },
      h("div", { class: "spec-big" }, "256K"),
      h("div", { class: "spec-d" }, "token context")),
    h("div", { class: "spec-tile" },
      h("div", { class: "spec-h" }, "Function calling"),
      h("div", { class: "spec-d" }, "tools \u00b7 early*")),
    h("div", { class: "spec-tile" },
      h("div", { class: "spec-h" }, "Multilingual"),
      h("div", { class: "spec-d mm-lang" }, "Hello")),
  ];

  const el = h("div", { class: "slide", id: "s-model" },
    h("div", { class: "slide-num-ghost" }, "07"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "07 \u2014 Meet the model"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "DiffusionGemma, at a glance"),
      h("div", { class: "mm-wrap anim", style: { "--d": "0.2s", marginTop: "24px" } }, flow),
      h("div", { class: "spec-grid anim", style: { "--d": "0.34s", marginTop: "22px" } }, ...tiles)),
    h("div", { class: "slide-foot" }, h("span", {}, "*function calling is supported \u2014 just not its strong suit.")));

  const OUTS = ["a caption.", "a summary.", "an answer."];
  const GREET = ["Hello", "\u4f60\u597d", "\u0645\u0631\u062d\u0628\u0627", "Hola", "\u3053\u3093\u306b\u3061\u306f"];
  return slide("s-model", "DiffusionGemma at a glance", el, (life) => {
    let mi = 0;
    const cycleIn = () => { inChips.forEach((c, k) => c.classList.toggle("active", k === mi % inChips.length)); mi++; };
    cycleIn();
    if (!REDUCED_MOTION) life.interval(cycleIn, 1100);
    let oi = 0;
    playResolve(life, outHost, { order: "diffusion", steps: 16, frameMs: 90, holdMs: 1500, nextText: () => OUTS[oi++ % OUTS.length] });
    const langEl = el.querySelector(".mm-lang");
    if (langEl) { let gi = 0; const t = () => { langEl.textContent = GREET[gi++ % GREET.length]; }; t(); if (!REDUCED_MOTION) life.interval(t, 1300); }
  });
}

/* ---- 8 · Fine-tune use-cases ------------------------------------------ */
function slideFinetunes() {
  /* sudoku 4×4 */
  const given = { 0: "1", 2: "3", 5: "4", 7: "2", 8: "2", 10: "4", 13: "3", 15: "1" };
  const solution = { 1: "2", 3: "4", 4: "3", 6: "1", 9: "1", 11: "3", 12: "4", 14: "2" };
  const wrongFirst = { 4: "1", 11: "2" };
  const sudoku = h("div", { class: "sudoku" },
    ...Array.from({ length: 16 }, (_, i) => h("div", { class: "cell " + (given[i] ? "given" : "solving") }, given[i] || "·")));
  const sCells = sudoku.children;
  /* code FIM — multiple holes, filled in parallel (name, iterable, expression) */
  const kw = (t) => h("span", { class: "kw" }, t);
  const holes = [h("span", { class: "hole" }, "······"), h("span", { class: "hole" }, "··"), h("span", { class: "hole" }, "·····")];
  const fim = h("div", { class: "fim" },
    h("div", { class: "ln" }, kw("def "), holes[0], "(xs):"),
    h("div", { class: "ln" }, "    out = []"),
    h("div", { class: "ln" }, "    ", kw("for "), "x ", kw("in "), holes[1], ":"),
    h("div", { class: "ln" }, "        out.append(", holes[2], ")"),
    h("div", { class: "ln" }, "    ", kw("return "), "out"));
  /* json */
  const jb = { name: h("span", { class: "blank" }, "·····"), id: h("span", { class: "blank" }, "··"), ok: h("span", { class: "blank" }, "····") };
  const json = h("div", { class: "jsonv" },
    "{", h("br", {}),
    "  ", h("span", { class: "k" }, '"name"'), ": ", jb.name, ",", h("br", {}),
    "  ", h("span", { class: "k" }, '"id"'), ": ", jb.id, ",", h("br", {}),
    "  ", h("span", { class: "k" }, '"ok"'), ": ", jb.ok, h("br", {}),
    "}");
  const el = h("div", { class: "slide", id: "s-finetunes" },
    h("div", { class: "slide-num-ghost" }, "08"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "08 — Fine-tuning shines on non-linear text"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem" } }, "Built to fill in the blanks"),
      h("div", { class: "ft-cards anim", style: { "--d": "0.2s", marginTop: "30px" } },
        h("div", { class: "ft-card" },
          h("h3", {}, h("span", { class: "ft-ico" }, "🧩"), "Sudoku & constraints"),
          h("div", { class: "ft-viz" }, sudoku),
          h("p", {}, "Guesses commit, clash, then ", h("span", { style: { color: "var(--good)" } }, "self-correct"), " — the whole grid at once.")),
        h("div", { class: "ft-card" },
          h("h3", {}, h("span", { class: "ft-ico" }, "{ }"), "Code infilling"),
          h("div", { class: "ft-viz" }, fim),
          h("p", {}, "Fill ", h("em", {}, "many"), " gaps at once — names, loops, returns — from both sides.")),
        h("div", { class: "ft-card" },
          h("h3", {}, h("span", { class: "ft-ico" }, "⌗"), "Structured output"),
          h("div", { class: "ft-viz" }, json),
          h("p", {}, "Emit JSON / tables where the whole shape resolves together.")))));
  return slide("s-finetunes", "Fine-tune use-cases", el, (life) => {
    const shuffle = (a) => { for (let k = a.length - 1; k > 0; k--) { const j = (Math.random() * (k + 1)) | 0; [a[k], a[j]] = [a[j], a[k]]; } return a; };
    function solveSudoku() {
      const blanks = shuffle(Object.keys(solution).map(Number));
      blanks.forEach((i, rank) => life.timeout(() => {
        const c = sCells[i];
        if (wrongFirst[i]) { c.textContent = wrongFirst[i]; c.className = "cell wrong"; }
        else { c.textContent = solution[i]; c.className = "cell right"; }
      }, 150 + rank * 95));
      const after = 150 + blanks.length * 95 + 400;
      Object.keys(wrongFirst).map(Number).forEach((i, k) => life.timeout(() => {
        const c = sCells[i]; c.textContent = solution[i]; c.className = "cell right";
        c.style.boxShadow = "0 0 14px rgba(52,168,83,.6)";
        life.timeout(() => { c.style.boxShadow = ""; }, 450);
      }, after + k * 300));
    }
    function resetSudoku() { for (let i = 0; i < 16; i++) if (!given[i]) { sCells[i].textContent = "·"; sCells[i].className = "cell solving"; } }
    function fillFim() { const v = ["double", "xs", "x * 2"]; holes.forEach((hl, k) => life.timeout(() => { hl.textContent = v[k]; hl.className = "hole filled"; }, k * 140)); }
    function resetFim() { const ph = ["······", "··", "·····"]; holes.forEach((hl, k) => { hl.textContent = ph[k]; hl.className = "hole"; }); }
    function fillJson() {
      jb.name.textContent = '"Gemma"'; jb.name.className = "blank set s";
      jb.id.textContent = "42"; jb.id.className = "blank set n";
      jb.ok.textContent = "true"; jb.ok.className = "blank set n";
    }
    function resetJson() { for (const b of Object.values(jb)) { b.className = "blank"; } jb.name.textContent = "·····"; jb.id.textContent = "··"; jb.ok.textContent = "····"; }
    if (REDUCED_MOTION) {
      for (const i of Object.keys(solution)) { sCells[i].textContent = solution[i]; sCells[i].className = "cell right"; }
      fillFim(); fillJson(); return;
    }
    function cycle() {
      resetSudoku(); resetFim(); resetJson();
      life.timeout(solveSudoku, 300);
      life.timeout(fillFim, 800);
      life.timeout(fillJson, 1400);
      life.timeout(cycle, 6200);
    }
    cycle();
  });
}

/* ---- 9 · Why I'm presenting this (ORKG Ask) --------------------------- */
function slideWhy() {
  const items = [
    h("div", { class: "why-item anim", style: { "--d": "0.2s" } },
      h("div", { class: "why-n" }, "1"),
      h("div", {}, h("h3", {}, "It's genuinely cool"), h("p", {}, "Watching text resolve from noise is a different way to think about generation."))),
    h("div", { class: "why-item ask anim", style: { "--d": "0.34s" } },
      h("div", { class: "why-n" }, "2"),
      h("div", {}, h("h3", {}, "Useful for our work"), h("p", {}, "Fast, structured, infill-style answers could help ", h("span", { class: "ask-chip" }, "ORKG Ask"), " and other projects..."))),
    h("div", { class: "why-item q anim", style: { "--d": "0.48s" } },
      h("div", { class: "why-n" }, "3"),
      h("div", {}, h("h3", { class: "q-pulse" }, "Do you have a use-case?"), h("p", {}, "Non-linear text, editing, constraints\u2026 where would this fit for ", h("em", {}, "you"), "?"))),
  ];
  const resolveHost = h("div", { class: "mono why-resolve" });
  const el = h("div", { class: "slide", id: "s-why" },
    h("div", { class: "slide-num-ghost" }, "09"),
    h("div", {},
      h("p", { class: "slide-kicker anim" }, "09 \u2014 Why I'm sharing this"),
      h("h2", { class: "slide-title anim", style: { "--d": "0.08s", fontSize: "2.6rem", marginBottom: "30px" } }, "Why this, why now"),
      h("div", { class: "why-grid" },
        h("div", { class: "why-list" }, ...items),
        h("div", { class: "why-viz anim", style: { "--d": "0.3s" } },
          h("div", { class: "vz-label" }, "the pitch, denoised"),
          resolveHost))));
  const PHRASES = ["it's genuinely cool!", "useful for ORKG Ask?", "what's your use-case?"];
  return slide("s-why", "Why present this?", el, (life) => {
    let idx = 0;
    const setActive = (a) => items.forEach((it, k) => it.classList.toggle("active", k === a));
    playResolve(life, resolveHost, {
      order: "diffusion", steps: 20, frameMs: 95, holdMs: 2300,
      nextText: () => { const a = idx % PHRASES.length; setActive(a); idx++; return PHRASES[a]; },
    });
  });
}

/* ---- 10 · Not the only one (Nemotron video) --------------------------- */
function slideNemotron() {
  const video = h("video", { src: "/static/assets/nemotron-demo.mp4", loop: true, muted: true, playsinline: true, controls: true, preload: "auto" });
  video.muted = true; video.playsInline = true;
  const el = h("div", { class: "slide", id: "s-nemotron" },
    h("div", { class: "slide-num-ghost" }, "10"),
    h("div", { class: "video-row" },
      h("div", { class: "video-frame anim-zoom" }, video),
      h("div", { class: "anim", style: { "--d": "0.2s" } },
        h("p", { class: "slide-kicker", style: { marginBottom: "10px" } }, "10 — Not the only one"),
        h("h2", { class: "slide-title", style: { fontSize: "2.6rem" } }, "Meet ", h("span", { class: "grad" }, "Nemotron-Labs-Diffusion")),
        h("p", { class: "slide-sub", style: { fontSize: "1.18rem" } }, "NVIDIA's 3B diffusion LM — a parallel sign that block-diffusion text models are heating up."))),
    h("div", { class: "slide-foot" }, h("span", {}, "Thanks — questions?"), h("span", { class: "src" }, "source: huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B")));
  return slide("s-nemotron", "A similar model: Nemotron", el, () => {
    video.currentTime = 0; const p = video.play(); if (p && p.catch) p.catch(() => {});
    return () => video.pause();
  });
}

/* ============================================================================
   DECK ENGINE
   ========================================================================= */
export function initSlides(root, ctx) {
  if (!root) return;

  const slides = [
    slideTitle(), slideVersus(), slideImageDiffusion(), slideAR(), slideDiffusion(),
    slideCloudLocal(), slideTradeoff(), slideModel(), slideFinetunes(), slideWhy(), slideNemotron(),
  ];

  /* ---- build DOM ---- */
  const track = h("div", { class: "deck-track" }, ...slides.map((s) => s.el));
  const stage = h("div", { class: "deck-stage" }, track);
  const overview = h("div", { class: "deck-overview", hidden: true });
  const stagewrap = h("div", { class: "deck-stagewrap" }, stage, overview);

  const prevBtn = h("button", { class: "deck-arrow", "aria-label": "Previous slide", title: "Previous (←)" }, "‹");
  const nextBtn = h("button", { class: "deck-arrow", "aria-label": "Next slide", title: "Next (→)" }, "›");
  const dots = slides.map((s, k) => h("button", { class: "deck-dot", "aria-label": `Go to slide ${k + 1}: ${s.title}`, onclick: () => goTo(k) }));
  const counter = h("span", { class: "deck-counter" });
  const ovBtn = h("button", { class: "deck-btn", title: "Overview (Esc)", onclick: () => toggleOverview() }, "▦ Overview");
  const bar = h("div", { class: "deck-bar" },
    h("div", { class: "side" }, ovBtn, h("span", { class: "deck-hint" }, "← → or Space")),
    h("div", { class: "side", style: { justifyContent: "center" } }, h("div", { class: "deck-nav" }, prevBtn, h("div", { class: "deck-dots" }, ...dots), nextBtn)),
    h("div", { class: "side right" }, counter));

  if (REDUCED_MOTION) root.classList.add("no-anim");
  root.replaceChildren(stagewrap, bar);

  /* ---- overview cards ---- */
  overview.replaceChildren(
    h("div", { class: "ov-head" }, h("h3", {}, "All slides"), h("span", { class: "deck-hint" }, "click a slide · Esc to close")),
    ...slides.map((s, k) => h("button", { class: "ov-card", onclick: () => { goTo(k); closeOverview(); } },
      h("div", { class: "ov-n" }, pad2(k + 1)), h("div", { class: "ov-t" }, s.title))));
  const ovCards = [...overview.querySelectorAll(".ov-card")];

  /* ---- state ---- */
  let index = 0, active = false, overviewOpen = false;

  function applyTransform() { track.style.transform = `translateX(${-index * STAGE_W}px)`; }
  function updateChrome() {
    dots.forEach((d, k) => d.setAttribute("aria-current", k === index ? "true" : "false"));
    ovCards.forEach((c, k) => c.setAttribute("aria-current", k === index ? "true" : "false"));
    counter.replaceChildren(h("b", {}, pad2(index + 1)), document.createTextNode(" / " + pad2(slides.length)));
    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === slides.length - 1;
  }
  function enterCurrent() { const s = slides[index]; s.el.classList.add("playing"); s.enter(); }
  function leaveAt(k) { const s = slides[k]; s.el.classList.remove("playing"); s.leave(); }

  function goTo(i) {
    i = clamp(i, 0, slides.length - 1);
    if (i === index) return;
    if (active) leaveAt(index);
    index = i;
    applyTransform();
    updateChrome();
    if (active) enterCurrent();
  }
  const next = () => goTo(index + 1);
  const prev = () => goTo(index - 1);

  /* ---- scale-to-fit ---- */
  function scaleToFit() {
    const w = stagewrap.clientWidth, hgt = stagewrap.clientHeight;
    if (!w || !hgt) return;
    const s = Math.min((w - 40) / STAGE_W, (hgt - 40) / STAGE_H);
    stage.style.setProperty("--deck-scale", clamp(s, 0.2, 2.4).toFixed(4));
  }
  if ("ResizeObserver" in window) new ResizeObserver(scaleToFit).observe(stagewrap);
  window.addEventListener("resize", scaleToFit);

  /* ---- overview ---- */
  function openOverview() { overviewOpen = true; overview.hidden = false; }
  function closeOverview() { overviewOpen = false; overview.hidden = true; }
  function toggleOverview() { overviewOpen ? closeOverview() : openOverview(); }

  /* ---- keyboard ---- */
  document.addEventListener("keydown", (e) => {
    if (!active) return;
    if (e.key === "Escape") { toggleOverview(); e.preventDefault(); return; }
    if (overviewOpen) return;
    if ((e.key === " " || e.key === "Enter") && e.target && e.target.tagName === "BUTTON") return;
    switch (e.key) {
      case "ArrowRight": case "PageDown": case " ": case "Spacebar": next(); e.preventDefault(); break;
      case "ArrowLeft": case "PageUp": prev(); e.preventDefault(); break;
      case "Home": goTo(0); e.preventDefault(); break;
      case "End": goTo(slides.length - 1); e.preventDefault(); break;
      case "o": case "O": toggleOverview(); e.preventDefault(); break;
    }
  });

  /* ---- swipe (touch / trackpad drag) ---- */
  let touchX = null;
  stagewrap.addEventListener("touchstart", (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  stagewrap.addEventListener("touchend", (e) => {
    if (touchX == null) return;
    const dx = e.changedTouches[0].clientX - touchX; touchX = null;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
  }, { passive: true });

  prevBtn.addEventListener("click", prev);
  nextBtn.addEventListener("click", next);

  /* ---- activate / deactivate tied to the view router ---- */
  function activate() {
    if (active) return;
    active = true;
    scaleToFit();
    requestAnimationFrame(scaleToFit);
    enterCurrent();
  }
  function deactivate() {
    if (!active) return;
    active = false;
    leaveAt(index);
    closeOverview();
  }
  document.addEventListener("viewchange", (e) => {
    if (e.detail && e.detail.view === "slides") activate(); else deactivate();
  });

  /* ---- initial paint ---- */
  // Optional deep-link to a slide, 1-based: /?s=5#slides  (handy for sharing).
  const startAt = parseInt(new URLSearchParams(location.search).get("s"), 10);
  if (Number.isFinite(startAt)) index = clamp(startAt - 1, 0, slides.length - 1);
  applyTransform();
  updateChrome();
  // If the deck view is already visible at boot (e.g. loaded via #slides), start it.
  const section = document.getElementById("view-slides");
  if (section && !section.hidden) activate();
}
