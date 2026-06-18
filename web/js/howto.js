/* =====================================================================
   howto.js — the scrollable explainer
   ---------------------------------------------------------------------
   - IntersectionObserver scroll-reveal for [data-reveal] sections.
   - Small, fully self-contained "denoising" animations (no server):
       * hero line resolving from noise
       * autoregressive (typewriter) vs diffusion (canvas) side-by-side
       * a 256-token-style mini canvas denoising into a paragraph
   These reuse the same visual language as the Playground for cohesion.
   ===================================================================== */

const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Glyph sets used to draw "noise" for unresolved tokens.
const GLYPHS = {
  shades: "░▒▓",
  blocks: "█▉▊▋",
  dots: "·•∙",
  braille: "⠁⠂⠄⡀⢀⠠⠐⠈",
  ascii: "#%@&$?*",
};

function pick(str) {
  return str[(Math.random() * str.length) | 0];
}

/** Build a short run of noise glyphs to stand in for an unresolved token. */
function noiseRun(len = 2, style = "shades") {
  const set = GLYPHS[style] || GLYPHS.shades;
  let s = "";
  for (let i = 0; i < len; i++) s += pick(set);
  return s;
}

/** Split text into word-ish tokens (keeping trailing whitespace attached). */
function tokenize(text) {
  return text.match(/\S+\s*|\s+/g) || [text];
}

/* ------------------------------------------------------------------ *
 * Resolver: animates a target string emerging from noise inside `host`.
 *
 *   order: "diffusion" -> tokens commit in confidence (random-ish) order,
 *                         the whole block visible as noise the entire time.
 *          "linear"    -> tokens commit strictly left-to-right (typewriter),
 *                         only revealed-so-far is shown (AR style).
 *
 * Returns a handle with .stop().
 * ------------------------------------------------------------------ */
function makeResolver(host, text, opts = {}) {
  const order = opts.order || "diffusion";
  const steps = opts.steps || 22;
  const frameMs = opts.frameMs || 90;
  const loop = opts.loop !== false;
  const holdMs = opts.holdMs || 1800;
  const glyph = opts.glyph || "shades";

  const tokens = tokenize(text);
  const n = tokens.length;

  // commitStep[i] = the step at which token i becomes final.
  const commitStep = new Array(n);
  if (order === "linear") {
    for (let i = 0; i < n; i++) commitStep[i] = i; // one per frame, L->R
  } else {
    // Spread tokens across `steps` passes; whitespace commits with neighbours.
    const idx = [...Array(n).keys()];
    for (let i = idx.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    idx.forEach((tokenIndex, rank) => {
      commitStep[tokenIndex] = Math.floor((rank / n) * (steps - 2)) + 1;
    });
  }
  const totalSteps = order === "linear" ? n : steps;

  // pre-create spans once
  host.replaceChildren(...tokens.map(() => document.createElement("span")));
  const spans = host.children;

  let step = 0;
  let timer = null;
  let stopped = false;
  let flickerTimer = null;

  function renderStep(s) {
    for (let i = 0; i < n; i++) {
      const span = spans[i];
      const committed = s >= commitStep[i];
      if (order === "linear") {
        // typewriter: hide everything past the cursor
        if (i < s) {
          span.className = "tok tok-committed";
          span.textContent = tokens[i];
        } else if (i === s) {
          span.className = "tok tok-tentative";
          span.textContent = tokens[i];
        } else {
          span.className = "";
          span.textContent = "";
        }
      } else {
        if (committed) {
          span.className = "tok tok-committed";
          span.textContent = tokens[i];
        } else {
          span.className = "tok tok-mask";
          span.dataset.noise = "1";
          // whitespace tokens stay as their own width
          span.textContent = /^\s+$/.test(tokens[i]) ? tokens[i] : noiseRun(Math.min(3, Math.max(1, tokens[i].trim().length)), glyph);
        }
      }
    }
    if (opts.onStep) opts.onStep(s, totalSteps);
  }

  function tick() {
    if (stopped) return;
    renderStep(step);
    if (step >= totalSteps) {
      if (loop && !REDUCED_MOTION) {
        timer = setTimeout(() => {
          step = 0;
          tick();
        }, holdMs);
      }
      return;
    }
    step++;
    timer = setTimeout(tick, frameMs);
  }

  // Flicker the noise glyphs so the unresolved field looks alive.
  if (!REDUCED_MOTION && order !== "linear") {
    flickerTimer = setInterval(() => {
      for (const span of host.querySelectorAll('.tok-mask[data-noise="1"]')) {
        if (/^\s+$/.test(span.textContent)) continue;
        span.textContent = noiseRun(span.textContent.length, glyph);
      }
    }, 110);
  }

  if (REDUCED_MOTION) {
    // Just show the final text, no animation.
    for (let i = 0; i < n; i++) {
      spans[i].className = "tok tok-committed";
      spans[i].textContent = tokens[i];
    }
    if (opts.onStep) opts.onStep(totalSteps, totalSteps);
  } else {
    tick();
  }

  return {
    stop() {
      stopped = true;
      clearTimeout(timer);
      clearInterval(flickerTimer);
    },
  };
}

/* ------------------------------------------------------------------ *
 * Public init
 * ------------------------------------------------------------------ */
export function initHowto(root, ctx) {
  if (!root) return;

  setupReveal(root);
  setupHero();
  setupComparison();
  setupMiniDenoise();
}

/* scroll-reveal: add .revealed when a section enters the viewport */
function setupReveal(root) {
  const targets = root.querySelectorAll("[data-reveal]");
  if (!("IntersectionObserver" in window)) {
    targets.forEach((t) => t.classList.add("revealed"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("revealed");
          io.unobserve(e.target);
        }
      }
    },
    { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
  );
  targets.forEach((t) => io.observe(t));
}

/* hero: one evocative line resolving from noise, looping slowly */
function setupHero() {
  const host = document.getElementById("hero-resolve");
  if (!host) return;
  makeResolver(host, "the sky is blue because air scatters short blue wavelengths most.", {
    order: "diffusion",
    steps: 20,
    frameMs: 95,
    holdMs: 2600,
    glyph: "shades",
  });
}

/* side-by-side: same sentence, two generation styles */
function setupComparison() {
  const ar = document.getElementById("cmp-ar");
  const diff = document.getElementById("cmp-diff");
  const sentence = "Diffusion lets the model revise its own draft.";
  if (ar) {
    makeResolver(ar, sentence, { order: "linear", frameMs: 95, holdMs: 2200 });
  }
  if (diff) {
    makeResolver(diff, sentence, { order: "diffusion", steps: 16, frameMs: 110, holdMs: 2200, glyph: "shades" });
  }
}

/* mini 256-token-style canvas denoising into a paragraph */
function setupMiniDenoise() {
  const host = document.getElementById("mini-denoise");
  const stepLabel = document.getElementById("mini-step");
  if (!host) return;
  const para =
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen. " +
    "Chlorophyll in the leaves absorbs light, driving the reactions that store energy in sugar.";
  makeResolver(host, para, {
    order: "diffusion",
    steps: 40,
    frameMs: 85,
    holdMs: 2800,
    glyph: "shades",
    onStep(s, total) {
      if (stepLabel) stepLabel.textContent = `step ${Math.min(s, total)} / ${total}`;
    },
  });
}
