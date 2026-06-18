/* =====================================================================
   app.js — boot + view routing + shared helpers
   ---------------------------------------------------------------------
   Responsibilities:
     - Fetch /api/health and /api/config once at startup.
     - Provide a small shared toolkit (DOM helper, formatters, fallback
       config) to the two view modules via a `ctx` object — this avoids
       circular imports (app -> howto, app -> playground only).
     - Wire the top-nav tab routing between "How it works" and "Playground".
     - Render the header status / mode badge.
   The two view modules are imported and initialized AFTER config is ready
   so their UIs stay in sync with the backend.
   ===================================================================== */

import { initHowto } from "./howto.js";
import { initPlayground } from "./playground.js";
import { initSlides } from "./slides.js";

/* ------------------------------------------------------------------ *
 * Fallback config — mirrors the shared contract exactly. Used when the
 * backend is unreachable (e.g. offline compute node) so the UI still
 * works end-to-end. Any fetched config is deep-merged over this.
 * ------------------------------------------------------------------ */
export const FALLBACK_CONFIG = {
  defaults: {
    max_new_tokens: 256,
    canvas_length: 256,
    max_denoising_steps: 48,
    temperature_start: 0.8,
    temperature_end: 0.4,
    entropy_bound: 0.1,
    adaptive_stop_threshold: 0.005,
    top_k: 0,
    seed: null,
  },
  bounds: {
    max_new_tokens: { min: 32, max: 1024, step: 32 },
    canvas_length: { min: 64, max: 256, step: 32 },
    max_denoising_steps: { min: 4, max: 64, step: 1 },
    temperature_start: { min: 0.0, max: 1.5, step: 0.05 },
    temperature_end: { min: 0.0, max: 1.5, step: 0.05 },
    entropy_bound: { min: 0.0, max: 1.0, step: 0.01 },
    adaptive_stop_threshold: { min: 0.0, max: 0.05, step: 0.001 },
    top_k: { min: 0, max: 200, step: 1 },
  },
  display_defaults: {
    playback_speed_ms: 80,
    mode: "buffer-then-play",
    show_mask_glyphs: true,
    color_by_confidence: true,
    mask_glyph_style: "shades",
  },
};

/* ----------------------------- helpers ----------------------------- */

/**
 * Tiny hyperscript-style DOM builder.
 * el("div", {class:"x", onClick:fn}, child, "text", [more, children])
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k === "for") node.htmlFor = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  appendChildren(node, children);
  return node;
}

function appendChildren(node, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === true) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}

export const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

/** Format a number compactly for stats display. */
export function fmtNum(n, digits = 1) {
  if (n == null || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1000) return Math.round(n).toLocaleString();
  return Number(n).toFixed(digits).replace(/\.0$/, "");
}

/** Resilient JSON fetch with a short timeout so the UI never hangs. */
async function fetchJSON(url, timeoutMs = 4000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { headers: { accept: "application/json" }, signal: ctrl.signal });
    if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

/** Deep-merge fetched config over the fallback so missing keys are safe. */
function mergeConfig(base, override) {
  if (!override || typeof override !== "object") return base;
  const out = { ...base };
  for (const key of ["defaults", "bounds", "display_defaults"]) {
    out[key] = { ...(base[key] || {}), ...(override[key] || {}) };
    // bounds are nested objects per-param
    if (key === "bounds" && override.bounds) {
      for (const p of Object.keys(override.bounds)) {
        out.bounds[p] = { ...(base.bounds[p] || {}), ...override.bounds[p] };
      }
    }
  }
  return out;
}

/* ------------------------------ state ------------------------------ */
const state = {
  health: null,
  config: FALLBACK_CONFIG,
  mode: "unknown", // real | mock | unknown | error
  view: "howto",
};

/* --------------------------- header status -------------------------- */
function renderHeaderStatus() {
  const host = document.getElementById("header-status");
  if (!host) return;
  const h = state.health;
  const dotState = state.mode === "real" || state.mode === "mock" ? state.mode : state.mode === "error" ? "error" : "unknown";
  host.replaceChildren(
    el("span", { class: "status-dot", "data-state": dotState }),
    el("span", { class: "status-text" }, h ? (h.mode === "real" ? "Live model" : "Mock mode") : state.mode === "error" ? "Backend offline" : "connecting…"),
    h ? el("span", { class: "status-meta" }, ` · ${shortModel(h.model_id)} · ${h.device || "?"} · ${h.precision || "?"}`) : null,
    // If real mode was requested but the server fell back to mock, warn loudly.
    h && h.detail ? el("span", { class: "status-warn", title: h.detail, style: { color: "#f5b547", marginLeft: "10px", fontWeight: "600", cursor: "help" } }, "⚠ fell back to mock") : null
  );
}

function shortModel(id) {
  if (!id) return "model";
  const parts = String(id).split("/");
  return parts[parts.length - 1];
}

/* ----------------------------- routing ----------------------------- */
const VIEWS = ["howto", "playground", "slides"];

function setView(view, { updateHash = true } = {}) {
  if (!VIEWS.includes(view)) view = "howto";
  state.view = view;

  for (const tab of document.querySelectorAll(".tab")) {
    const on = tab.dataset.view === view;
    tab.setAttribute("aria-selected", on ? "true" : "false");
  }

  let panel = null;
  for (const v of VIEWS) {
    const sec = document.getElementById(`view-${v}`);
    if (!sec) continue;
    const on = v === view;
    sec.hidden = !on;
    if (on) panel = sec;
  }

  if (updateHash && location.hash.slice(1) !== view) {
    history.replaceState(null, "", `#${view}`);
  }
  // focus the panel for keyboard users (without scrolling jump on hash views)
  if (panel) panel.focus({ preventScroll: false });
  window.scrollTo({ top: 0, behavior: "auto" });

  // Let self-contained views (the slide deck) pause/resume their animations
  // instead of burning CPU while hidden.
  document.dispatchEvent(new CustomEvent("viewchange", { detail: { view } }));
}

function initRouting() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setView(tab.dataset.view));
  });
  // any element with data-goto navigates (e.g. CTA buttons inside how-to)
  document.addEventListener("click", (e) => {
    const target = e.target.closest("[data-goto]");
    if (target) {
      e.preventDefault();
      setView(target.dataset.goto);
    }
  });
  window.addEventListener("hashchange", () => setView(location.hash.slice(1), { updateHash: false }));

  const initial = location.hash.slice(1) || "howto";
  setView(initial, { updateHash: false });
}

/* ------------------------------- boot ------------------------------ */
async function boot() {
  // Build the shared context handed to each view.
  const ctx = {
    el,
    clamp,
    fmtNum,
    state,
    navigate: (v) => setView(v),
    config: state.config,
    health: null,
    mode: "unknown",
  };

  // Initialize the how-to view immediately — it is fully self-contained
  // (canned animations) and must work even with zero network.
  try {
    initHowto(document.getElementById("view-howto"), ctx);
  } catch (err) {
    console.warn("howto init failed (non-fatal):", err);
  }

  // The slide deck is also fully self-contained (canned animations + bundled
  // video) and must work with zero network.
  try {
    initSlides(document.getElementById("slides-root"), ctx);
  } catch (err) {
    console.warn("slides init failed (non-fatal):", err);
  }

  // Fetch backend info in parallel; tolerate failure.
  const [health, config] = await Promise.allSettled([
    fetchJSON("/api/health"),
    fetchJSON("/api/config"),
  ]);

  if (health.status === "fulfilled") {
    state.health = health.value;
    state.mode = health.value.mode === "real" ? "real" : "mock";
  } else {
    state.mode = "error";
    console.warn("/api/health unavailable — using fallback/mock UI:", health.reason);
  }

  if (config.status === "fulfilled") {
    state.config = mergeConfig(FALLBACK_CONFIG, config.value);
  } else {
    console.warn("/api/config unavailable — using fallback config:", config.reason);
  }

  ctx.config = state.config;
  ctx.health = state.health;
  ctx.mode = state.mode;

  renderHeaderStatus();

  // Build the playground once config is known.
  try {
    initPlayground(document.getElementById("pg-root"), ctx);
  } catch (err) {
    console.error("playground init failed:", err);
    const root = document.getElementById("pg-root");
    if (root) root.replaceChildren(el("div", { class: "error-frame" }, el("strong", {}, "Playground failed to initialize. "), String(err && err.message ? err.message : err)));
  }
}

/* Module scripts are deferred, so the DOM is ready when this runs, but
   guard anyway for safety. */
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    initRouting();
    boot();
  });
} else {
  initRouting();
  boot();
}
