import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-condensed/400.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "./style.css";

import {
  VIEWS, clamp, fromByte, screenToVoxel, sliceProfile, voxelToMm, voxelToScreen,
} from "./geometry.mjs";
import { SliceRenderer } from "./renderer.js";
import { MD_UNIT, esc, known, reportHtml, scanAsText, scansAsText } from "./report.mjs";
import { disposeSubject, loadIndex, loadSubject, loadSummary } from "./volume.js";

const $ = (selector) => document.querySelector(selector);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const NARROW = 900;
const DEFAULTS = { level: 0.5, width: 0.5 };
// the contrast slider runs from flat (the whole grey range on screen) to steep
const widthFromSlider = (value) => 1.5 - value;
// which way each axis runs, low index first
const AXIS_ENDS = { 0: ["L", "R"], 1: ["P", "A"], 2: ["I", "S"] };
const MAPS = {
  fa: { short: "FA", name: "Fractional anisotropy", hint: "Black is 0, white is 1. White matter is bright." },
  md: { short: "MD", name: "Mean diffusivity", hint: `Black is 0, white is 3 ${MD_UNIT}. Fluid is bright.` },
};
const FULL_NAMES = { ATR: "anterior thalamic radiation", SLF: "superior longitudinal fasciculus" };
const TRACT_COLOURS = ["--tract-1", "--tract-2"];

const state = {
  voxel: [0, 0, 0],
  level: DEFAULTS.level,
  width: widthFromSlider(DEFAULTS.width),
  opacity: 0.5,
  map: "fa",
  show: [true, true],   // one switch per tract, in the order the scan lists them
  outline: true,
  layout: "all",
  active: "axial",   // the view the slice scrubber drives
  place: -1,
};

const app = $("#app");
const panesEl = $("#panes");
const panes = Object.fromEntries(
  [...document.querySelectorAll(".pane")].map((el) => [el.dataset.view, el]),
);
const track = $("#profile");
const trackCanvas = track.querySelector("canvas");
const notice = $("#notice");

let renderer = null;
let subject = null;
let profiles = {};
let loadToken = 0;
let ids = [];
let measurements = [];
let summary = null;

// ---- drawing -------------------------------------------------------------------

// draw once on the next animation frame, however many things changed before it.
// a tab in the background gets no animation frames, so a short timer backs it up
let scheduled = false;

function requestDraw() {
  if (scheduled) return;
  scheduled = true;
  const run = () => {
    if (!scheduled) return;
    scheduled = false;
    if (!renderer || !subject) return;
    renderer.draw(panes, state);
    drawOverlays();
    drawScrubber();
    drawHere();
  };
  requestAnimationFrame(run);
  setTimeout(run, 150);
}

function drawOverlays() {
  const dims = subject.meta.dims;
  for (const [name, pane] of Object.entries(panes)) {
    const view = VIEWS[name];
    const rect = renderer.sliceRect(name, pane);
    const at = voxelToScreen(name, state.voxel, dims);
    const cx = rect.x + at.fx * rect.w;
    const cy = rect.y + at.fy * rect.h;
    pane.style.setProperty("--cx", `${cx}px`);
    pane.style.setProperty("--cy", `${cy}px`);
    pane.querySelector(".cross--h").style.top = `${cy}px`;
    pane.querySelector(".cross--v").style.left = `${cx}px`;
    pane.querySelector(".pane__slice").textContent = `${state.voxel[view.axis] + 1} of ${dims[view.axis]}`;
    pane.classList.toggle("is-active", name === state.active);
  }
}

const tractCodes = () => Object.keys(subject.meta.tracts ?? {}).map(Number).sort((a, b) => a - b);

// per slice along one axis, how many voxels of each tract it holds
function tractProfile(axis) {
  if (!profiles[axis]) {
    profiles[axis] = tractCodes().map(
      (code) => sliceProfile(subject.bytes.tracts, subject.meta.dims, axis, code),
    );
  }
  return profiles[axis];
}

// the scrubber's track is a small chart: one bar per slice, as tall as the tracts in it,
// each tract in its own colour, one stacked on the other
function drawScrubber() {
  const axis = VIEWS[state.active].axis;
  const series = subject.bytes.tracts ? tractProfile(axis) : [];
  const n = subject.meta.dims[axis];
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const w = trackCanvas.clientWidth;
  const h = trackCanvas.clientHeight;
  if (trackCanvas.width !== Math.round(w * ratio) || trackCanvas.height !== Math.round(h * ratio)) {
    trackCanvas.width = Math.round(w * ratio);
    trackCanvas.height = Math.round(h * ratio);
  }
  const ctx = trackCanvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const column = w / n;
  const base = h - 5;
  ctx.fillStyle = css("--line-strong");
  ctx.fillRect(0, base, w, 1);

  let peak = 1;
  for (let i = 0; i < n; i++) peak = Math.max(peak, series.reduce((sum, counts) => sum + counts[i], 0));
  let here = 0;
  for (let i = 0; i < n; i++) {
    let top = base;
    series.forEach((counts, k) => {
      if (!counts[i]) return;
      const height = Math.max(1.5, (counts[i] / peak) * (base - 3));
      ctx.fillStyle = css(TRACT_COLOURS[k % TRACT_COLOURS.length]);
      ctx.fillRect(i * column, top - height, Math.max(column - 0.5, 1), height);
      top -= height;
      if (i === state.voxel[axis]) here += counts[i];
    });
  }

  const x = (state.voxel[axis] + 0.5) * column;
  ctx.fillStyle = css("--cross");
  ctx.fillRect(Math.round(x) - 1, 0, 2, h);
  ctx.beginPath();
  ctx.arc(Math.round(x), base + 1, 4, 0, Math.PI * 2);
  ctx.fill();

  const [low, high] = AXIS_ENDS[axis];
  $("#scrub-low").textContent = low;
  $("#scrub-high").textContent = high;
  $("#scrub-where").innerHTML = `<b>${state.voxel[axis] + 1}</b> of ${n}`;
  track.setAttribute("aria-valuemin", "1");
  track.setAttribute("aria-valuemax", String(n));
  track.setAttribute("aria-valuenow", String(state.voxel[axis] + 1));
  track.setAttribute(
    "aria-valuetext",
    `${state.active} slice ${state.voxel[axis] + 1} of ${n}, ${here ? "a tract crosses this slice" : "no tract in this slice"}`,
  );
}

function valueAt(name, voxel) {
  const [nx, ny] = subject.meta.dims;
  const bytes = subject.bytes[name];
  return bytes ? bytes[voxel[0] + nx * (voxel[1] + ny * voxel[2])] : 0;
}

function tractTag(code) {
  const name = subject.meta.tracts?.[String(code)];
  if (!name) return "";
  return `<span class="tag tag--tract" style="--tint:var(${TRACT_COLOURS[(code - 1) % TRACT_COLOURS.length]})">${esc(name)}</span>`;
}

function drawHere() {
  const { dims, voxel_mm: mm, white } = subject.meta;
  const [r, a, s] = voxelToMm(state.voxel, dims, mm);
  const side = (value, plus, minus) => `${value >= 0 ? plus : minus} ${Math.abs(value).toFixed(0)}`;
  const tag = tractTag(valueAt("tracts", state.voxel));
  $("#here").innerHTML = [
    ["Position", `${side(r, "R", "L")}, ${side(a, "A", "P")}, ${side(s, "S", "I")} mm`],
    ["FA", fromByte(valueAt("fa", state.voxel), white?.fa ?? 1).toFixed(2)],
    ["MD", `${fromByte(valueAt("md", state.voxel), white?.md ?? 3).toFixed(2)} <small>${MD_UNIT}</small>`],
    ["Inside", tag || "no tract"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

// ---- moving around -------------------------------------------------------------

function moveTo(voxel) {
  if (!voxel) return;
  const dims = subject.meta.dims;
  state.voxel = voxel.map((v, i) => clamp(Math.round(v), 0, dims[i] - 1));
  requestDraw();
}

function step(axis, amount) {
  const next = state.voxel.slice();
  next[axis] += amount;
  moveTo(next);
}

function setActive(name) {
  if (state.active === name) return;
  state.active = name;
  requestDraw();
}

function goToPlace(index) {
  const places = subject.meta.places ?? [];
  if (!places.length) return;
  state.place = (index + places.length) % places.length;
  moveTo(places[state.place].voxel);
  document.querySelectorAll("#places button").forEach((button, i) => {
    button.setAttribute("aria-current", String(i === state.place));
  });
}

function pointInPane(name, event) {
  const pane = panes[name];
  const box = pane.getBoundingClientRect();
  const rect = renderer.sliceRect(name, pane);
  if (rect.w < 1) return null;
  const fx = clamp((event.clientX - box.left - rect.x) / rect.w, 0, 1);
  const fy = clamp((event.clientY - box.top - rect.y) / rect.h, 0, 1);
  return screenToVoxel(name, fx, fy, state.voxel, subject.meta.dims);
}

// keep receiving the pointer while it is dragged outside the element
function capture(element, event) {
  try {
    element.setPointerCapture(event.pointerId);
  } catch {
    // a pointer that is already gone cannot be captured, and the drag simply ends early
  }
}

function bindPane(name, pane) {
  const view = VIEWS[name];
  let dragging = false;

  pane.addEventListener("pointerdown", (event) => {
    if (!subject) return;
    dragging = true;
    capture(pane, event);
    setActive(name);
    moveTo(pointInPane(name, event));
  });
  pane.addEventListener("pointermove", (event) => {
    if (dragging && subject) moveTo(pointInPane(name, event));
  });
  const stop = () => { dragging = false; };
  pane.addEventListener("pointerup", stop);
  pane.addEventListener("pointercancel", stop);
  pane.addEventListener("focus", () => setActive(name));

  pane.addEventListener("wheel", (event) => {
    if (!subject) return;
    event.preventDefault();
    setActive(name);
    step(view.axis, event.deltaY < 0 ? 1 : -1);
  }, { passive: false });

  pane.addEventListener("keydown", (event) => {
    if (!subject) return;
    const inPlane = {
      ArrowLeft: [view.h, view.flipH ? 1 : -1], ArrowRight: [view.h, view.flipH ? -1 : 1],
      ArrowUp: [view.v, 1], ArrowDown: [view.v, -1],
    };
    const through = { ArrowUp: 1, ArrowDown: -1, PageUp: 10, PageDown: -10 };
    if (event.shiftKey && inPlane[event.key]) step(...inPlane[event.key]);
    else if (through[event.key]) step(view.axis, through[event.key]);
    else if (event.key === "Home") step(view.axis, -1e6);
    else if (event.key === "End") step(view.axis, 1e6);
    else return;
    event.preventDefault();
  });
}

function bindScrubber() {
  const axis = () => VIEWS[state.active].axis;
  let dragging = false;
  const seek = (event) => {
    const box = trackCanvas.getBoundingClientRect();
    const fraction = clamp((event.clientX - box.left) / box.width, 0, 1);
    const next = state.voxel.slice();
    next[axis()] = Math.floor(fraction * subject.meta.dims[axis()]);
    moveTo(next);
  };
  track.addEventListener("pointerdown", (event) => {
    if (!subject) return;
    dragging = true;
    capture(track, event);
    seek(event);
  });
  track.addEventListener("pointermove", (event) => { if (dragging && subject) seek(event); });
  track.addEventListener("pointerup", () => { dragging = false; });
  track.addEventListener("pointercancel", () => { dragging = false; });
  track.addEventListener("keydown", (event) => {
    if (!subject) return;
    const amount = {
      ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1,
      PageUp: 10, PageDown: -10, Home: -1e6, End: 1e6,
    }[event.key];
    if (!amount) return;
    event.preventDefault();
    step(axis(), amount);
  });
  $("#slice-back").addEventListener("click", () => subject && step(axis(), -1));
  $("#slice-forward").addEventListener("click", () => subject && step(axis(), 1));
}

// ---- layout: which views, and the two side panels ---------------------------------

function setLayout(layout) {
  state.layout = layout;
  panesEl.dataset.layout = layout;
  if (layout !== "all") state.active = layout;
  requestDraw();
}

function setPanel(side, open) {
  const hidden = `${side}-hidden`;
  app.classList.toggle(hidden, !open);
  $(`#toggle-${side}`).setAttribute("aria-expanded", String(open));
  // the canvas has to follow the stage as the panel slides
  const start = performance.now();
  const follow = () => {
    requestDraw();
    if (performance.now() - start < 320) setTimeout(follow, 40);
  };
  follow();
}

function bindLayout() {
  document.querySelectorAll('input[name="layout"]').forEach((radio) => {
    radio.addEventListener("change", () => { if (radio.checked) setLayout(radio.value); });
  });
  for (const side of ["controls", "readout"]) {
    $(`#toggle-${side}`).addEventListener("click", () => {
      const open = app.classList.contains(`${side}-hidden`);
      // on a narrow screen the panels are drawers, and only one is out at a time
      if (open && window.innerWidth <= NARROW) setPanel(side === "controls" ? "readout" : "controls", false);
      setPanel(side, open);
    });
  }
  const embedded = new URL(window.location.href).searchParams.get("embed") === "1";
  if (embedded || window.innerWidth <= NARROW) {
    setPanel("controls", false);
    setPanel("readout", false);
  }
  // from here on the panels may slide
  setTimeout(() => app.classList.remove("is-starting"), 60);
}

// ---- scans or group results ---------------------------------------------------------

async function copyText(text, status) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // older browsers, and pages not served over https: select and copy the long way
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  status.textContent = "Copied";
  setTimeout(() => { status.textContent = ""; }, 2000);
}

function setMode(mode) {
  const results = mode === "results";
  app.classList.toggle("is-results", results);
  $("#report").hidden = !results;
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  if (!results) requestDraw();
}

function fillReport() {
  const run = summary?.run;
  $("#report-meta").textContent = run
    ? `dtic ${run.versions?.dtic ?? ""}, DIPY ${run.versions?.dipy ?? ""}, finished ${(run.finished ?? "").slice(0, 10)}`
    : "";
  $("#report-body").innerHTML = reportHtml(summary, measurements);
  $("#copy-scans").hidden = !measurements.length;
  $("#copy-summary").hidden = !summary?.markdown;
}

function bindReport() {
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener("change", () => { if (radio.checked) setMode(radio.value); });
  });
  $("#copy-scans").addEventListener("click", () => copyText(scansAsText(measurements), $("#report-copied")));
  $("#copy-summary").addEventListener("click", () => copyText(summary.markdown, $("#report-copied")));
  $("#copy-scan").addEventListener("click", () => {
    if (subject) copyText(scanAsText(subject.meta), $("#scan-copied"));
  });
  // a scan's name in the table opens that scan
  $("#report-body").addEventListener("click", (event) => {
    const button = event.target instanceof Element ? event.target.closest("button[data-scan]") : null;
    if (!button) return;
    setMode("scans");
    openScan(button.dataset.scan);
  });
}

// ---- the control panel -----------------------------------------------------------

function setMap(map) {
  state.map = map;
  document.querySelector(`input[name="map"][value="${map}"]`).checked = true;
  $("#map-hint").textContent = `${MAPS[map].name}. ${MAPS[map].hint}`;
  $("#map-badge").textContent = MAPS[map].short;
  requestDraw();
}

function openScan(id) {
  if (!ids.includes(id)) return;
  $("#scan").value = id;
  show(id);
}

function stepScan(amount) {
  if (!ids.length) return;
  const at = ids.indexOf($("#scan").value);
  openScan(ids[(at + amount + ids.length) % ids.length]);
}

function bindControls() {
  $("#tract-rows").addEventListener("change", (event) => {
    const box = event.target instanceof Element ? event.target.closest("input[data-tract]") : null;
    if (!box) return;
    state.show[Number(box.dataset.tract)] = box.checked;
    requestDraw();
  });
  $("#show-cross").addEventListener("change", (event) => {
    panesEl.classList.toggle("no-cross", !event.target.checked);
  });
  document.querySelectorAll('input[name="map"]').forEach((radio) => {
    radio.addEventListener("change", () => { if (radio.checked) setMap(radio.value); });
  });
  document.querySelectorAll('input[name="style"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.outline = document.querySelector('input[name="style"]:checked').value === "outline";
      requestDraw();
    });
  });
  $("#opacity").addEventListener("input", (event) => {
    state.opacity = Number(event.target.value);
    $("#opacity-out").textContent = `${Math.round(state.opacity * 100)}%`;
    requestDraw();
  });
  const image = () => {
    state.level = 1 - Number($("#level").value);
    state.width = widthFromSlider(Number($("#width").value));
    $("#level-out").textContent = Math.round(Number($("#level").value) * 100);
    $("#width-out").textContent = Math.round(Number($("#width").value) * 100);
    requestDraw();
  };
  $("#level").addEventListener("input", image);
  $("#width").addEventListener("input", image);
  $("#reset-image").addEventListener("click", () => {
    $("#level").value = DEFAULTS.level;
    $("#width").value = DEFAULTS.width;
    image();
  });
  $("#prev-place").addEventListener("click", () => goToPlace(state.place - 1));
  $("#next-place").addEventListener("click", () => goToPlace(state.place + 1));
  $("#scan").addEventListener("change", (event) => show(event.target.value));
  $("#prev-scan").addEventListener("click", () => stepScan(-1));
  $("#next-scan").addEventListener("click", () => stepScan(1));

  document.addEventListener("keydown", (event) => {
    if (!subject || event.ctrlKey || event.metaKey || event.altKey) return;
    if (app.classList.contains("is-results")) return;
    if (event.target instanceof Element && event.target.closest("input, select, textarea")) return;
    const key = event.key.toLowerCase();
    if (key === "[") goToPlace(state.place - 1);
    else if (key === "]") goToPlace(state.place + 1);
    else if (key === ",") stepScan(-1);
    else if (key === ".") stepScan(1);
    else if (key === "m") setMap(state.map === "fa" ? "md" : "fa");
    else if (key === "t") {
      // all on unless all are on already
      const on = !state.show.every(Boolean);
      document.querySelectorAll("#tract-rows input[data-tract]").forEach((box) => {
        box.checked = on;
        state.show[Number(box.dataset.tract)] = on;
      });
      requestDraw();
    } else if (key === "o") {
      const other = document.querySelector('input[name="style"]:not(:checked)');
      other.checked = true;
      other.dispatchEvent(new Event("change"));
    }
  });
}

// ---- the measurements panel ---------------------------------------------------------

const fixed = (value, digits) => (known(value) ? value.toFixed(digits) : "–");
const tint = (index) => `var(${TRACT_COLOURS[index % TRACT_COLOURS.length]})`;

function fillReadout(meta) {
  const s = meta.stats ?? {};
  const tracts = Object.entries(s.tracts ?? {});
  const failed = s.reg_ok === false;

  // the four numbers the study is about: FA and MD in each tested tract
  $("#tract-cards").innerHTML = tracts.map(([name, v], i) => `
    <li style="--tint:${tint(i)}">
      <p class="tracts__name"><b>${esc(name)}</b> ${esc(FULL_NAMES[name] ?? "")}</p>
      <p class="tracts__value"><span>FA</span><b>${fixed(v.fa, 3)}</b></p>
      <p class="tracts__value"><span>MD</span><b>${fixed(v.md, 3)}</b><small>${MD_UNIT}</small></p>
    </li>`).join("");
  const rule = meta.min_fa > 0
    ? `the voxels inside each outline with FA of at least ${meta.min_fa}`
    : "every voxel inside each outline";
  $("#hero-note").textContent = failed
    ? "The registration check failed for this scan, so the tracts are not trusted and its values are left out of the tests."
    : `Each value is the mean over ${rule}, left and right sides together.`;
  $("#hero-note").classList.toggle("is-warning", failed);

  const parts = Object.entries(s.parts ?? {}).filter(([, v]) => known(v.fa) || known(v.md));
  $("#parts-group").hidden = !parts.length;
  $("#parts").innerHTML = `<thead><tr><th scope="col">Part</th><th scope="col">FA</th><th scope="col">MD</th></tr></thead>
    <tbody>${parts.map(([name, v]) => `<tr><th scope="row">${esc(name.replace(/_/g, " ")
    .replace(/ L$/, " left").replace(/ R$/, " right"))}</th><td>${fixed(v.fa, 3)}</td><td>${fixed(v.md, 3)}</td></tr>`).join("")}</tbody>`;

  const facts = [];
  if (known(s.n_directions)) {
    facts.push(["Directions", `${s.n_directions} at b = ${s.b_value}`]);
    facts.push(["b0 volumes", `${s.n_b0}`]);
    if (known(s.acq_as_expected)) {
      facts.push(["Protocol check", s.acq_as_expected
        ? '<span class="tag tag--plain">as expected</span>' : '<span class="tag tag--warn">differs</span>']);
    }
  }
  if (s.voxel_mm) facts.push(["Voxel", `${esc(s.voxel_mm)} mm`]);
  if (known(s.motion_mean_mm)) {
    facts.push(["Head motion", `${fixed(s.motion_mean_mm, 1)} mm <small>largest ${fixed(s.motion_max_mm, 1)}</small>`]);
  }
  if (known(s.reg_similarity)) {
    facts.push(["Registration check", `${fixed(s.reg_similarity, 2)}`
      + (failed ? '<span class="tag tag--warn">failed</span>' : '<span class="tag tag--plain">passed</span>')]);
  }
  tracts.forEach(([name, v]) => {
    if (known(v.voxels)) {
      facts.push([`${esc(name)} voxels used`,
        `${v.voxels}${known(v.kept_pct) ? ` <small>${fixed(v.kept_pct, 0)}% of the tract</small>` : ""}`]);
    }
  });
  $("#facts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  // one switch per tract, in its own colour
  const names = Object.entries(meta.tracts ?? {}).sort(([a], [b]) => Number(a) - Number(b));
  $("#tract-rows").innerHTML = names.map(([, name], i) => `
    <label class="check">
      <input type="checkbox" data-tract="${i}" ${state.show[i] ? "checked" : ""} />
      <span class="swatch swatch--tract" style="--tint:${tint(i)}" aria-hidden="true"></span>
      <span class="check__text">${esc(name)} <small>${esc(FULL_NAMES[name] ?? "")}</small></span>
    </label>`).join("");

  const places = meta.places ?? [];
  const order = names.map(([, name]) => name);
  $("#places").innerHTML = places.length ? "" : '<li class="places__none">No tract was placed on this scan.</li>';
  places.forEach((place, i) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.style.setProperty("--tint", tint(Math.max(order.indexOf(place.tract), 0)));
    button.innerHTML = `<span class="dot" aria-hidden="true"></span><span>${esc(place.tract)}</span>`
      + `<span class="side">${esc(place.side)}</span>`;
    button.addEventListener("click", () => goToPlace(i));
    item.append(button);
    $("#places").append(item);
  });
  $("#prev-place").disabled = $("#next-place").disabled = !places.length;
  $("#attribution").textContent = meta.attribution || "";
}

function say(html, retry) {
  notice.hidden = !html;
  notice.innerHTML = html ? `<p>${html}</p>` : "";
  if (html && retry) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Try again";
    button.addEventListener("click", retry);
    notice.append(button);
  }
}

async function show(id) {
  const token = ++loadToken;
  say("Loading the scan.");
  try {
    const next = await loadSubject(id);
    if (token !== loadToken) { disposeSubject(next); return; }   // a newer choice won
    disposeSubject(subject);
    subject = next;
    profiles = {};
    state.place = -1;
    renderer.setSubject(subject);
    fillReadout(subject.meta);
    $("#scan-count").textContent = `${ids.indexOf(id) + 1} of ${ids.length}`;
    for (const [name, pane] of Object.entries(panes)) {
      for (const [edge, letter] of Object.entries(VIEWS[name].labels)) {
        pane.querySelector(`.mark--${edge}`).textContent = letter;
      }
    }
    const places = subject.meta.places ?? [];
    if (places.length) goToPlace(0);
    else moveTo(subject.meta.dims.map((n) => n / 2));
    const url = new URL(window.location.href);
    url.searchParams.set("scan", id);
    window.history.replaceState(null, "", url);
    say("");
  } catch (error) {
    if (token === loadToken) say(`The scan could not be loaded. ${esc(error.message)}`, () => show(id));
  }
}

async function start() {
  try {
    renderer = new SliceRenderer($("#gl"), { tracts: TRACT_COLOURS.map(css) });
  } catch (error) {
    say("This browser cannot draw the images: it needs WebGL 2.");
    return;
  }
  Object.entries(panes).forEach(([name, pane]) => bindPane(name, pane));
  bindScrubber();
  bindControls();
  bindLayout();
  bindReport();
  setMap(state.map);
  new ResizeObserver(requestDraw).observe($("#stage"));
  window.addEventListener("resize", requestDraw);

  try {
    const index = await loadIndex();
    ids = index.subjects;
    measurements = index.measurements;
    summary = await loadSummary();
  } catch (error) {
    ids = [];
  }
  fillReport();
  if (!ids.length) {
    say("There are no scans to show yet. Run <code>dtic run</code>, then <code>dtic view</code>.");
    return;
  }
  const select = $("#scan");
  ids.forEach((id) => select.add(new Option(id, id)));
  $("#prev-scan").disabled = $("#next-scan").disabled = ids.length < 2;
  const wanted = new URL(window.location.href).searchParams.get("scan");
  select.value = ids.includes(wanted) ? wanted : ids[0];
  show(select.value);
}

start();
