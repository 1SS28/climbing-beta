import { load as loadDetector } from "./detect.js";

/* Everything works in a downscaled copy of the photo rather than its full
   resolution. iOS caps canvas area near 16 MP and phone photos are 12 to 48,
   so a 24 MP shot silently fails to draw. The homography only needs the
   reference points and the measured points to share one pixel space, so a
   smaller space costs no accuracy: the bolt block still spans hundreds of
   pixels. */
const WORK = 1600;

const $ = (s) => document.querySelector(s);
const canvas = $("#c");
const ctx = canvas.getContext("2d");

let photo = null;          // the downscaled bitmap everything is measured in
let holds = [];            // detected, or [] when detection is unavailable
let picks = [];            // [{xy, hold}] in the order the climber tapped
let refs = [];             // calibration points
let gaps = [];             // the last result, drawn on the photo
let mode = "select";
let moving = null;
let busy = false;
let revision = 0;
let view = 1;
let detector = null;
let last = null;           // the last measurement, as the API returned it
let tape = [];             // what the climber's tape measure read, per gap, in metres
let shown = null;          // the step last brought into view
let climb = null;          // the beta the API found, with every frame of it
let route = [];            // the hold ids the beta was asked for, in pick order
let frame = 0;
let playing = false;
let timer = null;
let recording = null;      // the video being saved while the beta plays
let pen = null;            // line scale while recording, so it does not follow the screen

const LIMB_NAME = { LH: "Left hand", RH: "Right hand", LF: "Left foot", RF: "Right foot" };
const CHAINS = { LH: ["shoulder_l", "elbow_l", "LH"], RH: ["shoulder_r", "elbow_r", "RH"],
                 LF: ["hip_l", "knee_l", "ankle_l", "LF"], RF: ["hip_r", "knee_r", "ankle_r", "RF"] };

const PER_UNIT = { in: 0.0254, cm: 0.01 };
// Route, calibration and distance each get their own colour on the photo.
const COLOUR = { route: "#3ddc97", routeInk: "#052417", ref: "#b794ff", gap: "#ffc65c",
                 gapInk: "#2a1d00", shadow: "rgba(8,11,15,.9)" };

// What the browser can record a canvas as, best first. Most video tools take
// MP4; Firefox only writes WebM.
const VIDEO = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
const videoType = () => (window.MediaRecorder && canvas.captureStream
  ? VIDEO.find((t) => MediaRecorder.isTypeSupported(t)) : undefined);

const checked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
const method = () => checked("method");
const unit = () => checked("unit");
const refsNeeded = () => (method() === "grid" ? 4 : 2);
const number = (id) => {
  const v = +$(id).value;
  return Number.isFinite(v) && v > 0 ? v : null;
};

function calibrated() {
  if (refs.length !== refsNeeded()) return false;
  if (method() === "reference") return number("#metres") !== null;
  return number("#spacing") !== null && Number.isInteger(number("#cols")) &&
         Number.isInteger(number("#rows"));
}

function update() {
  const live = !!photo && !busy && !recording;
  for (const id of ["calibrate", "select", "manual"]) $("#" + id).disabled = !live;
  $("#file").disabled = busy || !!recording;
  $("#undo").disabled = $("#clear").disabled = !live || !picks.length;
  $("#undo-ref").disabled = !live || !refs.length;
  $("#measure").disabled = !live || picks.length < 2 || !calibrated();
  $("#beta-card").hidden = !last || picks.length < 3;
  $("#beta").disabled = !live;
  for (const id of ["prev", "play", "next", "scrub", "height"]) $("#" + id).disabled = !!recording;
  $("#video").disabled = !!recording;
  $("#video").textContent = recording ? "Recording…" : "Save as video";
  $("#video-row").hidden = !videoType();
  $("#beta-status").textContent = climb ? `${climb.moves.length} moves` : "";
  $("#select").setAttribute("aria-pressed", mode === "select");
  $("#manual").setAttribute("aria-pressed", mode === "manual");
  $("#cal-status").textContent = `${refs.length} of ${refsNeeded()}`;
  $("#route-status").textContent = picks.length ? `${picks.length} picked` : "";

  // One step at a time is highlighted: the first that is not done yet.
  let current = null;
  for (const [id, done] of [["step-photo", !!photo], ["step-scale", calibrated()],
                            ["step-route", picks.length >= 2 && !!last]]) {
    const state = done ? "done" : current ? "todo" : "active";
    if (state === "active") current = id;
    $("#" + id).dataset.state = state;
  }
  // The sheet is short on a phone, so bring the next step into it, but only
  // when the step changes, never while the climber is scrolling it themselves.
  if (current && current !== shown) {
    const panel = $("#panel"), card = $("#" + current);
    if (shown) panel.scrollTo({ top: card.offsetTop - 8, behavior: "smooth" });
    shown = current;
  }
  $("#ready").textContent = !photo ? "Add a photo to begin."
    : !calibrated() ? "Set the scale: tap the reference points and fill in the real numbers."
    : picks.length < 2 ? "Tap at least two holds."
    : `${picks.length} points ready.`;

  const list = $("#picked");
  list.replaceChildren();
  picks.forEach((p, i) => {
    const li = document.createElement("li");
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = i + 1;
    const label = document.createElement("span");
    label.className = "what";
    label.textContent = p.hold == null ? "By hand" : "Detected";
    li.append(badge, label);
    for (const [text, act] of [
      ["Move", () => { moving = i; mode = "move"; say(`Tap where point ${i + 1} belongs.`); update(); }],
      ["Remove", () => { picks.splice(i, 1); reset(); changed(); }],
    ]) {
      const b = document.createElement("button");
      b.textContent = text;
      b.setAttribute("aria-label", `${text} point ${i + 1}`);
      b.disabled = busy;
      b.onclick = act;
      li.append(b);
    }
    list.append(li);
  });
}

const say = (m) => { $("#status").textContent = m; };
const reset = () => { moving = null; mode = "select"; };

function changed() {
  revision++;
  gaps = [];
  last = null;
  tape = [];
  clearBeta();
  $("#out").hidden = true;
  $("#error").textContent = "";
  update();
  draw();
}

function shape(x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

function pill(text, x, y, k, fill, ink) {
  ctx.font = `800 ${12 * k}px -apple-system, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 14 * k, h = 20 * k;
  shape(x - w / 2, y - h / 2, w, h, h / 2);
  ctx.fillStyle = COLOUR.shadow;
  ctx.lineWidth = 3 * k;
  ctx.strokeStyle = COLOUR.shadow;
  ctx.stroke();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.fillStyle = ink;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y + 0.5 * k);
}

// A dark line under a coloured one keeps it readable over any wall.
function stroke(points, colour, width, k) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const [style, w] of [[COLOUR.shadow, width + 3], [colour, width]]) {
    ctx.strokeStyle = style;
    ctx.lineWidth = w * k;
    ctx.beginPath();
    points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  }
}

function draw() {
  if (!photo) return;
  ctx.drawImage(photo, 0, 0);
  const k = pen ?? 1 / view;
  const chosen = new Set(picks.map((p) => p.hold));

  for (const h of holds) {
    const [x1, y1, x2, y2] = h.box;
    shape(x1, y1, x2 - x1, y2 - y1, 5 * k);
    if (chosen.has(h.id)) {
      ctx.fillStyle = "rgba(61,220,151,.22)";
      ctx.fill();
      ctx.strokeStyle = COLOUR.route;
      ctx.lineWidth = 2 * k;
    } else {
      ctx.strokeStyle = "rgba(255,255,255,.32)";
      ctx.lineWidth = 1.2 * k;
    }
    ctx.stroke();
  }

  // The distances give way to the figure while it is showing.
  if (!climb) for (const g of gaps) stroke([picks[g.from].xy, picks[g.to].xy], COLOUR.gap, 2.5, k);

  // Reference points as crosshairs: they are aimed at a bolt hole, not a hold.
  const names = refsNeeded() === 4 ? ["TL", "TR", "BL", "BR"] : ["A", "B"];
  refs.forEach(([x, y], i) => {
    for (const [style, w] of [[COLOUR.shadow, 4.5], [COLOUR.ref, 2]]) {
      ctx.strokeStyle = style;
      ctx.lineWidth = w * k;
      ctx.beginPath();
      ctx.arc(x, y, 8 * k, 0, Math.PI * 2);
      for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
        ctx.moveTo(x + dx * 11 * k, y + dy * 11 * k);
        ctx.lineTo(x + dx * 16 * k, y + dy * 16 * k);
      }
      ctx.stroke();
    }
    pill(names[i], x, y - 28 * k, k, COLOUR.ref, "#1c0f33");
  });

  for (const g of climb ? [] : gaps) {
    const a = picks[g.from].xy, b = picks[g.to].xy;
    pill(`${g.metres.toFixed(2)} m`, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, k, COLOUR.gap, COLOUR.gapInk);
  }

  picks.forEach(({ xy: [x, y] }, i) => {
    ctx.beginPath();
    ctx.arc(x, y, 11 * k, 0, Math.PI * 2);
    ctx.fillStyle = COLOUR.route;
    ctx.fill();
    ctx.lineWidth = 2.5 * k;
    ctx.strokeStyle = COLOUR.shadow;
    ctx.stroke();
    ctx.fillStyle = COLOUR.routeInk;
    ctx.font = `800 ${12 * k}px -apple-system, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), x, y + 0.5 * k);
  });
  if (climb) drawFigure(climb.frames[frame], k);
}

function drawFigure(f, k) {
  const J = f.joints;
  const tone = (limb) => (f.beyond.includes(limb) ? "#ff7a6b" : f.moving === limb ? COLOUR.gap : "#f3f6fa");
  const torso = [J.shoulder_l, J.shoulder_r, J.hip_r, J.hip_l];
  ctx.beginPath();
  torso.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = "rgba(183,148,255,.28)";
  ctx.fill();
  stroke([...torso, torso[0]], "#f3f6fa", 3, k);
  const chest = [(J.shoulder_l[0] + J.shoulder_r[0]) / 2, (J.shoulder_l[1] + J.shoulder_r[1]) / 2];
  stroke([chest, J.neck], "#f3f6fa", 4, k);
  for (const [limb, chain] of Object.entries(CHAINS)) stroke(chain.map((n) => J[n]), tone(limb), 5, k);
  ctx.beginPath();
  ctx.arc(J.head[0], J.head[1], f.head_px, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(183,148,255,.35)";
  ctx.fill();
  ctx.lineWidth = 3 * k;
  ctx.strokeStyle = COLOUR.shadow;
  ctx.stroke();
  ctx.lineWidth = 2 * k;
  ctx.strokeStyle = "#f3f6fa";
  ctx.stroke();
  for (const limb of Object.keys(CHAINS)) {
    ctx.beginPath();
    ctx.arc(J[limb][0], J[limb][1], 6 * k, 0, Math.PI * 2);
    ctx.fillStyle = tone(limb);
    ctx.fill();
    ctx.lineWidth = 2 * k;
    ctx.strokeStyle = COLOUR.shadow;
    ctx.stroke();
  }
}

function fit() {
  if (!photo) return;
  const stage = $("#stage");
  const pad = getComputedStyle(stage);
  const w = stage.clientWidth - parseFloat(pad.paddingLeft) - parseFloat(pad.paddingRight);
  const h = stage.clientHeight - parseFloat(pad.paddingTop) - parseFloat(pad.paddingBottom);
  // A hidden tab or a layout that has not settled reports zero here. Scaling to
  // it would style the canvas to 0px and leave it invisible and untappable even
  // after the space arrives, so wait for a size instead.
  if (w < 1 || h < 1) return;
  view = Math.min(w / photo.width, h / photo.height);
  canvas.style.width = `${photo.width * view}px`;
  canvas.style.height = `${photo.height * view}px`;
  draw();
}
new ResizeObserver(fit).observe($("#stage"));

// A scroll or pinch must not drop a point where it happens to end.
let down = null;
canvas.onpointerdown = (e) => { down = [e.clientX, e.clientY]; };
canvas.onpointercancel = () => { down = null; };
canvas.onclick = (e) => {
  if (!photo || busy || recording || !down) return;
  if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 8) { down = null; return; }
  const r = canvas.getBoundingClientRect();
  // Without a laid-out canvas the arithmetic below is NaN, and NaN passes every
  // comparison that would otherwise reject it, including the hold-distance test.
  if (r.width < 1 || r.height < 1) { down = null; return; }
  down = null;
  const xy = [
    Math.min(photo.width, Math.max(0, (e.clientX - r.left) * canvas.width / r.width)),
    Math.min(photo.height, Math.max(0, (e.clientY - r.top) * canvas.height / r.height)),
  ];

  if (mode === "calibrate") {
    refs.push(xy);
    if (refs.length === refsNeeded()) {
      mode = "select";
      say(calibrated() ? "Scale set. Now tap holds in the order you climb them."
                       : "Points set. Fill in the real numbers beside them.");
    }
    else say(`Reference point ${refs.length + 1} of ${refsNeeded()}.`);
  } else if (mode === "move" && moving !== null) {
    picks[moving] = { xy, hold: null };
    reset();
    say("Point moved.");
  } else if (mode === "manual") {
    picks.push({ xy, hold: null });
  } else {
    const near = holds
      .map((h) => ({ h, d: Math.hypot(h.centre[0] - xy[0], h.centre[1] - xy[1]) * view }))
      .sort((a, b) => a.d - b.d)[0];
    if (!near || near.d > 28) {
      say(holds.length ? "No hold there. Use Add point to place one by hand."
                       : "No detections. Use Add point to place points by hand.");
      return;
    }
    const at = picks.findIndex((p) => p.hold === near.h.id);
    if (at >= 0) picks.splice(at, 1);
    else picks.push({ xy: [...near.h.centre], hold: near.h.id });
  }
  changed();
};

$("#select").onclick = () => { reset(); say("Tap holds in the order you climb them."); update(); };
$("#manual").onclick = () => { moving = null; mode = "manual"; say("Tap anywhere to add a point."); update(); };
$("#undo").onclick = () => { picks.pop(); reset(); changed(); };
$("#clear").onclick = () => { picks = []; reset(); changed(); };
$("#calibrate").onclick = () => {
  refs = [];
  moving = null;
  mode = "calibrate";
  say(refsNeeded() === 4 ? "Tap four bolt holes: top-left, top-right, bottom-left, bottom-right."
                         : "Tap the two ends of your known distance.");
  changed();
};
$("#undo-ref").onclick = () => { refs.pop(); mode = "calibrate"; changed(); };
const methodChanged = () => {
  refs = [];
  reset();
  $("#grid-fields").hidden = refsNeeded() === 2;
  $("#reference-fields").hidden = refsNeeded() === 4;
  $("#calibrate").textContent = refsNeeded() === 4 ? "Tap 4 bolt holes" : "Tap 2 points";
  changed();
};
for (const r of document.querySelectorAll('input[name="method"]')) r.onchange = methodChanged;
for (const id of ["metres", "cols", "rows", "spacing"]) $("#" + id).oninput = changed;

$("#file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  busy = true;
  photo = null; holds = []; picks = []; refs = []; gaps = [];
  reset();
  canvas.hidden = true;
  $("#empty").hidden = false;
  $("#found").textContent = "";
  changed();
  say("Opening photo…");

  try {
    const bitmap = await createImageBitmap(file);           // honours EXIF orientation
    const k = Math.min(1, WORK / Math.max(bitmap.width, bitmap.height));
    // Even sides, which the H.264 encoder behind a saved video needs.
    const even = (v) => Math.max(2, Math.round(v * k / 2) * 2);
    const work = new OffscreenCanvas(even(bitmap.width), even(bitmap.height));
    work.getContext("2d").drawImage(bitmap, 0, 0, work.width, work.height);
    bitmap.close();
    photo = work;
    canvas.width = work.width;
    canvas.height = work.height;
    canvas.hidden = false;
    $("#empty").hidden = true;
    fit();
    say("Finding holds…");
    await detect();
  } catch (err) {
    $("#error").textContent = `Could not open that photo: ${err.message}`;
    say("Choose a different photo.");
  } finally {
    busy = false;
    $("#file").value = "";
    update();
    draw();
  }
};

async function detect() {
  const bar = $("#bar");
  bar.hidden = false;
  try {
    if (!detector) {
      bar.classList.remove("busy");
      detector = await loadDetector((seen, total) => {
        bar.firstElementChild.style.width = total ? `${(seen / total) * 100}%` : "";
        $("#found").textContent = total
          ? `Downloading the hold finder · ${(seen / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB · first time only`
          : "Downloading the hold finder · first time only";
      });
    }
    bar.classList.add("busy");
    $("#found").textContent = "Finding holds…";
    holds = await detector.detect(photo);
    $("#found").textContent = `${holds.length} holds found on your ${detector.device === "webgpu" ? "GPU" : "CPU"}.`;
    say(holds.length ? "Next, set the scale from the bolt holes." : "No holds found. Place points by hand.");
  } catch (err) {
    // Detection is a convenience. Measuring by hand is the part that must work.
    holds = [];
    $("#found").textContent = "Automatic hold finding is unavailable here. Add points by hand instead.";
    say("Set the scale, then add points by hand.");
    console.warn("detector unavailable:", err);
  } finally {
    bar.hidden = true;
    bar.classList.remove("busy");
  }
}

$("#measure").onclick = async () => {
  busy = true;
  $("#error").textContent = "";
  say("Measuring…");
  update();
  const version = revision;
  const calibration = currentCalibration();

  try {
    const response = await fetch("/api/measure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        size: [photo.width, photo.height],
        calibration,
        points: picks.map((p) => p.xy),
      }),
    });
    let result;
    try { result = await response.json(); }
    catch { throw new Error("The server sent something unreadable. Try again."); }
    if (!response.ok) {
      throw new Error(typeof result.detail === "string" ? result.detail
        : "Check the reference points and the numbers beside them.");
    }
    if (version !== revision) { say("Inputs changed. Measure again."); return; }

    gaps = result.gaps;
    draw();
    $("#out").hidden = false;
    $("#total-m").textContent = `${result.total_m.toFixed(2)} m`;
    $("#rise-m").textContent = `${result.rise_m.toFixed(2)} m`;
    $("#basis").textContent = result.basis;
    $("#note").textContent = result.note;
    $("#warning").hidden = !result.warning;
    $("#warning").textContent = result.warning || "";
    last = result;
    tape = gaps.map(() => null);
    showGaps();
    update();
    $("#out").scrollIntoView({ behavior: "smooth", block: "start" });
    say("Done. Check a gap with a tape, or move a point to measure again.");
  } catch (err) {
    if (version === revision) {
      $("#error").textContent = err.message;
      say("Check the inputs and try again.");
    }
  } finally {
    busy = false;
    update();
  }
};

/* Ground truth, collected where it is measured. Written down by hand, a tape
   reading has to be matched to the right pair of taps afterwards from a
   description like "blue jug to pink crimp", which is where it goes wrong. */
function showGaps() {
  const list = $("#gaps");
  list.replaceChildren();
  gaps.forEach((g, i) => {
    const li = document.createElement("li");
    const pair = document.createElement("span");
    pair.className = "pair";
    pair.textContent = `${g.from + 1} → ${g.to + 1}`;
    const metres = document.createElement("span");
    metres.className = "metres";
    metres.textContent = `${g.metres.toFixed(2)} m`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "any";
    input.inputMode = "decimal";
    input.placeholder = "tape";
    input.setAttribute("aria-label", `Tape measure from point ${g.from + 1} to ${g.to + 1}`);
    const err = document.createElement("span");
    err.className = "err";
    input.oninput = () => {
      const v = +input.value;
      tape[i] = input.value !== "" && Number.isFinite(v) && v > 0 ? v * PER_UNIT[unit()] : null;
      scoreGaps();
    };
    li.append(pair, metres, input, err);
    list.append(li);
  });
  scoreGaps();
}

function scoreGaps() {
  const rows = [...$("#gaps").querySelectorAll(".err")];
  const errors = [];
  gaps.forEach((g, i) => {
    if (tape[i] == null) { rows[i].textContent = ""; rows[i].className = "err"; return; }
    const off = g.metres - tape[i];
    const pct = (off / tape[i]) * 100;
    errors.push(Math.abs(pct));
    const sign = (v) => (v >= 0 ? "+" : "−");
    rows[i].textContent = `${sign(off)}${Math.abs(off * 100).toFixed(1)} cm · ${sign(pct)}${Math.abs(pct).toFixed(1)}%`;
    rows[i].className = `err ${Math.abs(pct) < 5 ? "good" : Math.abs(pct) < 10 ? "warn" : "bad"}`;
  });
  $("#tape-summary").textContent = errors.length
    ? `${errors.length} checked · mean error ${(errors.reduce((a, b) => a + b, 0) / errors.length).toFixed(1)}% · worst ${Math.max(...errors).toFixed(1)}%`
    : "";
}

// Re-read every box in the new unit rather than converting what was typed.
for (const r of document.querySelectorAll('input[name="unit"]')) {
  r.onchange = () => [...$("#gaps").querySelectorAll("input")].forEach((input) => input.oninput());
}

function currentCalibration() {
  return method() === "grid"
    ? { method: "grid", corners: refs, cols: number("#cols"), rows: number("#rows"), spacing_m: number("#spacing") }
    : { method: "reference", points: refs, metres: number("#metres") };
}

$("#save").onclick = async () => {
  if (!last || !photo) return;
  const jpeg = await photo.convertToBlob({ type: "image/jpeg", quality: 0.9 });
  const photoData = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(jpeg);
  });
  const units = unit();
  const round = (v) => Math.round(v * 10) / 10;
  const record = {
    app: "hold-distance",
    version: 1,
    saved: new Date().toISOString(),
    // Every coordinate below is in this copy of the photo, not the original.
    photo: { width: photo.width, height: photo.height, jpeg: photoData },
    detector: detector ? detector.device : null,
    holds: holds.map((h) => h.box.map(round)),
    calibration: currentCalibration(),
    // Exactly what the API was sent, so the measurement can be rerun to the digit.
    points: picks.map((p) => ({ xy: p.xy, hold: p.hold })),
    result: last,
    tape: gaps.map((g, i) => ({ from: g.from, to: g.to, measured_m: g.metres,
                                 tape_m: tape[i], unit: units })),
  };
  deliver([new File([JSON.stringify(record)], `hold-distance-${stamp(record.saved)}.json`,
                    { type: "application/json" })], "Hold distance check");
};

const stamp = (iso = new Date().toISOString()) => iso.slice(0, 16).replace(/[-:T]/g, "");

// On a phone the share sheet is the natural way to send files on; elsewhere,
// a plain download of each.
async function deliver(files, title) {
  if (navigator.canShare && navigator.canShare({ files })) {
    try { await navigator.share({ files, title }); return; }
    catch (err) { if (err.name === "AbortError") return; }
  }
  for (const file of files) {
    const url = URL.createObjectURL(file);
    const a = Object.assign(document.createElement("a"), { href: url, download: file.name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

/* The beta. The API runs the search and works out every frame; this only
   plays them, so what is drawn is exactly what was computed. */
function clearBeta() {
  pause();
  climb = null;
  frame = 0;
  $("#player").hidden = true;
  $("#beta-error").textContent = "";
}

function betaRequest() {
  const W = photo.width, H = photo.height;
  const clip = (x, y) => [Math.min(W, Math.max(0, x)), Math.min(H, Math.max(0, y))];
  const square = ([x, y], r) => [clip(x - r, y - r), clip(x + r, y - r), clip(x + r, y + r), clip(x - r, y + r)];
  const polys = holds.map(({ box: [x1, y1, x2, y2] }) => [clip(x1, y1), clip(x2, y1), clip(x2, y2), clip(x1, y2)]);
  // A point placed by hand has no outline, so give it a typical hold's.
  const spans = holds.map((h) => h.span).sort((a, b) => a - b);
  const half = (spans.length ? spans[spans.length >> 1] : 40) / 2;
  const ids = picks.map((p) => (p.hold != null ? p.hold : polys.push(square(p.xy, half)) - 1));
  return { ids, body: { size: [W, H], calibration: currentCalibration(), holds: polys, route: ids,
                        height_m: number("#height") || 1.68 } };
}

const where = (id) => {
  const at = route.indexOf(id);
  return at >= 0 ? `hold ${at + 1}` : "a foothold";
};

function caption() {
  const f = climb.frames[frame];
  let text = f.move === 0 ? "Start: both hands and both feet on their first holds." : (() => {
    const m = climb.moves[f.move - 1];
    return `Move ${f.move} of ${climb.moves.length} · ${LIMB_NAME[m.limb]} to ${where(m.to)} · ${m.metres.toFixed(2)} m`;
  })();
  if (f.beyond.length) text += ` · ${f.beyond.map((l) => LIMB_NAME[l].toLowerCase()).join(" and ")} out of reach`;
  $("#caption").textContent = text;
  for (const b of $("#moves").querySelectorAll("button")) b.setAttribute("aria-current", +b.dataset.move === f.move);
}

function show(i) {
  frame = Math.max(0, Math.min(climb.frames.length - 1, i));
  $("#scrub").value = frame;
  caption();
  draw();
  if (recording && !recording.stopping) recording.shown.push([performance.now() - recording.t0, frame]);
}

function pause() {
  playing = false;
  clearTimeout(timer);
  $("#play").textContent = "Play";
  if (recording && !recording.stopping) {
    const r = recording, { recorder, keep } = r;
    r.stopping = true;
    // Rest on the last frame for a second. It has to be redrawn throughout:
    // with nothing new drawn the recorder ends the video on the last change.
    let rest = keep ? 10 : 0;
    const hold = () => {
      if (rest-- > 0) return draw(), setTimeout(hold, 100);
      r.duration = (performance.now() - r.t0) / 1000;
      recorder.stop();
    };
    hold();
  }
}

function play() {
  if (frame >= climb.frames.length - 1) show(0);
  playing = true;
  $("#play").textContent = "Pause";
  const tick = () => {
    if (!playing) return;
    if (frame >= climb.frames.length - 1) {
      if (recording) recording.keep = true;
      return pause();
    }
    show(frame + 1);
    // Hold each stance long enough to read before the next limb moves.
    timer = setTimeout(tick, climb.frames[frame].key ? 550 : 45);
  };
  timer = setTimeout(tick, 300);
}

const keyframes = () => climb.frames.flatMap((f, i) => (f.key ? [i] : []));
$("#play").onclick = () => (playing ? pause() : play());
$("#prev").onclick = () => { pause(); show(keyframes().filter((i) => i < frame).pop() ?? 0); };
$("#next").onclick = () => { pause(); show(keyframes().find((i) => i > frame) ?? climb.frames.length - 1); };
$("#scrub").oninput = () => { pause(); show(+$("#scrub").value); };
// Which way a limb goes on the photo. The climber faces the wall, so their
// left is the photo's left.
function heading(dx, up) {
  const across = Math.abs(up) > 2 * Math.abs(dx) ? "" : dx < 0 ? "left" : "right";
  const along = Math.abs(dx) > 2 * Math.abs(up) ? "" : up > 0 ? "up" : "down";
  return [along, across].filter(Boolean).join(" and ");
}

/* The beta written out and timed against the video saved with it. The prompt
   is for a video model to read; the frames are what video_check.py holds that
   model's output to. */
function transcript(r, video) {
  const at = new Map();
  for (const [t, i] of r.shown) if (!at.has(i)) at.set(i, +(t / 1000).toFixed(3));
  const keys = keyframes();
  const finish = route.at(-1);
  const moves = climb.moves.map((m, k) => {
    const a = climb.frames[keys[k]].joints[m.limb], b = climb.frames[keys[k + 1]].joints[m.limb];
    const way = heading(b[0] - a[0], a[1] - b[1]);
    const verb = m.limb.endsWith("H") ? "reaches" : "steps";
    const first = climb.frames.findIndex((f) => f.move === k + 1);
    return { n: k + 1, limb: m.limb, from: m.from, to: m.to, metres: m.metres,
             start_s: at.get(first) ?? null, end_s: at.get(keys[k + 1]) ?? null,
             text: `${LIMB_NAME[m.limb]} ${verb} ${way ? way + " " : ""}about ${Math.round(m.metres * 100)} cm` +
                   (m.to === finish ? " to the finish hold" : "") };
  });
  const prompt = [
    "One climber, seen from behind, on an indoor bouldering wall. Fixed camera, no cuts.",
    "They start with both hands and both feet on holds, then move one hand or foot at a time:",
    ...moves.map((m) => `${m.n}. ${m.text}.`),
    "They finish with both hands together on the finish hold.",
  ].join(" ");
  return {
    app: "climbing-beta", kind: "beta-transcript", version: 1, saved: new Date().toISOString(),
    video: { file: video.name, type: video.type, width: canvas.width, height: canvas.height,
             duration_s: +r.duration.toFixed(3) },
    // Pixels below are in the video, which is this copy of the photo.
    size: [photo.width, photo.height], calibration: currentCalibration(), basis: climb.basis,
    height_m: number("#height") || 1.68,
    route: picks.map((p, i) => ({ n: i + 1, hold: route[i], xy: p.xy })),
    start: climb.start, prompt, moves,
    frames: climb.frames.map((f, i) => ({ t: at.get(i) ?? null, move: f.move, key: f.key, moving: f.moving,
                                          beyond: f.beyond, joints: f.joints })),
  };
}

/* The video is the canvas itself, recorded while the beta plays through once,
   so it shows exactly the frames the player does at the same pace. */
$("#video").onclick = () => {
  const type = videoType();
  if (!climb || !type || recording) return;
  const recorder = new MediaRecorder(canvas.captureStream(30), { mimeType: type, videoBitsPerSecond: 8e6 });
  const chunks = [];
  const r = { recorder, keep: false, stopping: false, shown: [], t0: 0, duration: 0 };
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    for (const t of recorder.stream.getTracks()) t.stop();
    recording = pen = null;
    update();
    draw();
    if (!r.keep) return say("Recording stopped.");
    const name = `climbing-beta-${stamp()}`;
    const video = new File(chunks, `${name}.${type.startsWith("video/mp4") ? "mp4" : "webm"}`,
                           { type: type.split(";")[0] });
    const text = new File([JSON.stringify(transcript(r, video), null, 1)], `${name}.json`,
                          { type: "application/json" });
    deliver([video, text], "Climbing beta");
    say("Video and transcript saved.");
  };
  pause();
  recording = r;
  pen = Math.max(canvas.width, canvas.height) / 800;
  update();
  recorder.start();
  r.t0 = performance.now();
  say("Recording the beta…");
  show(0);
  play();
};

$("#height").oninput = () => { if (climb) clearBeta(); update(); draw(); };

$("#beta").onclick = async () => {
  clearBeta();
  busy = true;
  update();
  say("Finding the beta…");
  const version = revision;
  const { ids, body } = betaRequest();
  try {
    const response = await fetch("/api/beta", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    let result;
    try { result = await response.json(); }
    catch { throw new Error("The server sent something unreadable. Try again."); }
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Check the scale and the route, then try again.");
    if (version !== revision) return;
    if (!result.ok) {
      $("#beta-error").textContent = `No beta: ${result.reason}`;
      say("No beta for this route. The reason is under Beta.");
      return;
    }
    climb = result;
    route = ids;
    const list = $("#moves");
    list.replaceChildren();
    climb.moves.forEach((m, i) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.dataset.move = i + 1;
      b.textContent = `${i + 1}. ${LIMB_NAME[m.limb]} to ${where(m.to)} · ${m.metres.toFixed(2)} m`;
      b.onclick = () => { pause(); show(climb.frames.findIndex((f) => f.move === i + 1 && f.key)); };
      li.append(b);
      list.append(li);
    });
    const skipped = climb.skipped.map((id) => route.indexOf(id) + 1).sort((a, b) => a - b);
    const listed = skipped.length > 1 ? `holds ${skipped.slice(0, -1).join(", ")} and ${skipped.at(-1)}` : `hold ${skipped[0]}`;
    const beyond = climb.frames.filter((f) => f.key && f.beyond.length).length;
    $("#beta-note").textContent = [
      skipped.length ? `The search did not need ${listed} for hands.` : "",
      beyond ? `${beyond} stance${beyond > 1 ? "s ask" : " asks"} more reach than a body of this height has, shown in red.` : "",
      "Feet can use any detected hold. This is the search's beta; it has not yet been checked against real climbs.",
    ].filter(Boolean).join(" ");
    $("#scrub").max = climb.frames.length - 1;
    $("#player").hidden = false;
    show(0);
    $("#beta-card").scrollIntoView({ behavior: "smooth", block: "start" });
    say(`Beta found: ${climb.moves.length} moves. Press Play.`);
  } catch (err) {
    if (version === revision) $("#beta-error").textContent = err.message;
  } finally {
    busy = false;
    update();
  }
};

update();
