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

const refsNeeded = () => ($("#method").value === "grid" ? 4 : 2);
const number = (id) => {
  const v = +$(id).value;
  return Number.isFinite(v) && v > 0 ? v : null;
};

function calibrated() {
  if (refs.length !== refsNeeded()) return false;
  if ($("#method").value === "reference") return number("#metres") !== null;
  return number("#spacing") !== null && Number.isInteger(number("#cols")) &&
         Number.isInteger(number("#rows"));
}

function update() {
  const live = !!photo && !busy;
  for (const id of ["calibrate", "select", "manual"]) $("#" + id).disabled = !live;
  $("#file").disabled = busy;
  $("#undo").disabled = $("#clear").disabled = !live || !picks.length;
  $("#undo-ref").disabled = !live || !refs.length;
  $("#measure").disabled = !live || picks.length < 2 || !calibrated();
  $("#select").setAttribute("aria-pressed", mode === "select");
  $("#manual").setAttribute("aria-pressed", mode === "manual");
  $("#cal-status").textContent = `${refs.length} of ${refsNeeded()} reference points`;
  $("#ready").textContent = !photo ? "Add a photo to begin."
    : !calibrated() ? "Set the scale: tap the reference points and fill in the real numbers."
    : picks.length < 2 ? "Tap at least two holds."
    : `${picks.length} points ready.`;

  const list = $("#picked");
  list.replaceChildren();
  picks.forEach((p, i) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `${i + 1}. ${p.hold == null ? "point" : "hold"}`;
    li.append(label);
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
  $("#out").hidden = true;
  $("#error").textContent = "";
  update();
  draw();
}

function plaque(text, x, y, k, colour) {
  ctx.font = `600 ${13 * k}px system-ui`;
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "#10141deb";
  ctx.fillRect(x - 4 * k, y - 15 * k, w + 8 * k, 21 * k);
  ctx.fillStyle = colour;
  ctx.fillText(text, x, y);
}

function draw() {
  if (!photo) return;
  ctx.drawImage(photo, 0, 0);
  const k = 1 / view;

  ctx.lineWidth = 1.5 * k;
  ctx.strokeStyle = "#ffffff55";
  for (const h of holds) {
    const [x1, y1, x2, y2] = h.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  for (const g of gaps) {
    const a = picks[g.from].xy, b = picks[g.to].xy;
    ctx.strokeStyle = "#ffd082";
    ctx.lineWidth = 2.5 * k;
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
    plaque(`${g.metres.toFixed(2)} m`, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, k, "#ffd082");
  }

  const names = refsNeeded() === 4 ? ["TL", "TR", "BL", "BR"] : ["A", "B"];
  for (const [points, colour, labels] of [
    [picks.map((p) => p.xy), "#69b2ff", null],
    [refs, "#ffd082", names],
  ]) {
    points.forEach(([x, y], i) => {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2.5 * k;
      ctx.beginPath();
      ctx.arc(x, y, 7 * k, 0, Math.PI * 2);
      ctx.stroke();
      plaque(labels ? labels[i] : String(i + 1), x + 10 * k, y - 10 * k, k, colour);
    });
  }
}

function fit() {
  if (!photo) return;
  const stage = $("#stage");
  const w = stage.clientWidth, h = stage.clientHeight;
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
  if (!photo || busy || !down) return;
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
    if (refs.length === refsNeeded()) { mode = "select"; say("Scale set. Now tap holds in order."); }
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
$("#method").onchange = () => {
  refs = [];
  reset();
  $("#grid-fields").hidden = refsNeeded() === 2;
  $("#reference-fields").hidden = refsNeeded() === 4;
  changed();
};
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
    const work = new OffscreenCanvas(Math.round(bitmap.width * k), Math.round(bitmap.height * k));
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
  try {
    if (!detector) {
      bar.hidden = false;
      detector = await loadDetector((seen, total) => {
        bar.firstElementChild.style.width = total ? `${(seen / total) * 100}%` : "";
      });
    }
    holds = await detector.detect(photo);
    $("#found").textContent = `${holds.length} holds found on your ${detector.device === "webgpu" ? "GPU" : "CPU"}.`;
    say(holds.length ? "Set the scale, then tap holds in order." : "No holds found. Place points by hand.");
  } catch (err) {
    // Detection is a convenience. Measuring by hand is the part that must work.
    holds = [];
    $("#found").textContent = "Automatic hold finding is unavailable here. Add points by hand instead.";
    say("Set the scale, then add points by hand.");
    console.warn("detector unavailable:", err);
  } finally {
    bar.hidden = true;
  }
}

$("#measure").onclick = async () => {
  busy = true;
  $("#error").textContent = "";
  say("Measuring…");
  update();
  const version = revision;
  const calibration = refsNeeded() === 2
    ? { method: "reference", points: refs, metres: number("#metres") }
    : { method: "grid", corners: refs, cols: number("#cols"), rows: number("#rows"),
        spacing_m: number("#spacing") };

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
    $("#total").textContent = `${result.total_m.toFixed(2)} m total · ${result.rise_m.toFixed(2)} m rise`;
    $("#basis").textContent = result.basis;
    $("#note").textContent = result.note;
    const list = $("#gaps");
    list.replaceChildren();
    for (const g of gaps) {
      const li = document.createElement("li");
      li.textContent = `${g.from + 1} → ${g.to + 1}: ${g.metres.toFixed(2)} m`;
      list.append(li);
    }
    if (result.warning) {
      const p = document.createElement("li");
      p.className = "warning";
      p.textContent = result.warning;
      list.append(p);
    }
    say("Done. Move a point or change the scale to measure again.");
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

update();
