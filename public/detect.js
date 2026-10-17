/* Hold detection in the browser, from an ONNX export of the trained weights.

   The detector cannot be deployed server-side: torch is 558 MB against a
   500 MB Python bundle, and there is no GPU. Running it here costs 0.3 points
   of recall (96.6% at 1280px against 96.9% for the .pt model at 1600px) and
   means the photo never leaves the phone.

   Ultralytics' own browser library panics on this model inside its WASM
   segmentation path, so this drives onnxruntime-web and does its own
   letterboxing and NMS. On a 24 MP wall photo it returns 150 holds where the
   Python pipeline returns 152.

   Boxes only. The 32 mask coefficients are in the output and ignored: a box
   centre stands in for the mask centroid, which differs by a centimetre or two
   on an asymmetric hold. Worth revisiting once measurement has a real error
   figure to compare against. */

const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
const EDGE = 1280;
const CONF = 0.15;   // tuned for recall: a missed hold cannot be measured
const IOU = 0.7;     // Ultralytics' own default
const MAX = 300;

export async function load(onProgress) {
  const ort = await import(ORT + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ORT;

  const weights = await download("models/holds_1280.onnx", onProgress);
  let session, device;
  for (const ep of ["webgpu", "wasm"]) {
    try {
      session = await ort.InferenceSession.create(weights, { executionProviders: [ep] });
      device = ep;
      break;
    } catch (err) {
      if (ep === "wasm") throw err;
    }
  }
  return { device, detect: (image) => detect(ort, session, image) };
}

async function download(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`model download failed (${response.status})`);
  const total = +response.headers.get("content-length") || 0;
  const chunks = [];
  let seen = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    seen += value.length;
    if (onProgress) onProgress(seen, total);
  }
  const bytes = new Uint8Array(seen);
  let at = 0;
  for (const c of chunks) { bytes.set(c, at); at += c.length; }
  return bytes;
}

async function detect(ort, session, image) {
  // Full-resolution phone photos overflow the engine, and the model only ever
  // sees EDGE pixels, so letterbox first: scale to fit, centre-pad with grey,
  // exactly as Ultralytics does.
  const scale = Math.min(EDGE / image.width, EDGE / image.height);
  const w = Math.round(image.width * scale), h = Math.round(image.height * scale);
  const padx = Math.floor((EDGE - w) / 2), pady = Math.floor((EDGE - h) / 2);

  const canvas = new OffscreenCanvas(EDGE, EDGE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, EDGE, EDGE);
  ctx.drawImage(image, padx, pady, w, h);

  const px = ctx.getImageData(0, 0, EDGE, EDGE).data;
  const area = EDGE * EDGE;
  const chw = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    chw[i] = px[i * 4] / 255;
    chw[area + i] = px[i * 4 + 1] / 255;
    chw[2 * area + i] = px[i * 4 + 2] / 255;
  }

  const out = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", chw, [1, 3, EDGE, EDGE]),
  });
  const tensor = out[session.outputNames[0]];
  const n = tensor.dims[2], d = tensor.data;

  const found = [];
  for (let i = 0; i < n; i++) {
    const score = d[4 * n + i];
    if (score < CONF) continue;
    const bw = d[2 * n + i], bh = d[3 * n + i];
    found.push({
      score,
      x1: (d[i] - bw / 2 - padx) / scale, y1: (d[n + i] - bh / 2 - pady) / scale,
      x2: (d[i] + bw / 2 - padx) / scale, y2: (d[n + i] + bh / 2 - pady) / scale,
    });
  }
  found.sort((a, b) => b.score - a.score);

  const kept = [];
  for (const box of found) {
    if (kept.length >= MAX) break;
    if (!kept.some((k) => overlap(k, box) > IOU)) kept.push(box);
  }
  return kept.map((b, id) => ({
    id,
    centre: [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2],
    box: [b.x1, b.y1, b.x2, b.y2],
    span: Math.sqrt((b.x2 - b.x1) * (b.y2 - b.y1)),
  }));
}

function overlap(a, b) {
  const w = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const h = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const i = w * h;
  return i / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i);
}
