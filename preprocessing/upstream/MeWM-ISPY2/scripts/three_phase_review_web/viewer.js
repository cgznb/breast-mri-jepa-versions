"use strict";

const manifest = window.REVIEW_MANIFEST;
const $ = id => document.getElementById(id);
const phaseNames = ["平扫", "首个增强相", "晚期增强相", "首个增强 − 平扫", "晚期增强 − 平扫"];
const phaseKeys = ["pre_aqc0", "first_post_aqc1", "metadata_late"];
const referenceNames = {target: "真实随访", source: "早期输入", target_vq: "随访 VQ 重建", source_vq: "输入 VQ 重建"};
const state = {index: 0, step: 50, phase: 1, sample: "sample1", reference: "target", neighborhood: "prediction", checkpoint: "best", anchor: 16, selected: 16, view: "slices", mask: true, level: 2, width: 5, token: 0};
const packs = new Map();
const pending = new Map();
const decoded = new Map();
let current = null;
let drawRequest = 0;
let filtered = [];
const format = value => value == null || !Number.isFinite(value) ? "—" : value.toFixed(3);
const caseData = () => manifest.cases[state.index];
const sampleName = () => state.sample === "mean" ? "4 次生成均值" : `独立样本 ${state.sample.slice(-1)}`;
const checkpointName = () => state.checkpoint === "best" ? "最优权重" : "最终权重";
const offsetLabel = z => z === state.anchor ? "同层" : `${z > state.anchor ? "+" : ""}${z - state.anchor}`;

window.registerReviewData = (key, payload) => {
  const request = pending.get(key);
  if (key.startsWith(caseData().id + "/")) packs.set(key, payload);
  if (request) request.resolve(payload);
};

function loadPack(kind) {
  const key = `${caseData().id}/${kind}`;
  if (packs.has(key)) return Promise.resolve(packs.get(key));
  if (pending.has(key)) return pending.get(key).promise;
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  const script = document.createElement("script");
  const timer = setTimeout(() => reject(new Error(`加载超时：${kind}`)), 60000);
  pending.set(key, {promise, resolve, reject});
  script.src = `data/${key}.js`;
  script.onerror = () => reject(new Error(`无法读取病例数据：${kind}`));
  document.head.append(script);
  promise.finally(() => { clearTimeout(timer); script.remove(); pending.delete(key); }).catch(() => {});
  return promise;
}

function readPNG(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = image.width; canvas.height = image.height;
        const context = canvas.getContext("2d", {willReadFrequently: true});
        context.drawImage(image, 0, 0);
        resolve(context.getImageData(0, 0, image.width, image.height).data);
      } catch (error) { reject(error); }
    };
    image.onerror = () => reject(new Error("图像数据解码失败"));
    image.src = url;
  });
}

async function decodeVolume(kind, name) {
  const key = `${caseData().id}/${kind}/${name}`;
  if (decoded.has(key)) return decoded.get(key);
  const promise = (async () => {
    const pack = await loadPack(kind);
    const item = pack.volumes[name];
    if (!item) throw new Error(`缺少图像：${name}`);
    const pixels = await readPNG(item.png);
    const array = new Float32Array(pixels.length / 4);
    for (let i = 0; i < array.length; i++) array[i] = (pixels[i * 4] * 256 + pixels[i * 4 + 1]) * item.scale + item.lower;
    array.zeroTolerance = item.scale / 2 + 0.00001;
    if (array.length !== 3 * 32 * 128 * 128) throw new Error("图像维度不匹配");
    return array;
  })();
  decoded.set(key, promise);
  return promise;
}

async function decodeMask() {
  const key = `${caseData().id}/mask`;
  if (decoded.has(key)) return decoded.get(key);
  const promise = (async () => {
    const pack = await loadPack("reference");
    const pixels = await readPNG(pack.mask);
    const mask = new Uint8Array(pixels.length / 4);
    for (let i = 0; i < mask.length; i++) mask[i] = pixels[4 * i] > 127 ? 1 : 0;
    return mask;
  })();
  decoded.set(key, promise);
  return promise;
}

function valueAt(volume, z, pixel) {
  const position = z * 16384 + pixel;
  if (state.phase < 3) return volume[state.phase * 524288 + position];
  return volume[(state.phase - 2) * 524288 + position] - volume[position];
}

function nonzeroReference(value) {
  const tolerance = (current.reference.zeroTolerance || 0.0001) * (state.phase < 3 ? 1 : 2);
  return Math.abs(value) > tolerance;
}

function paint(canvas, volume, z, mask = current?.mask) {
  const context = canvas.getContext("2d");
  if (!volume || z < 0 || z > 31) { context.clearRect(0, 0, 128, 128); return; }
  const image = context.createImageData(128, 128);
  const lower = state.level - state.width / 2;
  for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) {
    const original = (127 - y) * 128 + x;
    const index = (y * 128 + x) * 4;
    const gray = Math.round(Math.max(0, Math.min(1, (valueAt(volume, z, original) - lower) / state.width)) * 255);
    image.data[index] = image.data[index + 1] = image.data[index + 2] = gray;
    image.data[index + 3] = 255;
    const m = z * 16384 + original;
    if (state.mask && mask && mask[m] && (x === 0 || x === 127 || y === 0 || y === 127 || !mask[m - 1] || !mask[m + 1] || !mask[m - 128] || !mask[m + 128])) {
      image.data[index] = 51; image.data[index + 1] = 211; image.data[index + 2] = 163;
    }
  }
  context.putImageData(image, 0, 0);
}

function currentMetric() {
  if (state.phase > 2) return null;
  const method = state.sample === "mean" ? "mean" : "sample";
  const sample = state.sample === "mean" ? -1 : Number(state.sample.slice(-1)) - 1;
  return caseData().metrics.find(r => r.steps === state.step && r.phase === phaseKeys[state.phase] && r.method === method && r.sample === sample);
}

function updateMeasures() {
  const row = currentMetric();
  const diversity = caseData().diversity.find(r => r.steps === state.step && r.phase === phaseKeys[state.phase]);
  const fields = [["预测 / 随访 · T0 MAE", row?.t0_target_mae], ["输入 / 随访 · T0 MAE", row?.t0_real_change_mae], ["预测 / 输入 · T0 MAE", row?.t0_source_mae], ["4 次噪声间差异", diversity?.t0_pairwise_mae]];
  $("case-measures").replaceChildren(...fields.map(([name, value]) => {
    const group = document.createElement("div"), term = document.createElement("dt"), detail = document.createElement("dd");
    term.textContent = name; detail.textContent = format(value); group.append(term, detail); return group;
  }));
  let total = 0, count = 0;
  if (current) for (let pixel = 0; pixel < 16384; pixel++) {
    const target = valueAt(current.reference, state.anchor, pixel);
    if (nonzeroReference(target)) { total += Math.abs(valueAt(current.neighborhood, state.selected, pixel) - target); count++; }
  }
  $("selected-mae").textContent = count ? format(total / count) : "—";
}

function resetWindow() {
  const bounds = state.phase < 3 ? caseData().window : caseData().enhancementWindow;
  state.level = (bounds[0] + bounds[1]) / 2;
  state.width = bounds[1] - bounds[0];
  $("level").min = String(Math.min(-10, bounds[0] - state.width));
  $("level").max = String(Math.max(30, bounds[1] + state.width));
  $("width").max = String(Math.max(30, state.width * 3));
  syncWindow();
}

function syncWindow() {
  $("level").value = state.level; $("width").value = state.width;
  $("level-value").value = state.level.toFixed(2); $("width-value").value = state.width.toFixed(2);
  $("anchor-range").value = $("anchor-number").value = state.anchor;
}

function buildNeighbors() {
  const build = offsets => offsets.map(offset => {
    const z = state.anchor + offset;
    const button = document.createElement("button");
    button.className = "slice-tile"; button.dataset.z = z;
    button.title = `z ${z} / ${offset > 0 ? "+" : ""}${offset} 层`;
    button.setAttribute("aria-label", button.title);
    if (z >= 0 && z < 32) {
      const canvas = document.createElement("canvas"); canvas.width = canvas.height = 128;
      button.append(canvas);
      button.onclick = () => selectSlice(z);
      button.ondblclick = () => { selectSlice(z); openZoom(); };
    } else {
      button.disabled = true;
      const unavailable = document.createElement("div"); unavailable.className = "tile-unavailable"; unavailable.textContent = "超出裁剪范围"; button.append(unavailable);
    }
    const label = document.createElement("span"); label.className = "tile-label";
    const first = document.createElement("span"), second = document.createElement("span");
    first.textContent = `${offset > 0 ? "+" : ""}${offset}`; second.textContent = z >= 0 && z < 32 ? `z ${z}` : "—";
    label.append(first, second); button.append(label); return button;
  });
  $("before-grid").replaceChildren(...build(Array.from({length: 10}, (_, i) => i - 10)));
  $("after-grid").replaceChildren(...build(Array.from({length: 10}, (_, i) => i + 1)));
  const count = Math.min(10, state.anchor) + Math.min(10, 31 - state.anchor);
  $("neighbor-count").textContent = `${count} 个有效层 / 20 个位置 · 前后各 10 层`;
}

function draw() {
  if (!current) return;
  const z = state.selected;
  $("reference-title").textContent = referenceNames[state.reference];
  $("reference-z").textContent = `${state.reference.startsWith("source") ? caseData().earlier : caseData().later} · z ${state.anchor}`;
  $("prediction-title").textContent = state.neighborhood === "prediction" ? (z === state.anchor ? "同层预测" : "邻域预测") : referenceNames[state.neighborhood];
  $("prediction-z").textContent = `z ${z} · ${offsetLabel(z)}`;
  $("neighborhood-title").textContent = state.neighborhood === "prediction" ? "生成切片邻域" : `${referenceNames[state.neighborhood]}邻域`;
  paint($("reference-canvas"), current.reference, state.anchor);
  paint($("prediction-canvas"), current.neighborhood, z);
  document.querySelectorAll(".slice-tile").forEach(button => {
    const canvas = button.querySelector("canvas");
    if (canvas) paint(canvas, current.neighborhood, Number(button.dataset.z));
    button.classList.toggle("active", Number(button.dataset.z) === z);
  });
  updateMeasures();
  paint($("step-reference"), current.reference, state.anchor);
  $("step-reference-title").textContent = `${referenceNames[state.reference]} · z ${state.anchor}`;
  $("step-context").textContent = `${checkpointName()} · ${phaseNames[state.phase]} · ${sampleName()} · 预测 z ${z}（${offsetLabel(z)}）`;
  for (const [step, volume] of Object.entries(current.steps || {})) {
    paint($(`compare-${step}`), volume, z);
    let sum = 0, count = 0;
    for (let p = 0; p < 16384; p++) { const target = valueAt(current.reference, state.anchor, p); if (nonzeroReference(target)) { sum += Math.abs(valueAt(volume, z, p) - target); count++; } }
    $(`compare-value-${step}`).textContent = count ? format(sum / count) : "—";
  }
  $("footer-sample").textContent = `${checkpointName()} · Heun ${state.step} 步 / ${state.step * 2} NFE · ${sampleName()}`;
  if ($("zoom-dialog").open) drawZoom();
}

function scheduleDraw() { cancelAnimationFrame(drawRequest); drawRequest = requestAnimationFrame(draw); }

async function refresh() {
  const token = ++state.token;
  document.body.classList.add("loading"); $("status").textContent = "加载图像"; $("status").classList.add("busy"); $("error").hidden = true;
  try {
    const requests = [decodeVolume("reference", state.reference), decodeMask(), decodeVolume(`step${state.step}`, state.sample)];
    if (state.neighborhood !== "prediction") requests.push(decodeVolume("reference", state.neighborhood));
    const [reference, mask, prediction, realNeighborhood] = await Promise.all(requests);
    if (token !== state.token) return;
    current = {reference, mask, prediction, neighborhood: realNeighborhood || prediction, steps: {}};
    if (state.view === "steps") {
      const volumes = await Promise.all([10, 20, 30, 50].map(step => decodeVolume(`step${step}`, state.sample)));
      if (token !== state.token) return;
      current.steps = Object.fromEntries([10, 20, 30, 50].map((step, index) => [step, volumes[index]]));
    }
    draw(); $("status").textContent = "图像就绪";
  } catch (error) {
    if (token !== state.token) return;
    $("error").textContent = error.message; $("error").hidden = false; $("status").textContent = "加载失败";
  } finally {
    if (token === state.token) { document.body.classList.remove("loading"); $("status").classList.remove("busy"); }
  }
}

function renderList() {
  const query = $("search").value.trim().toLowerCase(), transition = $("transition-filter").value;
  filtered = manifest.cases.map((row, index) => ({row, index})).filter(({row}) => row.patient.toLowerCase().includes(query) && (transition === "all" || row.transition === transition));
  $("case-list").replaceChildren(...filtered.map(({row, index}) => {
    const button = document.createElement("button"); button.className = "case-row"; button.classList.toggle("active", index === state.index); button.dataset.index = index;
    const title = document.createElement("strong"), ordinal = document.createElement("span"); ordinal.className = "ordinal"; ordinal.textContent = String(index + 1).padStart(2, "0"); title.append(ordinal, document.createTextNode(row.patient));
    const details = document.createElement("span"); details.className = "row-details";
    const interval = document.createElement("span"), mark = document.createElement("span"); interval.textContent = row.transition; mark.textContent = row.clipped ? "T0 裁剪" : "验证病例"; details.append(interval, mark);
    button.append(title, details); button.onclick = () => chooseCase(index); return button;
  }));
  if (!filtered.length) { const empty = document.createElement("div"); empty.className = "empty-list"; empty.textContent = "无匹配病例"; $("case-list").append(empty); }
  $("case-position").textContent = `${state.index + 1} / ${manifest.cases.length}`;
  const position = filtered.findIndex(row => row.index === state.index);
  $("previous-case").disabled = position <= 0; $("next-case").disabled = position < 0 || position >= filtered.length - 1;
}

function chooseCase(index) {
  if (index < 0 || index >= manifest.cases.length) return;
  state.index = index; state.anchor = state.selected = caseData().anchor;
  packs.clear(); decoded.clear(); current = null;
  document.querySelectorAll("canvas").forEach(canvas => canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height));
  $("patient-name").textContent = caseData().patient;
  const registration = {rigid_fallback: "刚性配准", deformable: "形变配准", fixed_reference: "T0 参考"}[caseData().registration] || caseData().registration || "配准数据";
  const fields = [caseData().transition, caseData().interval_missing ? "间隔缺失" : `${caseData().days} 天`, registration, "最优 = 最终 · 第 6,300 次更新"];
  $("case-meta").replaceChildren(...fields.map(value => { const span = document.createElement("span"); span.textContent = value; return span; }));
  if (caseData().clipped) { const span = document.createElement("span"); span.className = "warning"; span.textContent = "T0 区域被裁剪"; $("case-meta").append(span); }
  resetWindow(); buildNeighbors(); renderList(); document.body.classList.remove("sidebar-open");
  const url = new URL(location.href); url.hash = `case=${index + 1}&checkpoint=${state.checkpoint}`; history.replaceState(null, "", url);
  refresh();
}

function selectSlice(z) { state.selected = Math.min(31, Math.max(0, z)); draw(); }
function moveCase(direction) { const position = filtered.findIndex(row => row.index === state.index); const next = filtered[position + direction]; if (next) chooseCase(next.index); }

function drawZoom() {
  $("zoom-title").textContent = `${caseData().patient} · ${checkpointName()} · ${phaseNames[state.phase]} · ${state.step} 步`;
  $("zoom-reference-title").textContent = `${referenceNames[state.reference]} · z ${state.anchor}`;
  $("zoom-prediction-title").textContent = `${state.neighborhood === "prediction" ? sampleName() : referenceNames[state.neighborhood]} · z ${state.selected}`;
  $("zoom-position").textContent = `z ${state.selected} · ${offsetLabel(state.selected)}`;
  paint($("zoom-reference"), current.reference, state.anchor); paint($("zoom-prediction"), current.neighborhood, state.selected);
}

function openZoom() { if (current) { $("zoom-dialog").showModal(); drawZoom(); } }

function changeView(view) {
  state.view = view;
  for (const name of ["slices", "steps", "metrics"]) $(`${name}-view`).hidden = name !== view;
  document.querySelectorAll("[data-view]").forEach(button => { const active = button.dataset.view === view; button.classList.toggle("active", active); button.setAttribute("aria-selected", String(active)); });
  refresh();
}

function init() {
  if (!manifest || manifest.cases.length !== 30 || !manifest.weightsIdentical) throw new Error("病例清单或权重核验信息无效");
  lucide.createIcons();
  for (const transition of [...new Set(manifest.cases.map(row => row.transition))].sort()) { const option = new Option(transition, transition); $("transition-filter").add(option); }
  $("search").oninput = $("transition-filter").onchange = renderList;
  $("previous-case").onclick = () => moveCase(-1); $("next-case").onclick = () => moveCase(1);
  $("menu").onclick = () => document.body.classList.toggle("sidebar-open");
  $("fullscreen").onclick = () => { if (document.fullscreenElement) document.exitFullscreen?.(); else document.documentElement.requestFullscreen?.().catch(() => {}); };
  $("checkpoint").onchange = event => { state.checkpoint = event.target.value; const url = new URL(location.href); url.hash = `case=${state.index + 1}&checkpoint=${state.checkpoint}`; history.replaceState(null, "", url); draw(); };
  document.querySelectorAll("[data-view]").forEach(button => button.onclick = () => changeView(button.dataset.view));
  document.querySelectorAll("[data-step]").forEach(button => button.onclick = () => {
    state.step = Number(button.dataset.step);
    document.querySelectorAll("[data-step]").forEach(other => { const active = Number(other.dataset.step) === state.step; other.classList.toggle("active", active); other.setAttribute("aria-pressed", String(active)); });
    refresh();
  });
  for (const name of ["sample", "reference", "neighborhood"]) $(name).onchange = event => { state[name] = event.target.value; refresh(); };
  $("phase").onchange = event => { state.phase = Number(event.target.value); resetWindow(); draw(); };
  $("mask-toggle").onchange = event => { state.mask = event.target.checked; draw(); };
  for (const id of ["anchor-range", "anchor-number"]) $(id).oninput = event => { const value = Number(event.target.value); if (!Number.isFinite(value)) return; state.anchor = state.selected = Math.max(0, Math.min(31, Math.round(value))); syncWindow(); buildNeighbors(); draw(); };
  for (const name of ["level", "width"]) $(name).oninput = event => { state[name] = Number(event.target.value); syncWindow(); scheduleDraw(); };
  $("reset-window").onclick = () => { state.anchor = state.selected = caseData().anchor; resetWindow(); buildNeighbors(); draw(); };
  $("previous-slice").onclick = $("zoom-previous").onclick = () => selectSlice(state.selected - 1);
  $("next-slice").onclick = $("zoom-next").onclick = () => selectSlice(state.selected + 1);
  $("center-slice").onclick = () => selectSlice(state.anchor);
  document.querySelectorAll("[data-enlarge]").forEach(button => button.onclick = openZoom);
  $("close-zoom").onclick = () => $("zoom-dialog").close();
  $("zoom-dialog").onclick = event => { if (event.target === $("zoom-dialog")) $("zoom-dialog").close(); };
  $("download-slice").onclick = () => {
    if (!current) return;
    const canvas = document.createElement("canvas"); canvas.width = 1060; canvas.height = 572;
    const context = canvas.getContext("2d"); context.fillStyle = "#f5f6f6"; context.fillRect(0, 0, canvas.width, canvas.height); context.fillStyle = "#24332e"; context.font = "16px system-ui";
    context.fillText(`${caseData().patient} · ${phaseNames[state.phase]} · ${checkpointName()} · ${state.step} steps · ${sampleName()}`, 16, 25);
    context.font = "13px system-ui"; context.fillText(`${referenceNames[state.reference]} z=${state.anchor}`, 16, 49); context.fillText(`${state.neighborhood === "prediction" ? "生成" : referenceNames[state.neighborhood]} z=${state.selected} / ${offsetLabel(state.selected)}`, 540, 49);
    context.drawImage($("reference-canvas"), 16, 59, 504, 504); context.drawImage($("prediction-canvas"), 540, 59, 504, 504);
    const link = document.createElement("a"); link.download = `${caseData().id}_${state.checkpoint}_${state.step}_${state.sample}_z${state.selected}.png`; link.href = canvas.toDataURL("image/png"); link.click();
  };
  document.addEventListener("keydown", event => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    if (event.key === "ArrowLeft") { event.preventDefault(); selectSlice(state.selected - 1); }
    if (event.key === "ArrowRight") { event.preventDefault(); selectSlice(state.selected + 1); }
  });
  for (const step of [10, 20, 30, 50]) {
    const figure = document.createElement("figure"), caption = document.createElement("figcaption"), title = document.createElement("span"), nfe = document.createElement("span"), canvas = document.createElement("canvas"), measure = document.createElement("div"), label = document.createElement("span"), value = document.createElement("strong");
    title.textContent = `${step} 步`; nfe.textContent = `${step * 2} NFE`; caption.append(title, nfe); canvas.width = canvas.height = 128; canvas.id = `compare-${step}`; measure.className = "step-value"; label.textContent = "所选层 MAE"; value.id = `compare-value-${step}`; value.textContent = "—"; measure.append(label, value); figure.append(caption, canvas, measure); $("step-grid").append(figure);
  }
  const table = $("summary-table").querySelector("tbody");
  for (const phase of phaseKeys) for (const step of [10, 20, 30, 50]) {
    const metric = method => manifest.summary.image_metrics.find(row => row.phase === phase && row.steps === (method === "source_copy" ? 0 : step) && row.method === method)?.t0_target_mae;
    const diversity = manifest.summary.diversity.find(row => row.phase === phase && row.steps === step).t0_pairwise_mae;
    const row = document.createElement("tr");
    for (const text of [phaseNames[phaseKeys.indexOf(phase)], step, format(metric("source_copy")), format(metric("sample")), format(metric("mean")), format(diversity)]) { const cell = document.createElement("td"); cell.textContent = text; row.append(cell); }
    table.append(row);
  }
  const hash = new URLSearchParams(location.hash.slice(1));
  if (hash.get("checkpoint") === "final") { state.checkpoint = "final"; $("checkpoint").value = "final"; }
  chooseCase(Math.max(0, Math.min(29, (Number(hash.get("case")) || 1) - 1)));
}

try { init(); } catch (error) { $("error").hidden = false; $("error").textContent = error.message; $("status").textContent = "初始化失败"; }
