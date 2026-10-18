"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {chromium} = require("/root/autodl-tmp/.codex/euler20_review_browser/node_modules/playwright");

const root = path.resolve(__dirname, "../runs/registered_three_phase_roi32_v1/fm_review_30cases_20260917");
const web = path.join(root, "web");
const output = path.join(web, "checks");
fs.mkdirSync(output, {recursive: true});

async function ready(page) {
  await page.waitForFunction(() => document.getElementById("status").textContent === "图像就绪" && !document.body.classList.contains("loading"), null, {timeout: 60000});
  assert.equal(await page.locator("#error").isVisible(), false);
}

async function pixelData(page, id = "prediction-canvas") {
  return page.evaluate(id => {
    const context = document.getElementById(id).getContext("2d");
    const values = context.getImageData(0, 0, 128, 128).data;
    let minimum = 255, maximum = 0, total = 0;
    const seen = new Set();
    for (let i = 0; i < values.length; i += 4) { minimum = Math.min(minimum, values[i]); maximum = Math.max(maximum, values[i]); total += values[i]; seen.add(values[i]); }
    return {minimum, maximum, mean: total / 16384, colors: seen.size, pixels: Array.from(values.filter((_, index) => index % 116 === 0))};
  }, id);
}

async function main() {
  const browser = await chromium.launch({executablePath: "/root/autodl-tmp/.codex/euler20_review_browser/browsers/chromium-1243/chrome-linux64/chrome", headless: true, args: ["--no-sandbox"]});
  const context = await browser.newContext({viewport: {width: 1440, height: 1100}, acceptDownloads: true});
  const page = await context.newPage();
  const errors = [], externalRequests = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => { if (/^https?:/.test(request.url())) externalRequests.push(request.url()); });
  await page.goto(pathToFileURL(path.join(web, "index.html")).href);
  await ready(page);
  assert.equal(await page.locator(".case-row").count(), 30);
  assert.equal(await page.locator(".slice-tile").count(), 20);
  assert.equal(await page.locator(".slice-tile:disabled").count(), 0);
  const neighbors = await page.locator(".slice-tile").evaluateAll(nodes => nodes.map(node => Number(node.dataset.z)));
  assert.equal(new Set(neighbors).size, 20);
  let first = await pixelData(page);
  assert(first.colors > 80 && first.maximum - first.minimum > 100);
  await page.screenshot({path: path.join(output, "desktop.png"), fullPage: true});

  await page.selectOption("#checkpoint", "final");
  assert.deepEqual((await pixelData(page)).pixels, first.pixels);
  assert.match(await page.locator("footer").innerText(), /最终权重/);
  await page.selectOption("#checkpoint", "best");
  await page.selectOption("#sample", "sample2"); await ready(page);
  assert.notDeepEqual((await pixelData(page)).pixels, first.pixels);
  await page.selectOption("#sample", "mean"); await ready(page);
  const meanPixels = await pixelData(page);
  assert.notDeepEqual(meanPixels.pixels, first.pixels);
  await page.selectOption("#sample", "sample1"); await ready(page);
  for (const step of [10, 20, 30, 50]) { await page.click(`[data-step="${step}"]`); await ready(page); assert((await pixelData(page)).colors > 80); }
  await page.click('[data-view="steps"]'); await ready(page);
  for (const step of [10, 20, 30, 50]) assert((await pixelData(page, `compare-${step}`)).colors > 80);
  await page.screenshot({path: path.join(output, "step_comparison.png"), fullPage: true});
  await page.click('[data-view="slices"]'); await ready(page);
  for (const phase of ["0", "1", "2", "3", "4"]) { await page.selectOption("#phase", phase); assert((await pixelData(page)).colors > 30); }
  await page.selectOption("#phase", "1");
  const real = await pixelData(page, "reference-canvas");
  await page.selectOption("#reference", "source"); await ready(page);
  assert.notDeepEqual((await pixelData(page, "reference-canvas")).pixels, real.pixels);
  await page.selectOption("#reference", "target"); await ready(page);
  await page.selectOption("#neighborhood", "target"); await ready(page);
  assert.deepEqual((await pixelData(page)).pixels, (await pixelData(page, "reference-canvas")).pixels);
  await page.selectOption("#neighborhood", "prediction"); await ready(page);
  const anchor = Number(await page.inputValue("#anchor-number"));
  await page.locator(".slice-tile").first().click();
  assert.match(await page.locator("#prediction-z").innerText(), new RegExp(`z ${anchor - 10}`));
  assert.match(await page.locator("#reference-z").innerText(), new RegExp(`z ${anchor}`));
  await page.click('[data-enlarge="prediction"]');
  assert(await page.locator("#zoom-dialog").isVisible());
  await page.click("#zoom-next");
  assert.match(await page.locator("#zoom-position").innerText(), new RegExp(`z ${anchor - 9}`));
  await page.click("#close-zoom");
  await page.fill("#anchor-number", "0");
  assert.equal(await page.locator(".slice-tile:disabled").count(), 10);
  await page.fill("#anchor-number", "31");
  assert.equal(await page.locator(".slice-tile:disabled").count(), 10);
  await page.click("#reset-window");
  assert.equal(await page.locator(".slice-tile:disabled").count(), 0);
  const masked = await pixelData(page);
  await page.uncheck("#mask-toggle");
  assert.notDeepEqual((await pixelData(page)).pixels, masked.pixels);
  await page.check("#mask-toggle");
  await page.evaluate(() => { const range = document.getElementById("width"); range.value = "1"; range.dispatchEvent(new Event("input", {bubbles: true})); });
  await page.waitForTimeout(100);
  assert.notDeepEqual((await pixelData(page)).pixels, masked.pixels);
  await page.click("#reset-window");
  const downloadPromise = page.waitForEvent("download"); await page.click("#download-slice");
  const download = await downloadPromise; assert(download.suggestedFilename().endsWith(".png"));
  await download.saveAs(path.join(output, "download_example.png"));

  const transitions = await page.locator("#transition-filter option").evaluateAll(nodes => nodes.slice(1).map(node => node.value));
  for (const transition of transitions) { await page.selectOption("#transition-filter", transition); assert.equal(await page.locator(".case-row").count(), 5); }
  await page.selectOption("#transition-filter", "all");
  const seenPatients = [];
  for (let index = 0; index < 30; index++) {
    await page.locator(`.case-row[data-index="${index}"]`).click(); await ready(page);
    const pixels = await pixelData(page);
    assert(pixels.colors > 30 && pixels.maximum - pixels.minimum > 40, `blank case ${index + 1}`);
    assert.equal(await page.locator(".slice-tile").count(), 20);
    assert.equal(await page.locator(".slice-tile:disabled").count(), 0);
    seenPatients.push(await page.locator("#patient-name").innerText());
  }
  assert.equal(new Set(seenPatients).size, 30);
  await page.fill("#search", "no-such-patient"); assert.equal(await page.locator(".case-row").count(), 0);
  await page.fill("#search", ""); assert.equal(await page.locator(".case-row").count(), 30);
  await page.locator('.case-row[data-index="0"]').click(); await ready(page);
  await page.click('[data-view="metrics"]'); await ready(page);
  assert.equal(await page.locator("#summary-table tbody tr").count(), 12);
  assert(await page.locator(".plot").evaluateAll(images => images.every(image => image.complete && image.naturalWidth > 100)));
  await page.click('[data-view="slices"]'); await ready(page);
  for (const viewport of [{width: 1920, height: 1080}, {width: 390, height: 844}, {width: 360, height: 780}]) {
    await page.setViewportSize(viewport);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(overflow, false, `horizontal overflow at ${viewport.width}`);
    if (viewport.width < 800) {
      await page.click("#menu"); assert(await page.locator("body").evaluate(body => body.classList.contains("sidebar-open")));
      await page.locator('.case-row[data-index="1"]').click(); await ready(page);
      assert.equal(await page.locator("body").evaluate(body => body.classList.contains("sidebar-open")), false);
    }
    await page.screenshot({path: path.join(output, `viewport_${viewport.width}.png`), fullPage: true});
  }
  assert.deepEqual(errors, []); assert.deepEqual(externalRequests, []);
  fs.writeFileSync(path.join(output, "browser_verification.json"), JSON.stringify({status: "passed", patients: 30, neighbors_per_case: 20, step_counts: [10, 20, 30, 50], viewports: [1440, 1920, 390, 360], offline_file_url: true, identical_checkpoint_pixels: true, sample_switch_changes_pixels: true, edge_slices_not_repeated: true, external_requests: 0, browser_errors: errors}, null, 2) + "\n");
  console.log("Browser verification passed: 30 patients, 20 neighbors, four step counts, both checkpoints, desktop/mobile, offline");
  await browser.close();
}

main().catch(error => { console.error(error); process.exit(1); });
