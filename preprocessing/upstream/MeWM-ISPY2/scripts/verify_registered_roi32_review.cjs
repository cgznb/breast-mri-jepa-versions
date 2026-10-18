"use strict";
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const assert = require("node:assert/strict");
const {chromium} = require("/root/autodl-tmp/.codex/roi32_viewer_browser_20260917/node_modules/playwright");

async function main() {
  const root=path.resolve(process.argv[2]);
  const output=path.join(root,"verification");fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true});
  const errors=[];
  let inspected=0;
  const ready=async(page)=>page.waitForFunction(()=>document.querySelector("#slice-grid")?.dataset.ready==="true" && [...document.querySelectorAll("#references canvas,#slice-grid canvas")].every(canvas=>canvas.dataset.loaded==="true"));
  const captureErrors=page=>{page.on("pageerror",error=>errors.push(error.message));page.on("console",message=>{if(message.type()==="error")errors.push(message.text());});};
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});captureErrors(page);
    await page.goto(pathToFileURL(path.join(root,"index.html")).href);await ready(page);
    assert.equal(await page.locator(".case-row").count(),30);
    assert.equal(await page.locator("#checkpoint-select option").count(),2);
    assert.equal(await page.locator("#slice-grid .slice-tile").count(),20);
    await page.locator("#center-toggle").check();await ready(page);
    assert.equal(await page.locator("#slice-grid .slice-tile").count(),21);
    await page.locator("#slice-number").fill("0");await page.locator("#slice-number").press("Tab");await ready(page);
    assert.equal(await page.locator("#slice-grid .out-of-range").count(),10);
    assert.equal(await page.locator("#slice-grid canvas").count(),11);
    await page.locator("#slice-number").fill("16");await page.locator("#slice-number").press("Tab");await ready(page);
    assert.equal(await page.locator("#slice-grid canvas").count(),21);
    await page.locator("#center-toggle").uncheck();await ready(page);
    const sourceBefore=await page.locator("#references canvas").first().screenshot();
    const bestImage=await page.locator("#references canvas").nth(2).screenshot();
    await page.locator("#checkpoint-select").selectOption("last");await ready(page);
    const sourceAfter=await page.locator("#references canvas").first().screenshot();
    const lastImage=await page.locator("#references canvas").nth(2).screenshot();
    assert(sourceBefore.equals(sourceAfter),"Checkpoint selection changed source image");
    assert(!bestImage.equals(lastImage),"Best/final forecast images unexpectedly identical");
    for(const checkpoint of ["best","last"]){
      await page.locator("#checkpoint-select").selectOption(checkpoint);await ready(page);
      for(let i=0;i<30;i++){
        await page.locator(".case-row").nth(i).click();await ready(page);
        assert.equal(await page.locator("#error-message").isVisible(),false);
        assert.equal(await page.locator("#slice-grid .slice-tile").count(),20);
        inspected++;
      }
    }
    await page.locator(".case-row").first().click();await ready(page);
    for(const steps of [10,20,30,50])for(const sample of ["0","1"]){
      await page.locator(`[data-step="${steps}"]`).click();await page.locator("#sample-select").selectOption({value:sample});await ready(page);
      assert.equal(await page.locator("#references canvas").nth(2).getAttribute("data-kind"),`s${steps}_${sample}`);
    }
    await page.locator('[data-view="steps"]').click();await ready(page);
    assert.equal(await page.locator("#slice-grid canvas").count(),4);
    await page.locator('[data-view="noise"]').click();await ready(page);
    assert.equal(await page.locator("#slice-grid canvas").count(),9);
    const sampleOne=await page.locator("#slice-grid canvas").nth(0).screenshot();
    const sampleTwo=await page.locator("#slice-grid canvas").nth(1).screenshot();
    assert(!sampleOne.equals(sampleTwo),"Noise sample atlases unexpectedly identical");
    await page.locator('[data-view="target"]').click();await ready(page);
    assert(await page.locator("#slice-grid canvas").first().getAttribute("data-kind")==="target");
    await page.locator('[data-view="generated"]').click();await ready(page);
    await page.locator("#window-select").selectOption("fixed");await ready(page);
    await page.locator("#mask-toggle").uncheck();await ready(page);
    await page.locator("#references button").first().click();
    await page.waitForFunction(()=>document.querySelector("#large-image").dataset.loaded==="true");
    assert(await page.locator("#image-dialog").isVisible());await page.locator('[data-close="image-dialog"]').click();
    await page.locator("#report-button").click();assert(await page.locator("#report-dialog table").count()===3);
    await page.locator('[data-close="report-dialog"]').click();
    await page.locator("#search").fill("NO_SUCH_PATIENT");assert.equal(await page.locator(".case-row").count(),0);
    await page.locator("#search").fill("");await page.locator("#pair-filter").selectOption("T0->T1");assert.equal(await page.locator(".case-row").count(),5);
    await page.locator("#pair-filter").selectOption("");await page.locator("#clip-filter").selectOption("clipped");assert.equal(await page.locator(".case-row").count(),2);
    await page.locator("#clip-filter").selectOption("");await page.locator(".case-row").first().click();await ready(page);
    await page.locator("#checkpoint-select").selectOption("best");await page.locator("#window-select").selectOption("joint");await page.locator("#mask-toggle").check();await page.locator('[data-step="30"]').click();await page.locator("#sample-select").selectOption({value:"0"});await ready(page);
    await page.screenshot({path:path.join(output,"desktop.png"),fullPage:true});
    await page.locator("#references canvas").first().screenshot({path:path.join(output,"source-canvas.png")});
    const desktopOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
    assert.equal(desktopOverflow,false,"Desktop horizontal overflow");
    for(const viewport of [{width:390,height:844},{width:768,height:1024}]){
      const mobile=await browser.newPage({viewport});captureErrors(mobile);
      await mobile.goto(pathToFileURL(path.join(root,"index.html")).href);await ready(mobile);
      assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,"Responsive horizontal overflow");
      if(viewport.width===390){await mobile.locator("#menu-button").click();await mobile.locator(".case-row").nth(29).click();await ready(mobile);assert.equal(await mobile.locator("#scrim").isVisible(),false);}
      await mobile.screenshot({path:path.join(output,viewport.width===390?"mobile.png":"tablet.png"),fullPage:true});
      await mobile.close();
    }
    assert.deepEqual(errors,[]);
    const result={status:"passed",checkpointPatientViews:inspected,desktop:[1440,1000],mobile:[390,844],tablet:[768,1024],steps:[10,20,30,50],samples:2,neighborSlots:20,withCenter:21,outOfBoundsPlaceholders:true,identicalSourceAcrossCheckpoints:true,differentForecastsAcrossCheckpoints:true,differentNoiseSamples:true,errors};
    fs.writeFileSync(path.join(output,"browser.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
  } finally {await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
