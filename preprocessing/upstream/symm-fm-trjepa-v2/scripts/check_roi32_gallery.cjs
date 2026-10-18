const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require('/root/autodl-tmp/.codex/roi32_viewer_browser_20260917/node_modules/playwright');

async function main(){
 const root=path.resolve(process.argv[2]);const gallery=path.join(root,'gallery');const shots=path.join(root,'browser');fs.mkdirSync(shots,{recursive:true});
 const url=pathToFileURL(path.join(gallery,'index.html')).href;
 const browser=await chromium.launch({headless:true,executablePath:'/root/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell'});
 const errors=[];const requests=[];const checks=[];
 async function ready(page,name,z=null){await page.waitForFunction(({name,z})=>{const b=document.querySelector('#board');return b.dataset.case===name&&(z===null||Number(b.dataset.slice)===z)&&b.querySelectorAll('canvas').length>0},{name,z});assert.equal(await page.locator('#error').isVisible(),false)}
 try{
  for(const viewport of [{width:1440,height:1100},{width:768,height:1024},{width:390,height:844},{width:320,height:760}]){
   const page=await browser.newPage({viewport});page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(!r.url().startsWith('file:'))requests.push(r.url())});
   await page.goto(url);await ready(page,'C01');assert.equal(await page.locator('#case option').count(),30);assert.equal(await page.locator('#board canvas').count(),18);
   assert.equal(await page.locator('#metrics tr').count(),6);assert(await page.locator('#previous').isDisabled());
   await page.locator('#next').click();await ready(page,'C02');await page.locator('#previous').click();await ready(page,'C01');
   await page.locator('#slice').fill('0');await page.locator('#slice').dispatchEvent('input');await ready(page,'C01',0);
   await page.locator('#slice').fill('31');await page.locator('#slice').dispatchEvent('input');await ready(page,'C01',31);
   await page.locator('#reset').click();await page.waitForFunction(()=>document.querySelector('#board').dataset.slice!=='31');
   await page.selectOption('#view','error');await page.waitForFunction(()=>document.querySelector('#board').dataset.view==='error');
   await page.selectOption('#phase','1');await page.waitForFunction(()=>document.querySelectorAll('#board canvas').length===6);
   await page.locator('#roi').check();await ready(page,'C01');await page.locator('#roi').uncheck();
   await page.selectOption('#phase','all');await page.selectOption('#view','forecast');await ready(page,'C01');
   if(viewport.width===1440){for(let i=1;i<=30;i++){const c=`C${String(i).padStart(2,'0')}`;await page.selectOption('#case',c);await ready(page,c)}assert(await page.locator('#next').isDisabled())}
   await page.selectOption('#case','C01');await ready(page,'C01');
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   const links=await page.locator('a[href]').evaluateAll(nodes=>nodes.map(x=>x.getAttribute('href')));for(const link of links)fs.accessSync(path.join(gallery,link));
   await page.screenshot({path:path.join(shots,`viewport_${viewport.width}.png`),fullPage:true});
   if(viewport.width===1440)await page.locator('canvas[data-kind="source"][data-phase="1"]').screenshot({path:path.join(shots,'rendered_source.png')});
   await page.goto(url+'#C17/7/error/2');await ready(page,'C17',7);assert.equal(await page.locator('#phase').inputValue(),'2');
   checks.push({viewport,case_count:30,phase_slice_mode_roi_controls:true,hash_navigation:true,no_document_overflow:true,local_links:links.length});await page.close();
  }
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  const report={passed:true,checks,javascript_errors:errors,external_requests:requests,offline_file_open:true};
  fs.writeFileSync(path.join(root,'browser_verification.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close()}
}
main().catch(e=>{console.error(e);process.exitCode=1});
