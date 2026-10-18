"use strict";
const TEXT = {
  headerSubtitle:"\u914d\u51c6 DCE0 \u751f\u6210\u590d\u6838", cases:"\u75c5\u4f8b", allPairs:"\u5168\u90e8\u65f6\u95f4\u5bf9", allCrops:"\u5168\u90e8\u88c1\u526a", contained:"\u6700\u5927\u533a\u57df\u5b8c\u6574", clipped:"\u6700\u5927\u533a\u57df\u622a\u65ad",
  validationOnly:"30 \u4f4d\u9a8c\u8bc1\u60a3\u8005 / \u65e0\u72ec\u7acb\u6d4b\u8bd5\u96c6", steps:"\u91c7\u6837\u6b65\u6570", sample:"\u6837\u672c", window:"\u7070\u5ea6\u7a97", jointWindow:"\u6e90\u56fe + \u76ee\u6807\u8054\u5408\u7a97", referenceSlice:"\u539f\u59cb\u5207\u7247 Z", includeCenter:"\u5305\u542b\u4e2d\u5fc3\u5c42", reportTitle:"30 \u75c5\u4f8b\u8bca\u65ad\u6307\u6807",
  maskNote:"\u7eff\u8272\u8f6e\u5ed3\uff1a\u56fa\u5b9a T0 first-post \u6a21\u578b mask\uff0c\u975e\u968f\u8bbf\u5206\u5272\u3002", generated:"\u751f\u6210\u5207\u7247", targetView:"\u771f\u5b9e\u76ee\u6807\u5207\u7247", stepsView:"\u6b65\u6570\u5bf9\u7167", noiseView:"\u566a\u58f0\u6837\u672c", source:"\u6e90\u56fe", target:"\u771f\u5b9e\u76ee\u6807", prediction:"\u751f\u6210", outOfRange:"\u8d85\u51fa 32 \u5c42", empty:"\u65e0\u5339\u914d\u75c5\u4f8b", loading:"\u8f7d\u5165\u4e2d", center:"\u4e2d\u5fc3\u5c42", mean8:"8 \u6b21\u5747\u503c", close:"\u5173\u95ed", reference:"\u539f\u59cb\u53c2\u8003",
  toSource:"\u751f\u6210\u2192\u6e90\u56fe MAE", toTarget:"\u751f\u6210\u2192\u76ee\u6807 MAE", diversity:"\u4e24\u6837\u672c\u4e4b\u95f4 MAE", metricsNote:"\u5168\u4f53 30 \u4f4d\u60a3\u8005\uff1b\u6e90\u56fe\u4e0e\u76ee\u6807\u6709\u6548\u4f53\u7d20\u4ea4\u96c6\uff1bMAE \u8d8a\u4f4e\u8d8a\u597d\u3002", checkpoint:"\u6743\u91cd", best:"\u6700\u4f18 EMA", last:"\u6700\u7ec8 EMA"
};
const $ = (id) => document.getElementById(id);
const collection = window.REVIEW_COLLECTION || {[window.REVIEW_DATA?.checkpoint || "best"]:window.REVIEW_DATA};
let data = collection.best || collection.last;
const state = {index:0,steps:30,sample:0,preset:"joint",mask:true,z:16,focusZ:16,includeCenter:false,view:"generated",focus:null};
const images = new Map();
let renderToken = 0;
let modal = null;
const fmt = (value) => Number.isFinite(value) ? value.toFixed(3) : "--";
const signed = (value) => (value > 0 ? "+" : "") + value;
const current = () => data.cases[state.index];
const method = () => `s${state.steps}_${state.sample}`;
const icon = (name) => `<i data-lucide="${name}"></i>`;
const hydrateIcons = () => window.lucide.createIcons();

function loadImage(path) {
  if (images.has(path)) return images.get(path);
  const promise = new Promise((resolve,reject) => {
    const image = new Image(); image.onload=()=>resolve(image); image.onerror=()=>reject(new Error("Image failed: " + path)); image.src=path;
  });
  images.set(path,promise);
  if(images.size>36) images.delete(images.keys().next().value);
  return promise;
}

async function draw(canvas, kind, z, snapshot, token) {
  const base = `${snapshot.record.frames}/${kind}_${snapshot.preset}.png`;
  const [image, mask] = await Promise.all([loadImage(base), snapshot.mask ? loadImage(`${snapshot.record.frames}/mask.png`) : Promise.resolve(null)]);
  if(token !== renderToken || !canvas.isConnected) return;
  const ctx=canvas.getContext("2d"); ctx.imageSmoothingEnabled=false;
  const x=(z%8)*128, y=Math.floor(z/8)*128;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(image,x,y,128,128,0,0,canvas.width,canvas.height);
  if(mask) ctx.drawImage(mask,x,y,128,128,0,0,canvas.width,canvas.height);
  canvas.dataset.loaded="true"; canvas.dataset.kind=kind; canvas.dataset.z=String(z);
}

function filtered() {
  const term=$("search").value.trim().toLowerCase(), pair=$("pair-filter").value, crop=$("clip-filter").value;
  return data.cases.map((record,index)=>({record,index})).filter(({record}) => (!term || record.patient_id.toLowerCase().includes(term)) && (!pair || record.time_pair===pair) && (!crop || record.largest_clipped===(crop==="clipped")));
}

function renderList() {
  const rows=filtered(); $("case-count").textContent=`${rows.length} / 30`;
  $("case-list").replaceChildren();
  rows.forEach(({record,index})=>{
    const button=document.createElement("button"); button.className="case-row"+(index===state.index?" active":""); button.dataset.index=index;
    button.setAttribute("aria-current", index===state.index ? "true" : "false");
    button.innerHTML=`<span class="row-number">${String(record.number).padStart(2,"0")}</span><span><span class="row-id">${record.patient_id}</span><span class="row-meta"><span>${record.time_pair.replace("->"," \u2192 ")}</span>${record.largest_clipped?'<span class="clip-dot">\u622a\u65ad</span>':""}</span></span>`;
    button.onclick=()=>selectCase(index); $("case-list").append(button);
  });
  if(!rows.length) {const empty=document.createElement("p");empty.className="empty-list";empty.textContent=TEXT.empty;$("case-list").append(empty);}
}

function closeSidebar() {document.body.classList.remove("drawer-open");$("scrim").hidden=true;}
function selectCase(index) {
  state.index=index;state.z=current().anchor;state.focusZ=state.z;state.focus=null;
  closeSidebar();renderList();render();
  try{history.replaceState(null,"",`#${current().key}`);}catch{}
}
function moveCase(delta) {const rows=filtered();const i=rows.findIndex(row=>row.index===state.index);const next=i<0?0:Math.min(rows.length-1,Math.max(0,i+delta));if(rows[next])selectCase(rows[next].index);}
function setSlice(value){state.z=Math.max(0,Math.min(31,Math.round(Number(value)||0)));state.focusZ=state.z;state.focus=null;render();}

function reference(kind,title,z,snapshot,token,subtitle) {
  const figure=document.createElement("figure");figure.className="reference";
  figure.innerHTML=`<figcaption><span>${title}</span><button class="icon-button" title="\u653e\u5927" aria-label="\u653e\u5927 ${title}">${icon("maximize-2")}</button></figcaption><canvas width="256" height="256" aria-label="${title}"></canvas><div class="reference-footer"><span>Z ${z} / 31</span><span>${subtitle}</span></div>`;
  figure.querySelector("button").onclick=()=>openImage(kind,title,z);
  $("references").append(figure);
  return draw(figure.querySelector("canvas"),kind,z,snapshot,token);
}

function makeTile(item,snapshot,token) {
  const figure=document.createElement("figure");figure.className="slice-tile";
  if(item.z===state.z)figure.classList.add("center");
  if(item.kind===(state.focus?.kind||method())&&item.z===state.focusZ)figure.classList.add("selected");
  if(item.z<0||item.z>31) {
    figure.innerHTML=`<div class="out-of-range">${icon("scan-line")}<span>${TEXT.outOfRange}</span></div><figcaption><span>Z ${item.z}</span><span>${item.detail}</span></figcaption>`;
    $("slice-grid").append(figure);return Promise.resolve();
  }
  figure.innerHTML=`<button class="tile-button" title="${item.label} | ${item.detail}"><canvas width="128" height="128" aria-label="${item.label}"></canvas></button><figcaption><span>${item.label}</span><span>${item.detail}</span></figcaption>`;
  const button=figure.querySelector("button");
  button.onclick=()=>{state.focusZ=item.z;state.focus={kind:item.kind,title:item.title||item.label};render();};
  button.ondblclick=()=>openImage(item.kind,item.title||item.label,item.z);
  $("slice-grid").append(figure);return draw(figure.querySelector("canvas"),item.kind,item.z,snapshot,token);
}

async function render() {
  const token=++renderToken, record=current();
  $("slice-grid").dataset.ready="false";
  const snapshot={record,preset:state.preset,mask:state.mask};
  $("error-message").hidden=true;$("grid-status").textContent=TEXT.loading;
  $("case-number").textContent=`${TEXT.cases} ${String(record.number).padStart(2,"0")} / 30`;
  $("patient-title").textContent=record.patient_id;$("pair-title").textContent=record.time_pair.replace("->"," \u2192 ");
  $("crop-status").textContent=TEXT[record.largest_clipped?"clipped":"contained"];$("crop-status").className=record.largest_clipped?"clipped":"";
  $("slice-range").value=state.z;$("slice-number").value=state.z;$("slice-minus").disabled=state.z===0;$("slice-plus").disabled=state.z===31;
  const visible=filtered(),visibleIndex=visible.findIndex(row=>row.index===state.index);
  $("previous-case").disabled=visibleIndex===0||!visible.length;$("next-case").disabled=visibleIndex===visible.length-1||!visible.length;
  document.querySelectorAll("[data-step]").forEach(button=>button.setAttribute("aria-checked",String(Number(button.dataset.step)===state.steps)));
  document.querySelectorAll("[data-view]").forEach(button=>button.setAttribute("aria-selected",String(button.dataset.view===state.view)));
  $("inference-meta").textContent=`Heun \u00b7 NFE ${state.steps*2} \u00b7 seed ${record.seeds[state.sample]}`;
  $("checkpoint-meta").textContent=`${TEXT[data.checkpoint]} / ${data.checkpoint_reference_step}`;
  $("download-metrics").href=data.diagnostics_path||"diagnostics.json";
  const limits=state.preset==="joint"?record.window:[-1,3];$("window-label").textContent=`${TEXT.window} [${limits[0].toFixed(2)}, ${limits[1].toFixed(2)}]`;
  const focus=state.focus||{kind:method(),title:`${TEXT.prediction} ${state.steps} \u6b65 / ${state.sample+1}`};
  $("references").replaceChildren();
  const jobs=[reference("source",`${TEXT.source} ${record.time_pair.slice(0,2)}`,state.z,snapshot,token,TEXT.reference),reference("target",`${TEXT.target} ${record.time_pair.slice(-2)}`,state.z,snapshot,token,TEXT.reference),reference(focus.kind,focus.title,state.focusZ,snapshot,token,`${signed((state.focusZ-state.z)*2)} mm`)];
  const metric=record.steps.find(row=>row.steps===state.steps);
  $("case-metrics").innerHTML=[[TEXT.toSource,metric.source_mae],[TEXT.toTarget,metric.target_mae],[TEXT.diversity,metric.sample_diversity_mae]].map(([name,value])=>`<div class="metric"><span>${name}</span><strong>${fmt(value)}</strong></div>`).join("");
  $("slice-grid").replaceChildren();$("slice-grid").className="slice-grid";
  $("center-control").hidden=!(state.view==="generated"||state.view==="target");
  const tiles=[];
  if(state.view==="generated"||state.view==="target"){
    const kind=state.view==="generated"?method():"target";
    for(let offset=-10;offset<=10;offset++)if(offset!==0||state.includeCenter)tiles.push({kind,z:state.z+offset,label:`Z ${state.z+offset}`,detail:offset===0?TEXT.center:`${signed(offset*2)} mm`,title:state.view==="generated"?`${TEXT.prediction} ${state.steps} \u6b65 / ${state.sample+1}`:TEXT.target});
    $("grid-title").textContent=`${state.view==="generated"?TEXT.generated:TEXT.targetView} | Z ${state.z-10}\u2013${state.z+10}`;
  }else if(state.view==="steps"){
    $("slice-grid").classList.add("comparison");
    data.steps.forEach(steps=>tiles.push({kind:`s${steps}_${state.sample}`,z:state.focusZ,label:`${steps} \u6b65`,detail:`NFE ${steps*2}`,title:`${TEXT.prediction} ${steps} \u6b65 / ${state.sample+1}`}));
    $("grid-title").textContent=`${TEXT.stepsView} | Z ${state.focusZ} | seed ${record.seeds[state.sample]}`;
  }else{
    $("slice-grid").classList.add("noise");
    for(let sample=0;sample<8;sample++)tiles.push({kind:`s25_${sample}`,z:state.focusZ,label:`${TEXT.sample} ${sample+1}`,detail:"25 \u6b65",title:`25 \u6b65 / ${TEXT.sample} ${sample+1}`});
    tiles.push({kind:"mean25",z:state.focusZ,label:TEXT.mean8,detail:"25 \u6b65",title:TEXT.mean8});
    $("grid-title").textContent=`${TEXT.noiseView} | Heun 25 | Z ${state.focusZ}`;
  }
  jobs.push(...tiles.map(item=>makeTile(item,snapshot,token)));hydrateIcons();
  try{await Promise.all(jobs);if(token===renderToken){const valid=tiles.filter(item=>item.z>=0&&item.z<32).length;$("grid-status").textContent=`${valid} / ${tiles.length} \u5f20`;$('slice-grid').dataset.ready="true";}}
  catch(error){if(token===renderToken){$("error-message").hidden=false;$("error-message").textContent=error.message;$("grid-status").textContent="\u8f7d\u5165\u5931\u8d25";}}
}

function openImage(kind,title,z){modal={kind,title,z};$("image-dialog").showModal();renderModal();}
function renderModal(){if(!modal)return;$("image-title").textContent=`${current().patient_id} | ${modal.title}`;$("large-position").textContent=`Z ${modal.z} / 31 | ${signed((modal.z-state.z)*2)} mm`;$("large-previous").disabled=modal.z===0;$("large-next").disabled=modal.z===31;draw($("large-image"),modal.kind,modal.z,{record:current(),preset:state.preset,mask:state.mask},renderToken).catch(error=>{$("large-position").textContent=error.message;});}

function renderReport(){
  const d=data.diagnostics,a=d.aggregate;
  const rows=[["\u6e90\u56fe\u2192\u771f\u5b9e\u76ee\u6807",a.source_target_mae.mean],["\u5355\u6b21\u751f\u6210\u2192\u771f\u5b9e\u76ee\u6807",a.sample_target_mae.mean],["\u5355\u6b21\u751f\u6210\u2192\u6e90\u56fe",a.sample_source_mae.mean],["8 \u4e2a\u6837\u672c\u4e24\u4e24\u5dee\u5f02",a.sample_pairwise_mae.mean],["8 \u6b21\u5747\u503c\u2192\u771f\u5b9e\u76ee\u6807",a.mean_target_mae.mean]];
  $("report-content").innerHTML=`<p>${TEXT.metricsNote}</p><h3>25 \u6b65 / 8 \u4e2a\u566a\u58f0\u6837\u672c</h3><table><thead><tr><th>\u6bd4\u8f83</th><th>MAE</th></tr></thead><tbody>${rows.map(([label,value])=>`<tr><td>${label}</td><td>${fmt(value)}</td></tr>`).join("")}</tbody></table><p>\u6837\u672c\u5dee\u5f02 &lt; 0.000001\uff1a${d.near_zero_diversity_cases}/30\u3002\u751f\u6210\u66f4\u63a5\u8fd1\u6e90\u56fe\uff1a${d.closer_to_source_cases}/30\u3002</p><h3>\u540c\u566a\u58f0\u6b65\u6570\u5bf9\u7167 / \u6bcf\u4f8b 2 \u4e2a\u6837\u672c</h3><div class="table-scroll"><table><thead><tr><th>\u6b65\u6570</th><th>\u2192\u76ee\u6807 MAE</th><th>\u2192\u6e90\u56fe MAE</th><th>\u4e0e 50 \u6b65\u5dee\u5f02</th></tr></thead><tbody>${d.step_comparison.map(row=>`<tr><td>${row.steps}</td><td>${fmt(row.target_mae)}</td><td>${fmt(row.source_mae)}</td><td>${fmt(row.difference_from_50)}</td></tr>`).join("")}</tbody></table></div><h3>\u524d\u540e 10 \u5c42\u5339\u914d / \u63a2\u7d22\u6027</h3><table><thead><tr><th>\u6bd4\u8f83</th><th>\u540c\u5c42 MAE</th><th>\u7a97\u53e3\u6700\u5c0f MAE</th></tr></thead><tbody><tr><td>\u751f\u6210\u2192\u76ee\u6807</td><td>${fmt(a.same_slice_generated_mae.mean)}</td><td>${fmt(a.best_window_generated_mae.mean)}</td></tr><tr><td>\u6e90\u56fe\u2192\u76ee\u6807</td><td>${fmt(a.same_slice_source_mae.mean)}</td><td>${fmt(a.best_window_source_mae.mean)}</td></tr></tbody></table><p>\u7a97\u53e3\u6700\u5c0f\u8bef\u5dee\u4f7f\u7528\u771f\u5b9e\u76ee\u6807\u9009\u5c42\uff0c\u4e0d\u4ee3\u66ff\u56fa\u5b9a\u5750\u6807\u8bc4\u4f30\uff0c\u4e5f\u4e0d\u80fd\u5355\u72ec\u8bc1\u660e\u914d\u51c6\u9519\u8bef\u3002</p>`;
}

function init(){
  document.querySelectorAll("[data-text]").forEach(el=>el.textContent=TEXT[el.dataset.text]);
  Object.keys(collection).forEach(key=>$("checkpoint-select").add(new Option(TEXT[key],key)));
  $("checkpoint-select").value=data.checkpoint;
  $("checkpoint-select").onchange=event=>{const patient=current().patient_id;data=collection[event.target.value];state.index=data.cases.findIndex(record=>record.patient_id===patient);state.focus=null;renderList();renderReport();render();};
  data.steps.forEach(steps=>{const button=document.createElement("button");button.textContent=steps;button.dataset.step=steps;button.setAttribute("role","radio");button.onclick=()=>{state.steps=steps;state.focus=null;render();};$("step-buttons").append(button);});
  [["generated",TEXT.generated],["target",TEXT.targetView],["steps",TEXT.stepsView],["noise",TEXT.noiseView]].forEach(([view,label])=>{const button=document.createElement("button");button.textContent=label;button.dataset.view=view;button.setAttribute("role","tab");button.onclick=()=>{state.view=view;render();};$("view-tabs").append(button);});
  [...new Set(data.cases.map(record=>record.time_pair))].forEach(pair=>{const option=new Option(pair.replace("->"," \u2192 "),pair);$("pair-filter").add(option);});
  ["search","pair-filter","clip-filter"].forEach(id=>$(id).addEventListener(id==="search"?"input":"change",()=>{renderList();render();}));
  $("previous-case").onclick=()=>moveCase(-1);$("next-case").onclick=()=>moveCase(1);
  $("slice-range").oninput=event=>setSlice(event.target.value);$("slice-number").onchange=event=>setSlice(event.target.value);
  $("slice-minus").onclick=()=>setSlice(state.z-1);$("slice-plus").onclick=()=>setSlice(state.z+1);$("reset-slice").onclick=()=>setSlice(current().anchor);
  $("sample-select").onchange=event=>{state.sample=Number(event.target.value);state.focus=null;render();};
  $("window-select").onchange=event=>{state.preset=event.target.value;render();};$("mask-toggle").onchange=event=>{state.mask=event.target.checked;render();};
  $("center-toggle").onchange=event=>{state.includeCenter=event.target.checked;render();};
  $("menu-button").onclick=()=>{const open=document.body.classList.toggle("drawer-open");$("scrim").hidden=!open;};$("scrim").onclick=closeSidebar;
  $("report-button").onclick=()=>{$("report-dialog").showModal();};
  document.querySelectorAll("[data-close]").forEach(button=>button.onclick=()=>$(button.dataset.close).close());
  $("large-previous").onclick=()=>{modal.z=Math.max(0,modal.z-1);renderModal();};$("large-next").onclick=()=>{modal.z=Math.min(31,modal.z+1);renderModal();};
  document.addEventListener("keydown",event=>{if(["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName)||document.querySelector("dialog[open]"))return;if(event.key==="ArrowLeft")moveCase(-1);if(event.key==="ArrowRight")moveCase(1);});
  const key=location.hash.slice(1), index=data.cases.findIndex(record=>record.key===key);
  renderReport();selectCase(index<0?0:index);hydrateIcons();
}
if(data&&data.cases&&data.cases.length===30)init();else{$("error-message").hidden=false;$("error-message").textContent="Review data is incomplete.";}
