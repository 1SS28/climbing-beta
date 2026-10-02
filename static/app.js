const $ = s => document.querySelector(s);
const canvas = $('#c'), ctx = canvas.getContext('2d');
let photo = null, img = null, points = [], refs = [], mode = 'select', moving = null;
let gaps = [], busy = false, revision = 0, scale = 1;
const referenceCount = () => $('#method').value === 'grid' ? 4 : 2;
const positive = id => Number.isFinite(+$(id).value) && +$(id).value > 0;
function calibrated() {
  return refs.length === referenceCount() && ($('#method').value === 'reference' ? positive('#metres') :
    positive('#spacing') && positive('#cols') && positive('#rows') && Number.isInteger(+$('#cols').value) && Number.isInteger(+$('#rows').value));
}
function update() {
  const available = !!img && !busy;
  for (const id of ['fit','zoom-in','zoom-out','calibrate','select','manual']) $('#'+id).disabled = !available;
  $('#file').disabled = busy;
  $('#undo').disabled = $('#clear').disabled = !available || !points.length;
  $('#undo-reference').disabled = !available || !refs.length;
  $('#measure').disabled = !available || points.length < 2 || !calibrated();
  $('#select').setAttribute('aria-pressed', mode === 'select');
  $('#manual').setAttribute('aria-pressed', mode === 'manual');
  $('#calibration-status').textContent = `${refs.length} of ${referenceCount()} reference points`;
  $('#ready').textContent = !img ? 'Upload a photo first.' : !calibrated() ? 'Complete the reference points and their real dimensions.' : points.length < 2 ? 'Select at least two route points.' : `${points.length} points ready to measure.`;
  const list = $('#route-list'); list.replaceChildren();
  points.forEach((p, i) => {
    const li = document.createElement('li');
    li.append(`${p.hold == null ? 'Manual point' : 'Detected hold'} `);
    for (const [label, action] of [
      ['Move', () => { moving=i; mode='move'; $('#status').textContent=`Tap the new position for point ${i+1}.`; update(); }],
      ['Remove', () => { points.splice(i,1); moving=null; mode='select'; changed(); }],
      ...(i ? [['Earlier', () => { [points[i-1],points[i]]=[points[i],points[i-1]]; moving=null; mode='select'; changed(); }]] : [])
    ]) { const b=document.createElement('button'); b.textContent=label; b.setAttribute('aria-label',`${label} point ${i+1}`); b.disabled=busy; b.onclick=action; li.append(b); }
    list.append(li);
  });
}
function changed() { if ($('#status').textContent.startsWith('Distances ready')) $('#status').textContent='Inputs changed. Measure again to update distances.'; revision++; gaps=[]; $('#out').hidden=true; $('#error').textContent=''; update(); draw(); }
function label(text,x,y,k,color) {
  ctx.font=`600 ${14*k}px system-ui`; const width=ctx.measureText(text).width;
  ctx.fillStyle='#10141deb'; ctx.fillRect(x-4*k,y-16*k,width+8*k,22*k);
  ctx.fillStyle=color; ctx.fillText(text,x,y);
}
function draw() {
  if (!img) return;
  ctx.drawImage(img,0,0); const k=1/scale;
  ctx.lineWidth=1.5*k; ctx.strokeStyle='#ffffff66';
  for (const hold of photo.holds) {
    ctx.beginPath(); hold.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y)); ctx.closePath();ctx.stroke();
  }
  for (const g of gaps) {
    const a=points[g.from].xy,b=points[g.to].xy;
    ctx.strokeStyle='#ffd082';ctx.lineWidth=2*k;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
    label(`${g.metres.toFixed(2)} m`,(a[0]+b[0])/2,(a[1]+b[1])/2,k,'#ffd082');
  }
  for (const [collection,color,names] of [[points.map(p=>p.xy),'#69b2ff',null],[refs,'#ffd082',referenceCount()===4?['TL','TR','BL','BR']:['A','B']]]) {
    collection.forEach(([x,y],i)=>{ctx.strokeStyle=color;ctx.lineWidth=2*k;ctx.beginPath();ctx.arc(x,y,6*k,0,Math.PI*2);ctx.stroke();label(names?names[i]:String(i+1),x+9*k,y-9*k,k,color);});
  }
}
function zoom(value) {
  if (!img) return;
  scale=Math.max(.03,Math.min(4,value)); canvas.style.width=`${img.width*scale}px`; canvas.style.height=`${img.height*scale}px`; draw();
}
function fit() { if(img) zoom(Math.min($('#stage').clientWidth/img.width, $('#stage').clientHeight/img.height)); }
$('#fit').onclick=fit; $('#zoom-in').onclick=()=>zoom(scale*1.5); $('#zoom-out').onclick=()=>zoom(scale/1.5);
window.addEventListener('resize',fit);
// A scroll gesture must not also place a point when it ends.
let down=null;
canvas.onpointerdown=e=>{ down=[e.clientX,e.clientY]; };
canvas.onpointercancel=()=>{down=null;};
canvas.onclick=e=>{
  if (!img || busy || !down || Math.hypot(e.clientX-down[0],e.clientY-down[1])>8) return;
  down=null;
  const r=canvas.getBoundingClientRect(),xy=[(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height];
  if (mode==='calibrate') {
    refs.push(xy);
    if (refs.length===referenceCount()) { mode='select'; $('#status').textContent='Reference points set. Select or add route points.'; }
    else $('#status').textContent=`Pick reference point ${refs.length+1} of ${referenceCount()}.`;
  } else if (mode==='move' && moving!==null) {
    points[moving]={xy,hold:null}; moving=null;mode='select';$('#status').textContent='Point moved.';
  } else if (mode==='manual') points.push({xy,hold:null});
  else {
    const nearest=photo.holds.map(h=>({h,d:Math.hypot(h.centre[0]-xy[0],h.centre[1]-xy[1])*scale})).sort((a,b)=>a.d-b.d)[0];
    if (!nearest || nearest.d>24) { $('#status').textContent='No hold nearby. Choose Add points to place a point here.';return; }
    const existing=points.findIndex(p=>p.hold===nearest.h.id);
    if (existing>=0) points.splice(existing,1); else points.push({xy:[...nearest.h.centre],hold:nearest.h.id});
  }
  changed();
};
$('#select').onclick=()=>{mode='select';moving=null;$('#status').textContent='Tap detected holds in route order.';update();};
$('#manual').onclick=()=>{mode='manual';moving=null;$('#status').textContent='Tap anywhere on the wall to add a measurement point.';update();};
$('#clear').onclick=()=>{points=[];moving=null;mode='select';changed();};
$('#undo').onclick=()=>{points.pop();moving=null;mode='select';changed();};
$('#calibrate').onclick=()=>{refs=[];moving=null;mode='calibrate';$('#status').textContent=referenceCount()===4?'Pick top-left, top-right, bottom-left, bottom-right bolt holes.':'Pick the two endpoints of your known distance.';changed();};
$('#undo-reference').onclick=()=>{refs.pop();moving=null;mode='calibrate';changed();};
$('#method').onchange=()=>{
  refs=[];mode='select';moving=null;
  $('#reference-fields').hidden=referenceCount()===4;$('#grid-fields').hidden=referenceCount()===2;changed();
};
for (const id of ['metres','cols','rows','spacing']) $('#'+id).oninput=changed;
async function request(url,options) {
  const response=await fetch(url,options);
  let result;try {result=await response.json();}catch {throw new Error('The server returned an unreadable response. Please try again.');}
  if (!response.ok) throw new Error(typeof result.detail==='string'?result.detail:'Check the reference dimensions and selected points, then try again.');
  return result;
}
$('#file').onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  busy=true;photo=null;img=null;points=[];refs=[];mode='select';moving=null;canvas.hidden=true;$('#empty').hidden=false;$('#detected').textContent='';changed();
  $('#status').textContent='Detecting holds…';
  try {
    const form=new FormData();form.append('file',file);
    photo=await request('/api/photo',{method:'POST',body:form});
    const loaded=new Image();loaded.src=`/api/photo/${photo.id}.jpg`;await loaded.decode();img=loaded;
    canvas.width=img.width;canvas.height=img.height;canvas.hidden=false;$('#empty').hidden=true;fit();
    $('#detected').textContent=`${photo.holds.length} holds detected. You can also add points manually.`;
    $('#status').textContent='Set a known reference and select one route.';
  }catch(err){$('#error').textContent=err.message;$('#status').textContent='Photo could not be loaded.';}
  finally{busy=false;$('#file').value='';update();}
};
$('#measure').onclick=async()=>{
  busy=true;$('#error').textContent='';$('#status').textContent='Measuring…';update();const version=revision;
  const calibration=referenceCount()===2?{method:'reference',points:refs,metres:+$('#metres').value}:{method:'grid',corners:refs,cols:+$('#cols').value,rows:+$('#rows').value,spacing_m:+$('#spacing').value};
  try {
    const result=await request('/api/measure',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:photo.id,points:points.map(p=>p.xy),calibration})});
    if(version!==revision){$('#status').textContent='Inputs changed. Measure again to update distances.';return;}
    gaps=result.gaps;draw();$('#out').hidden=false;const target=$('#result');target.replaceChildren();
    for(const text of [`${result.total_m.toFixed(2)} m sum of gaps · ${result.rise_m.toFixed(2)} m span along the wall’s up-axis`,result.basis,result.note]){const p=document.createElement('p');p.textContent=text;target.append(p);}
    if(result.warning){const p=document.createElement('p');p.className='warning';p.textContent=result.warning;target.append(p);}
    const list=document.createElement('ol');for(const gap of gaps){const li=document.createElement('li');li.textContent=`Point ${gap.from+1} → ${gap.to+1}: ${gap.metres.toFixed(2)} m`;list.append(li);}target.append(list);
    $('#status').textContent='Distances ready. Adjust a point or reference to measure again.';
  }catch(err){if(version===revision){$('#error').textContent=err.message;$('#status').textContent='Check your inputs and try again.';}}
  finally{busy=false;update();}
};
update();
