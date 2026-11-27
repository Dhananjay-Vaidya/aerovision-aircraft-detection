'use strict';
const $ = id => document.getElementById(id);
let metadata, imageFile, busy = false, media = null, socket = null, frameTimer = null, cameraTimer = null;
let inFlight = false, sentFrame = null, lastResult = 0, cameraSequence = -1, activeTab = 'image';
const colors = {allied: '#78dc91', adversary: '#ff777f', uncertain: '#efc16d'};
function message(text = '') { $('message').textContent = text; $('message').hidden = !text; }
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid request. Check your file and settings.');
  return data;
}
function selections() { return {model: $('model').value, risk: $('risk').value}; }
function form(file) { const data = new FormData(); data.append('file', file); for (const [k,v] of Object.entries(selections())) if (v) data.append(k,v); return data; }
function updateTau() {
  const selected = metadata?.models.find(m => m.name === $('model').value);
  $('tau').textContent = selected?.tau[Number($('risk').value).toFixed(2)]?.toFixed(3) ?? '—';
}
function panel(result, fps) {
  const now = performance.now();
  $('fps').textContent = (fps ?? (lastResult ? 1000 / (now-lastResult) : 0)).toFixed(1);
  lastResult = now;
  $('inference').textContent = `${result.inference_ms.toFixed(1)} ms`;
  for (const key of Object.keys(colors)) $(`count-${key}`).textContent = result.counts[key];
  $('detections').replaceChildren();
  for (const d of result.detections) {
    const li = document.createElement('li'); li.style.color = colors[d.decision];
    li.textContent = `${d.class} · ${(d.conf*100).toFixed(0)}% · ${d.decision}`;
    if (d.reason) { const small = document.createElement('small'); small.textContent = d.reason; li.append(small); }
    $('detections').append(li);
  }
  if (!result.detections.length) { const li=document.createElement('li'); li.className='empty'; li.textContent='No aircraft detected above 25% confidence.'; $('detections').append(li); }
}
function draw(canvas, source, detections) {
  const w = source.naturalWidth || source.videoWidth || source.width, h = source.naturalHeight || source.videoHeight || source.height;
  canvas.width=w; canvas.height=h; canvas.hidden=false;
  const ctx=canvas.getContext('2d'); ctx.drawImage(source,0,0,w,h);
  const scale=Math.max(1,w/900); ctx.lineWidth=2*scale; ctx.font=`${14*scale}px Segoe UI, sans-serif`;
  for (const d of detections) {
    const [x,y,x2,y2]=d.box; ctx.strokeStyle=colors[d.decision];
    ctx.setLineDash(d.decision==='adversary'?[10*scale,6*scale]:d.decision==='uncertain'?[2*scale,5*scale]:[]);
    ctx.strokeRect(x,y,x2-x,y2-y); ctx.setLineDash([]);
    const label=`${d.class} · ${(d.conf*100).toFixed(0)}% · ${d.decision}`;
    const textWidth=ctx.measureText(label).width, ly=Math.max(20*scale,y);
    ctx.fillStyle='#0c1117'; ctx.fillRect(x,ly-21*scale,Math.min(w-x,textWidth+12*scale),21*scale);
    ctx.fillStyle=colors[d.decision]; ctx.fillText(label,x+5*scale,ly-5*scale);
  }
}
async function loadImage(url) { const img=new Image(); img.src=url; await img.decode(); return img; }
function chooseImage(file) { if (!file) return; imageFile=file; $('detect').disabled=false; message(`Selected ${file.name}. Click Detect aircraft.`); }
$('image-file').onchange=e=>chooseImage(e.target.files[0]);
for (const event of ['dragenter','dragover']) $('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('drag');});
for (const event of ['dragleave','drop']) $('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('drag');});
$('dropzone').addEventListener('drop',e=>chooseImage(e.dataTransfer.files[0]));
$('severity').oninput=()=>{$('severity-value').value=$('severity').value;};
async function task(button, work) {
  if (busy) return;
  busy=true; button.disabled=true; const old=button.textContent; button.textContent='Processing…'; message(); $('source-status').textContent='Processing';
  try { await work(); $('source-status').textContent='Complete'; }
  catch(e) { message(e.message); $('source-status').textContent='Error'; }
  finally { busy=false;button.disabled=false;button.textContent=old; }
}
$('detect').onclick=()=>task($('detect'),async()=>{
  if (!imageFile) throw new Error('Choose an image first.');
  const data=form(imageFile); data.append('corruption',$('corruption').value);data.append('severity',$('severity').value);
  const started=performance.now(), result=await api('/api/detect',{method:'POST',body:data});
  const img=await loadImage(`data:${result.image_mime};base64,${result.image_base64}`);
  draw($('image-canvas'),img,result.detections); panel(result,1000/(performance.now()-started));
});
let videoSummaries=[];
$('process-video').onclick=()=>task($('process-video'),async()=>{
  const file=$('video-file').files[0]; if(!file) throw new Error('Choose a video file first.');
  const data=form(file); data.append('every_n',$('every-n').value);
  message('Processing video. This may take a few minutes.');
  const result=await api('/api/video',{method:'POST',body:data});videoSummaries=result.frames;
  $('video-result').src=result.url;$('video-result').hidden=false;
  $('video-download').href=result.url;$('video-download').hidden=false;
  message(`Processed ${result.processed_frames} frames. Video is ready.`);
  updateVideoPanel();
});
function updateVideoPanel() {
  const t=$('video-result').currentTime; let entry=videoSummaries[0];
  for (const row of videoSummaries) { if (row.time_s>t) break; entry=row; }
  if(entry) { panel({...entry,detections:[]},0); $('detections').replaceChildren(); const li=document.createElement('li');li.className='empty';li.textContent=`Frame ${entry.frame}: counts shown above. Labels are embedded in the video.`;$('detections').append(li); }
}
$('video-result').ontimeupdate=updateVideoPanel;
$('video-result').onerror=()=>message('The browser could not play this video. Download the MP4, or check the server’s H.264 encoder.');
function stopWebcam() {
  clearInterval(frameTimer);frameTimer=null; if(socket) {socket.onclose=null;socket.close();socket=null;}
  media?.getTracks().forEach(t=>t.stop());media=null;$('webcam-source').srcObject=null;
  inFlight=false;$('webcam-start').disabled=false;$('webcam-stop').disabled=true;
}
$('webcam-start').onclick=async()=>{
  $('webcam-start').disabled=true;message();
  try {
    if(!navigator.mediaDevices) throw new Error('Webcam access requires localhost or HTTPS.');
    media=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:540}},audio:false});
    const video=$('webcam-source');video.srcObject=media;await video.play();
    const params=new URLSearchParams(selections());
    socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/stream?${params}`);
    const capture=document.createElement('canvas');
    socket.onopen=()=>{ $('webcam-stop').disabled=false;lastResult=0;$('source-status').textContent='Live';
      frameTimer=setInterval(()=>{
        if(inFlight||socket?.readyState!==WebSocket.OPEN||!video.videoWidth) return;
        inFlight=true;capture.width=video.videoWidth;capture.height=video.videoHeight;capture.getContext('2d').drawImage(video,0,0);
        sentFrame=document.createElement('canvas');sentFrame.width=capture.width;sentFrame.height=capture.height;sentFrame.getContext('2d').drawImage(capture,0,0);
        capture.toBlob(blob=>{if(blob&&socket?.readyState===WebSocket.OPEN) socket.send(blob); else inFlight=false;},'image/jpeg',.85);
      },100);
    };
    socket.onmessage=e=>{try {const result=JSON.parse(e.data); if(result.error) {message(result.error);stopWebcam();return;} draw($('webcam-canvas'),sentFrame,result.detections);panel(result);}catch(error){message(error.message);}finally{inFlight=false;}};
    socket.onerror=()=>{message('Webcam connection failed. Check that the server is running.');stopWebcam();};
    socket.onclose=()=>{message('Webcam connection closed. Start again to reconnect.');stopWebcam();};
  } catch(e) {stopWebcam();message(`Could not start webcam: ${e.message}`);}
};
$('webcam-stop').onclick=()=>{stopWebcam();$('source-status').textContent='Stopped';};
async function stopCamera() {
  clearInterval(cameraTimer);cameraTimer=null;$('camera-feed').hidden=true;$('camera-feed').removeAttribute('src');
  try {await api('/api/camera/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('camera-source').value})});}
  finally {$('camera-stop').disabled=true;$('camera-start').disabled=false;}
}
$('camera-start').onclick=async()=>{
  $('camera-start').disabled=true;message();cameraSequence=-1;lastResult=0;
  try {
    await api('/api/camera/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source:$('camera-source').value,...selections(),risk:Number($('risk').value)})});
    $('camera-stop').disabled=false;$('source-status').textContent='Connecting';
    let polling=false;
    cameraTimer=setInterval(async()=>{
      if(polling)return;polling=true;
      try {const status=await api('/api/camera/status');if(status.error) throw new Error(status.error);
        if(status.result&&status.sequence!==cameraSequence) {cameraSequence=status.sequence;panel(status.result);$('source-status').textContent='Live';
          if($('camera-feed').hidden) {$('camera-feed').src=`/api/camera/feed?t=${Date.now()}`;$('camera-feed').hidden=false;}}
      }catch(e){message(e.message);await stopCamera().catch(()=>{});}finally{polling=false;}
    },500);
  }catch(e){message(e.message);$('camera-start').disabled=false;}
};
$('camera-stop').onclick=()=>stopCamera().then(()=>{$('source-status').textContent='Stopped';}).catch(e=>message(e.message));
for(const button of document.querySelectorAll('[data-tab]')) button.onclick=async()=>{
  if(button.dataset.tab===activeTab)return;
  stopWebcam();if(cameraTimer)await stopCamera().catch(e=>message(e.message));$('video-result').pause();
  activeTab=button.dataset.tab;
  for(const tab of document.querySelectorAll('[data-tab]')) {const selected=tab===button;tab.setAttribute('aria-selected',selected);$(tab.dataset.tab).hidden=!selected;}
  $('source-status').textContent='Ready';lastResult=0;
};
for(const id of ['model','risk']) $(id).onchange=()=>{updateTau();stopWebcam();if(cameraTimer)stopCamera().catch(e=>message(e.message));};
window.addEventListener('pagehide',()=>{stopWebcam();if(cameraTimer)navigator.sendBeacon('/api/camera/stop',new Blob(['{}'],{type:'application/json'}));});
(async()=>{try{metadata=await api('/api/models');$('model').replaceChildren();for(const model of metadata.models){const o=document.createElement('option');o.value=o.textContent=model.name;$('model').append(o);}$('model').value=metadata.default||'';for(const name of metadata.corruptions){const o=document.createElement('option');o.value=name;o.textContent=name.replaceAll('_',' ');$('corruption').append(o);}updateTau();if(!metadata.models.length)message('No weights found. Place trained .pt files in the models/ folder and refresh this page.');}catch(e){message(e.message);}})();
