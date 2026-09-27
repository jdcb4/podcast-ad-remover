(() => {
 'use strict';
 const form=document.getElementById('config'), result=document.getElementById('result'), output=document.getElementById('output');
 const release=window.INSTALL_RELEASE;
 document.getElementById('release').textContent=`${release.channel} · ${release.image} · ${release.revision}`;
 let filename='compose.yaml', generated='';
 const shell=value=>"'"+String(value).replaceAll("'","'\"'\"'")+"'";
 // Single-quoted YAML strings, with Compose interpolation escaped separately.
 const yaml=value=>"'"+String(value).replaceAll("'","''").replaceAll('$',()=> '$$')+"'";
 form.elements.storage.addEventListener('change',()=>{form.elements.path.disabled=form.elements.storage.value!=='bind';form.elements.path.required=!form.elements.path.disabled;});
 form.elements.provider.addEventListener('change',()=>{form.elements.key.disabled=!form.elements.provider.value;});
 function clear(){form.elements.key.value='';generated='';output.textContent='';result.hidden=true;}
 document.getElementById('clear').addEventListener('click',clear);
 form.addEventListener('submit',event=>{
  event.preventDefault();
  const v=Object.fromEntries(new FormData(form)), errors=[];
  let url;try{url=new URL(v.url);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)errors.push('Use an HTTP or HTTPS application URL without credentials, query or fragment.');}catch(_){errors.push('Enter a valid application URL.');}
  if(!Number.isInteger(Number(v.port))||Number(v.port)<1||Number(v.port)>65535)errors.push('Port must be between 1 and 65535.');
  if(v.storage==='bind'&&(!v.path||/[\r\n,:]/.test(v.path)||!v.path.startsWith('/')))errors.push('Use an absolute Linux host directory without commas, colons or newlines.');
  if(v.key&&/[\r\n\0]/.test(v.key))errors.push('API keys cannot contain newlines.');
  result.hidden=false;
  if(errors.length){generated='';output.textContent='';document.getElementById('status').textContent=errors.join(' ');return;}
  const bytes=new Uint8Array(32);crypto.getRandomValues(bytes);
  const secret=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
  const env={SESSION_SECRET_KEY:secret,BASE_URL:url.href.replace(/\/$/,''),WHISPER_MODEL:v.model,LOG_LEVEL:v.log};
  if(v.https){env.COOKIE_SECURE='true';env.TRUST_PROXY_HEADERS='true';}
  if(v.provider&&v.key)env[v.provider]=v.key;
  const volume=v.storage==='bind'?v.path:'podcast-data';
  if(v.format==='compose'){
   filename='compose.yaml';
   generated=`services:\n  podcast-ad-remover:\n    image: ${yaml(release.image)}\n    restart: unless-stopped\n    ports:\n      - ${yaml(v.port+':8000')}\n    volumes:\n      - ${yaml(volume+':/data')}\n    environment:\n`+Object.entries(env).map(([k,x])=>`      ${k}: ${yaml(x)}`).join('\n')+'\n';
   if(v.gpu)generated+='    deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia\n              count: all\n              capabilities: [gpu]\n';
   if(v.storage!=='bind')generated+='volumes:\n  podcast-data:\n';
   document.getElementById('command').textContent='Save as compose.yaml, then run: docker compose up -d';
  }else{
   filename='install.sh';
   generated='#!/bin/sh\nset -eu\n'+`docker run -d --name podcast-ad-remover --restart unless-stopped \\\n  -p ${shell(v.port+':8000')} \\\n  --mount ${shell(`type=${v.storage==='bind'?'bind':'volume'},source=${volume},target=/data`)} \\\n`+Object.entries(env).map(([k,x])=>`  -e ${shell(k+'='+x)} \\`).join('\n')+'\n'+(v.gpu?'  --gpus all \\\n':'')+`  ${shell(release.image)}\n`;
   document.getElementById('command').textContent='Save as install.sh on your Docker host, then run: sh install.sh';
  }
  output.textContent=generated;document.getElementById('status').textContent='Generated locally. Review before running.';
 });
 document.getElementById('copy').addEventListener('click',async()=>{try{if(generated)await navigator.clipboard.writeText(generated);document.getElementById('status').textContent='Copied';}catch(_){document.getElementById('status').textContent='Clipboard unavailable. Select and copy the output, or download it.';}});
 document.getElementById('download').addEventListener('click',()=>{if(!generated)return;const url=URL.createObjectURL(new Blob([generated],{type:'text/plain'}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 window.addEventListener('pagehide',clear);
})();
