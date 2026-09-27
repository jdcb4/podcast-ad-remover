(() => {
 'use strict';
 const form=document.getElementById('config'), result=document.getElementById('result'), output=document.getElementById('output');
 const release=window.INSTALL_RELEASE;
 document.getElementById('release').textContent=`${release.channel} · ${release.image} · ${release.revision}`;
 let filename='compose.yaml', generated='', environment='', secret='';
 const shell=value=>"'"+String(value).replaceAll("'","'\"'\"'")+"'";
 const powershell=value=>"'"+String(value).replaceAll("'","''")+"'";
 const yaml=value=>"'"+String(value).replaceAll("'","''").replaceAll('$',()=> '$$')+"'";
 function newSecret(){const bytes=new Uint8Array(32);crypto.getRandomValues(bytes);return Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');}
 form.elements.storage.addEventListener('change',()=>{form.elements.path.disabled=form.elements.storage.value!=='bind';form.elements.path.required=!form.elements.path.disabled;});
 form.elements.provider.addEventListener('change',()=>{form.elements.key.disabled=!form.elements.provider.value;});
 function clear(){form.elements.key.value='';secret='';generated='';environment='';output.textContent='';result.hidden=true;}
 document.getElementById('clear').addEventListener('click',clear);
 form.addEventListener('submit',event=>{
  event.preventDefault();
  const v=Object.fromEntries(new FormData(form)), errors=[];
  let url;try{url=new URL(v.url);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)errors.push('Use an HTTP or HTTPS application URL without credentials, query or fragment.');}catch(_){errors.push('Enter a valid application URL.');}
  if(!Number.isInteger(Number(v.port))||Number(v.port)<1||Number(v.port)>65535)errors.push('Port must be between 1 and 65535.');
  const windowsPath=/^[A-Za-z]:[\\/]/.test(v.path||'');
  if(v.storage==='bind'&&(!v.path||/[\r\n,\0]/.test(v.path)||(!v.path.startsWith('/')&&!(v.shell==='powershell'&&windowsPath))))errors.push('Use an absolute host directory without commas or newlines; Windows paths require PowerShell.');
  if(v.key&&/[\r\n\0]/.test(v.key))errors.push('API keys cannot contain newlines.');
  result.hidden=false;
  if(errors.length){generated='';environment='';output.textContent='';document.getElementById('status').textContent=errors.join(' ');return;}
  secret ||= newSecret();
  const env={SESSION_SECRET_KEY:secret,BASE_URL:url.href.replace(/\/$/,'')};
  if(v.https){env.COOKIE_SECURE='true';env.TRUST_PROXY_HEADERS='true';}
  if(v.provider&&v.key)env[v.provider]=v.key;
  environment=Object.entries(env).map(([k,x])=>k+'='+x).join('\n')+'\n';
  const volume=v.storage==='bind'?v.path:'podcast-data';
  if(v.format==='compose'){
   filename='compose.yaml';
   generated=`services:\n  podcast-ad-remover:\n    image: ${yaml(release.image)}\n    restart: unless-stopped\n    ports:\n      - ${yaml(v.port+':8000')}\n    volumes:\n      - type: ${v.storage==='bind'?'bind':'volume'}\n        source: ${yaml(volume)}\n        target: /data\n    env_file:\n      - path: ./install.env\n        format: raw\n`;
   if(v.gpu)generated+='    deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia\n              count: all\n              capabilities: [gpu]\n';
   if(v.storage!=='bind')generated+='volumes:\n  podcast-data:\n';
   document.getElementById('command').textContent='Download compose.yaml and install.env into the same folder. Requires Docker Compose 2.30+. Run: docker compose up -d';
  }else{
   const ps=v.shell==='powershell', quote=ps?powershell:shell, continuation=ps?' `\n':' \\\n';
   filename=ps?'install.ps1':'install.sh';
   generated=(ps?"$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath $PSScriptRoot\n":"#!/bin/sh\nset -eu\ncd -- \"$(dirname -- \"$0\")\"\n")+['docker run -d --name podcast-ad-remover --restart unless-stopped',`  -p ${quote(v.port+':8000')}`,`  --mount ${quote(`type=${v.storage==='bind'?'bind':'volume'},source=${volume},target=/data`)}`,'  --env-file ./install.env',...(v.gpu?['  --gpus all']:[]),`  ${quote(release.image)}`].join(continuation)+'\n';
   if(ps)generated+='if ($LASTEXITCODE -ne 0) { throw "Docker failed with exit code $LASTEXITCODE" }\n';
   document.getElementById('command').textContent=`Download ${filename} and install.env into the same folder, then run: ${ps?'.\\install.ps1':'sh install.sh'}`;
  }
  output.textContent=generated;document.getElementById('status').textContent='Generated locally. Review before running. Credentials are only in the separate env file.';
 });
 document.getElementById('regenerate').addEventListener('click',()=>{secret=newSecret();form.requestSubmit();});
 document.getElementById('copy').addEventListener('click',async()=>{try{if(generated)await navigator.clipboard.writeText(generated);document.getElementById('status').textContent='Copied install file; download the env file separately.';}catch(_){document.getElementById('status').textContent='Clipboard unavailable. Select and copy the output, or download it.';}});
 function download(contents,name){if(!contents)return;const url=URL.createObjectURL(new Blob([contents],{type:'text/plain'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 document.getElementById('download').addEventListener('click',()=>download(generated,filename));
 document.getElementById('download-env').addEventListener('click',()=>download(environment,'install.env'));
 window.addEventListener('pagehide',clear);
})();
