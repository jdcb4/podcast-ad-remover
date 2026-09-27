(() => {
 'use strict';
 document.addEventListener('click', event => {
  if(event.target.closest('[data-add-podcast]') && document.getElementById('add-podcast-dialog')) { event.preventDefault(); document.getElementById('add-podcast-dialog').showModal(); }
  const open = event.target.closest('[data-open-dialog]');
  if (open) { const dialog = document.getElementById(open.dataset.openDialog); if (dialog) dialog.showModal(); }
  const close = event.target.closest('[data-close-dialog]');
  if (close) close.closest('dialog')?.close();
  if (event.target.closest('[data-toggle-sidebar]')) setSidebar(!document.querySelector('.app-sidebar').hasAttribute('data-open'));
  if (event.target.closest('[data-close-sidebar]')) setSidebar(false);
 });
 document.querySelectorAll('[data-copy-feed],[data-feed-apps]').forEach(button=>button.addEventListener('click',async()=>{
  try { const response=await fetch('/account/feed'); if(!response.ok) throw new Error('Could not load feed URL'); const links=await response.json();
   if(button.hasAttribute('data-feed-apps')) { window.location.assign(links.apps); }
   else { await navigator.clipboard.writeText(links.rss); button.textContent='Copied'; setTimeout(()=>button.textContent='Copy feed URL',2000); }
  } catch(error) { window.appToast?.(error.message,{type:'error'}); }
 }));
 const sidebar=document.querySelector('.app-sidebar'), menuButton=document.querySelector('[data-toggle-sidebar]');
 function setSidebar(open){
  sidebar.toggleAttribute('data-open',open);menuButton?.setAttribute('aria-expanded',String(open));
  for(const element of document.querySelectorAll('.app-main,.mobile-bottom,.mobile-appbar')) element.inert=open;
  if(open) sidebar.querySelector('[data-close-sidebar]')?.focus(); else menuButton?.focus();
 }
 document.addEventListener('keydown',event=>{
  if(!sidebar?.hasAttribute('data-open'))return;
  if(event.key==='Escape'){event.preventDefault();setSidebar(false);}
  if(event.key==='Tab'){
   const items=Array.from(sidebar.querySelectorAll('a,button')).filter(el=>!el.hidden&&el.getClientRects().length);
   const first=items[0],last=items.at(-1);
   if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
   else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
  }
 });
 matchMedia('(max-width:900px)').addEventListener('change',()=>{if(sidebar?.hasAttribute('data-open'))setSidebar(false);});
 const setupProvider=document.getElementById('setup-provider');
 if(setupProvider){const update=()=>{document.getElementById('setup-custom').hidden=setupProvider.value!=='custom';};setupProvider.addEventListener('change',update);update();}
 document.querySelectorAll('.app-sidebar nav>a,.mobile-bottom>a').forEach(link=>{
  const url=new URL(link.href),current=new URL(location.href);
  const active=url.pathname==='/'?current.pathname==='/'&&!url.hash&&(url.searchParams.get('view')||'mine')===(current.searchParams.get('view')||'mine'):url.pathname==='/admin/queue'?current.pathname==='/admin/queue':url.pathname==='/admin/ai/text-analysis'&&current.pathname.startsWith('/admin')&&current.pathname!=='/admin/queue';
  if(active)link.setAttribute('aria-current','page');
 });
 if (location.hash === '#add') document.getElementById('add-podcast-dialog')?.showModal();
 document.querySelectorAll('[data-json-form]').forEach(form => form.addEventListener('submit', async event => {
  event.preventDefault(); const status = form.querySelector('[role=status]');
  try { const response = await fetch(form.action,{method:'POST',body:new FormData(form)}); const data = await response.json();
   if (!response.ok) throw new Error(data.detail || 'Could not save'); status.textContent = data.detail || 'Saved';
  } catch(error) { status.textContent = error.message; }
 }));
 document.querySelectorAll('[data-reset-field]').forEach(button => button.addEventListener('click', () => {
  const field = document.getElementById(button.dataset.resetField); field.value = field.dataset.default || ''; field.dispatchEvent(new Event('input',{bubbles:true}));
 }));
 document.getElementById('preview-timeline-prompt')?.addEventListener('click', async () => {
  const output = document.getElementById('timeline-prompt-preview'); output.textContent='Loading…';
  document.getElementById('prompt-preview-dialog').showModal();
  try { const response=await fetch('/admin/prompts/timeline/preview',{method:'POST',body:new FormData(document.getElementById('timeline-prompts-form'))});
   const data=await response.json(); if(!response.ok) throw new Error(data.detail); output.textContent=data.system_prompt+'\n\n'+JSON.stringify(data.schema,null,2);
  } catch(error) { output.textContent=error.message; }
 });
 const provider = document.querySelector('[data-provider-select]');
 document.getElementById('setup-gpu')?.addEventListener('click',async()=>{
  try { const response=await fetch('/admin/ai/cuda/setup',{method:'POST'}); document.getElementById('cuda-message').textContent=response.ok?'GPU setup started; status will update below.':'GPU setup failed. Check logs.'; } catch (_) { document.getElementById('cuda-message').textContent='Could not reach the server.'; }
 });
 document.getElementById('preview-speech')?.addEventListener('click',async()=>{
  const status=document.getElementById('speech-status'); status.textContent='Generating preview…';
  try { const response=await fetch('/admin/ai/voice/preview',{method:'POST'}); if(!response.ok) throw new Error((await response.json()).detail);
   const audio=document.getElementById('speech-preview'); if(audio.src) URL.revokeObjectURL(audio.src); audio.src=URL.createObjectURL(await response.blob()); audio.hidden=false; status.textContent='Preview ready';
  } catch(error) {status.textContent=error.message;}
 });
 const speechProvider=document.getElementById('speech-provider');
 const speechOptions={};
 for(const id of ['speech-models','speech-voices']) speechOptions[id]=Array.from(document.querySelectorAll(`#${id} option`)).map(o=>({provider:o.dataset.provider,value:o.value}));
 function speechFields(event){
  document.getElementById('speech-custom').hidden=speechProvider.value!=='custom';
  for(const id of ['speech-models','speech-voices']){
   const list=document.getElementById(id);list.replaceChildren();
   const selected=speechOptions[id].filter(o=>o.provider===speechProvider.value || (id==='speech-voices' && speechProvider.value==='openrouter' && o.provider==='openai'));
   selected.forEach(o=>{const option=document.createElement('option');option.value=o.value;list.append(option);});
   if(event){document.getElementById(id==='speech-models'?'speech-model':'speech-voice').value=selected[0]?.value||'';}
  }
 }
 if(speechProvider){speechProvider.addEventListener('change',speechFields);speechFields();}
 function showProvider() { document.querySelectorAll('[data-provider-panel]').forEach(panel => { panel.hidden=panel.dataset.providerPanel!==provider.value; panel.querySelectorAll('input,select').forEach(input=>input.disabled=panel.hidden); }); }
 if(provider) { provider.addEventListener('change',showProvider); showProvider(); }
 document.querySelectorAll('[data-refresh-models],[data-test-provider]').forEach(button=>button.addEventListener('click',async()=>{
  const name=button.dataset.refreshModels||button.dataset.testProvider, panel=button.closest('[data-provider-panel]'), status=panel.querySelector('[data-provider-status]');
  const payload=new FormData();payload.set('provider',name);
  payload.set('model',panel.querySelector('input[list]')?.value||'');
  payload.set('api_key',panel.querySelector('input[type=password]')?.value||'');
  payload.set('base_url',panel.querySelector('[name=custom_llm_base_url]')?.value||'');
  button.disabled=true;status.textContent='Connecting…';
  try{const response=await fetch(button.dataset.refreshModels?'/admin/ai/refresh':'/admin/ai/test',{method:'POST',body:payload});const data=await response.json();
   if(!response.ok||data.error)throw Error(data.error||data.detail||'Connection failed');
   if(data.models){const list=document.getElementById('models-'+name);list.replaceChildren();data.models.forEach(model=>{const option=document.createElement('option');option.value=model;list.append(option);});status.textContent=`${data.models.length} models loaded. Choose one above.`;}
   else status.textContent='Connected';
  }catch(error){status.textContent=error.message;}finally{button.disabled=false;}
 }));
 const filterMobile = () => {
  const query=(document.getElementById('podcast-search')?.value||'').toLowerCase();
  const filter=document.getElementById('podcast-filter')?.value || 'all';
  const sort=document.getElementById('podcast-sort')?.value || 'alpha';
  const container=document.querySelector('.mobile-podcasts');
  const rows=Array.from(document.querySelectorAll('.mobile-podcast'));
  rows.forEach(row => {
   row.hidden=!row.dataset.title.includes(query) || (filter==='downloaded' && !Number(row.dataset.episodes)) || (filter==='processing' && !Number(row.dataset.processing)) || (filter==='recent' && !(Date.parse(row.dataset.recent)>Date.now()-7*86400000));
  });
  rows.sort((a,b)=>sort==='alpha'?a.dataset.title.localeCompare(b.dataset.title):sort==='recent'?(Date.parse(b.dataset.recent)||0)-(Date.parse(a.dataset.recent)||0):Number(b.dataset[sort==='plays'?'listens':'episodes'])-Number(a.dataset[sort==='plays'?'listens':'episodes']));
  rows.forEach(row=>container.append(row));
 };
 document.addEventListener('input', event=>{ if(event.target.id==='podcast-search') filterMobile(); });
 document.addEventListener('change', event=>{ if(['podcast-filter','podcast-sort'].includes(event.target.id)) filterMobile(); });
 document.addEventListener('dashboard-library-results-changed',filterMobile);
 document.addEventListener('library-membership-changed',event=>{
  if(document.getElementById('dashboard-podcast-results')?.dataset.libraryView==='mine' && !event.detail.in_user_library)
   document.querySelector(`.mobile-podcast[data-subscription-id="${event.detail.subscription_id}"]`)?.remove();
  filterMobile();
 });
 if(location.hash==='#processing-settings') document.getElementById('settings-form')?.classList.remove('hidden');
 filterMobile();
})();
