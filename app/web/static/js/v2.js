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
 document.querySelectorAll('[data-feed-apps]').forEach(button=>button.addEventListener('click',async()=>{
  try { const response=await fetch('/account/feed'); if(!response.ok) throw new Error('Could not load feed URL'); const links=await response.json();
   window.location.assign(links.apps);
  } catch(error) { window.appToast?.(error.message,{type:'error'}); }
 }));
 const sidebar=document.querySelector('.app-sidebar'), menuButton=document.querySelector('[data-toggle-sidebar]');
 function setSidebar(open){
  sidebar.toggleAttribute('data-open',open);document.querySelector('.sidebar-backdrop').hidden=!open;menuButton?.setAttribute('aria-expanded',String(open));
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
 document.querySelectorAll('.app-sidebar .primary-navigation>a,.mobile-bottom>a').forEach(link=>{
  const url=new URL(link.href),current=new URL(location.href);
  const active=url.pathname==='/'?current.pathname==='/'&&!url.hash&&(url.searchParams.get('view')||'mine')===(current.searchParams.get('view')||'mine'):url.pathname==='/admin/queue'?current.pathname==='/admin/queue':url.pathname==='/admin/ai/transcription'&&current.pathname.startsWith('/admin')&&current.pathname!=='/admin/queue';
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
  const status=document.getElementById('speech-status'),button=document.getElementById('preview-speech');button.disabled=true;status.textContent='Generating preview…';
  try { const response=await fetch('/admin/ai/voice/preview',{method:'POST',body:new FormData(document.getElementById('aiForm'))}); if(!response.ok) throw new Error((await response.json()).detail);
   const audio=document.getElementById('speech-preview'); if(audio.src) URL.revokeObjectURL(audio.src); audio.src=URL.createObjectURL(await response.blob()); audio.hidden=false; status.textContent='Preview ready';
  } catch(error) {status.textContent=error.message;}finally{previewEnabled();}
 });
 const speechProvider=document.getElementById('speech-provider');
 const speechOptions={};
 for(const id of ['speech-models','speech-voices']) speechOptions[id]=Array.from(document.querySelectorAll(`#${id} option`)).map(o=>({provider:o.dataset.provider,value:o.value}));
 let catalogRequest=0;
 function options(id,values){
  const list=document.getElementById(id), field=document.getElementById(id==='speech-models'?'speech-model':'speech-voice'), picker=document.getElementById(field.id+'-picker');
  list.replaceChildren();picker.replaceChildren();
  const empty=document.createElement('option');empty.value='';empty.textContent='Choose '+(id==='speech-models'?'model':'voice');picker.append(empty);
  values.forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=value;list.append(option.cloneNode(true));picker.append(option);});
  const manual=document.createElement('option');manual.value='__manual__';manual.textContent='Enter another ID…';picker.append(manual);
  picker.value=values.includes(field.value)?field.value:field.value?'__manual__':'';field.hidden=picker.value!=='__manual__';
 }
 function voiceFields(){
  if(!speechProvider)return;
  const provider=speechProvider.value,model=document.getElementById('speech-model').value;
  let source=provider==='openrouter'?(model.startsWith('openai/')?'openai':model.startsWith('google/gemini-')?'gemini':''):provider;
  let voices=speechOptions['speech-voices'].filter(o=>o.provider===source).map(o=>o.value);
  if(source==='openai'&&model.replace('openai/','').startsWith('tts-1'))voices=voices.filter(v=>!['ballad','verse','marin','cedar'].includes(v));
  options('speech-voices',voices);
 }
 function speechFields(event){
  catalogRequest++;
  document.getElementById('speech-custom').hidden=speechProvider.value!=='custom';
  const models=speechOptions['speech-models'].filter(o=>o.provider===speechProvider.value).map(o=>o.value);
  if(event){document.getElementById('speech-model').value=models[0]||'';document.getElementById('speech-voice').value='';}
  options('speech-models',models);
  voiceFields();
  document.getElementById('speech-catalog-status').textContent='Refresh to discover available models and voices. You can also enter an ID.';
 }
 if(speechProvider){speechProvider.addEventListener('change',speechFields);speechFields();document.getElementById('speech-model').addEventListener('input',()=>{catalogRequest++;voiceFields();});}
 for(const id of ['speech-model','speech-voice'])document.getElementById(id+'-picker')?.addEventListener('change',event=>{
  const field=document.getElementById(id);field.hidden=event.target.value!=='__manual__';
  if(event.target.value==='__manual__'){field.focus();field.select();}else{field.value=event.target.value;field.dispatchEvent(new Event('input',{bubbles:true}));}
 });
 document.querySelectorAll('[data-refresh-speech]').forEach(refresh=>refresh.addEventListener('click',async event=>{
  const button=event.currentTarget,status=document.getElementById('speech-catalog-status'),request=++catalogRequest;
  const data=new FormData();data.set('provider',speechProvider.value);data.set('model',document.getElementById('speech-model').value);
  data.set('api_key',document.querySelector('[name=speech_credential]').value);data.set('base_url',document.querySelector('[name=tts_base_url]').value);
  button.disabled=true;status.textContent='Refreshing catalog…';
  try{const response=await fetch('/admin/ai/voice/catalog',{method:'POST',body:data});const catalog=await response.json();
   if(request!==catalogRequest)return;
   if(!response.ok)throw Error(catalog.detail||'Could not refresh catalog.');
   if(button.dataset.refreshSpeech==='models')options('speech-models',catalog.models);else options('speech-voices',catalog.voices);
   if(speechProvider.value==='gemini')speechOptions['speech-voices']=speechOptions['speech-voices'].filter(o=>o.provider!=='gemini').concat(catalog.voices.map(value=>({provider:'gemini',value})));
   status.textContent=`${catalog.models.length} models · ${catalog.voices.length} voices. ${catalog.note}`;
  }catch(error){if(request===catalogRequest)status.textContent=error.message;}finally{button.disabled=false;}
 }));
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
 const previewButton=document.getElementById('preview-speech');
 function previewEnabled(){
  if(!previewButton)return;
  const provider=speechProvider.value;
  const fieldsReady=['speech-model','speech-voice'].every(id=>document.getElementById(id).value.trim());
  const keyReady=provider==='custom'||speechProvider.selectedOptions[0]?.dataset.credential==='true'||document.querySelector('[name=speech_credential]').value.trim();
  previewButton.disabled=provider==='unconfigured'||!fieldsReady||!keyReady||(provider==='custom'&&!document.querySelector('[name=tts_base_url]').value.trim());
 }
 document.getElementById('aiForm')?.addEventListener('input',previewEnabled);
 document.getElementById('aiForm')?.addEventListener('change',previewEnabled);previewEnabled();
 document.querySelectorAll('[data-autosize]').forEach(field=>{const resize=()=>{field.style.height='0px';const style=getComputedStyle(field);field.style.height=(field.scrollHeight+parseFloat(style.borderTopWidth)+parseFloat(style.borderBottomWidth))+'px';};field.addEventListener('input',resize);window.addEventListener('resize',resize);if(document.fonts)document.fonts.ready.then(resize);resize();});
 document.getElementById('copy-prompt')?.addEventListener('click',async()=>{const status=document.getElementById('copy-prompt-status');try{await navigator.clipboard.writeText(document.getElementById('timeline-prompt-preview').textContent);status.textContent='Copied';}catch{status.textContent='Could not copy. Select and copy the text manually.';}});
 document.querySelectorAll('[data-remove-key]').forEach(button=>button.addEventListener('click',async()=>{const status=button.closest('[data-provider-panel]').querySelector('[data-provider-status]');button.disabled=true;try{const response=await fetch('/admin/ai/credentials/'+button.dataset.removeKey,{method:'DELETE'});const data=await response.json();if(!response.ok)throw Error(data.detail);status.textContent=data.message;const keyInput=button.closest('[data-provider-panel]').querySelector('input[type=password]');if(keyInput){keyInput.setAttribute('placeholder','No saved key');keyInput.closest('label').querySelector('span').textContent='API key';}window.appToast?.(data.message,{type:'success'});}catch(error){status.textContent=error.message;}finally{button.disabled=false;}}));
 const history=document.getElementById('upgrade-history-data');
 if(history){
  try{
   const raw=JSON.parse(history.textContent),data=typeof raw==='string'?JSON.parse(raw):raw,container=document.getElementById('upgrade-history-content');
   const names={ai_model_cascade:'Gemini text',openai_model:'OpenAI text',anthropic_model:'Anthropic text',openrouter_model:'OpenRouter text',custom_llm_model:'Custom text',gemini_tts_model_cascade:'Gemini speech'};
   function section(title){const node=document.createElement('section'),heading=document.createElement('h3');heading.textContent=title;node.className='settings-section';node.append(heading);container.append(node);return node;}
   function paragraph(parent,text){const p=document.createElement('p');p.textContent=text;parent.append(p);}
   const actions=section('Actions recorded at upgrade');
   paragraph(actions,data.speech_setup_required?data.speech_action:'No speech configuration action was recorded.');
   paragraph(actions,'This records the upgrade, so an action shown here may already have been completed.');
   const models=section('Model selections');
   if(!data.models?.length)paragraph(models,'No model selections changed.');
   for(const model of data.models||[]){
    const item=document.createElement('div');item.className='migration-model';
    const heading=document.createElement('h4');heading.textContent=names[model.setting]||model.setting;item.append(heading);
    paragraph(item,'Retained: '+(model.selected||'Not configured'));
    const details=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('ul');summary.textContent='Previously configured models';details.append(summary,list);
    let previous=model.previous;try{previous=JSON.parse(previous);}catch{}
    for(const value of Array.isArray(previous)?previous:[previous]){const li=document.createElement('li');li.textContent=value||'Not configured';list.append(li);}
    item.append(details);models.append(item);
   }
   for(const [key,title] of [['credential_policy','Credentials'],['cut_tones','Cut tones'],['ownership','Ownership'],['converted_jobs','Queued jobs converted']])if(data[key]!==undefined)paragraph(section(title),String(data[key]));
   if(data.retired?.length){const node=section('Retired features'),list=document.createElement('ul');for(const value of data.retired){const li=document.createElement('li');li.textContent=value;list.append(li);}node.append(list);}
  }catch{document.getElementById('upgrade-history-content').textContent='The saved upgrade history cannot be displayed.';}
 }
})();
