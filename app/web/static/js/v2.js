(() => {
 'use strict';
 document.addEventListener('click', event => {
  if(event.target.closest('[data-add-podcast]') && document.getElementById('add-podcast-dialog')) { event.preventDefault(); document.getElementById('add-podcast-dialog').showModal(); }
  const open = event.target.closest('[data-open-dialog]');
  if (open) { const dialog = document.getElementById(open.dataset.openDialog); if (dialog) dialog.showModal(); }
  const close = event.target.closest('[data-close-dialog]');
  if (close) close.closest('dialog')?.close();
  if (event.target.closest('[data-toggle-sidebar]')) {
   const sidebar = document.querySelector('.app-sidebar');
   sidebar.toggleAttribute('data-open');
   event.target.closest('button').setAttribute('aria-expanded', sidebar.hasAttribute('data-open'));
  }
 });
 document.querySelectorAll('[data-copy-feed],[data-feed-apps]').forEach(button=>button.addEventListener('click',async()=>{
  try { const response=await fetch('/account/feed'); if(!response.ok) throw new Error('Could not load feed URL'); const links=await response.json();
   if(button.hasAttribute('data-feed-apps')) { window.location.assign(links.apps); }
   else { await navigator.clipboard.writeText(links.rss); button.textContent='Copied'; setTimeout(()=>button.textContent='Copy feed URL',2000); }
  } catch(error) { window.appToast?.(error.message,{type:'error'}); }
 }));
 document.addEventListener('keydown', event => { if(event.key==='Escape') document.querySelector('.app-sidebar')?.removeAttribute('data-open'); });
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
 function speechFields(){document.getElementById('speech-custom').hidden=speechProvider.value!=='custom';}
 if(speechProvider){speechProvider.addEventListener('change',speechFields);speechFields();}
 function showProvider() { document.querySelectorAll('[data-provider-panel]').forEach(panel => { panel.hidden=panel.dataset.providerPanel!==provider.value; panel.querySelectorAll('input,select').forEach(input=>input.disabled=panel.hidden); }); }
 if(provider) { provider.addEventListener('change',showProvider); showProvider(); }
 const filterMobile = () => {
  const query=(document.getElementById('podcast-search')?.value||'').toLowerCase();
  const filter=document.getElementById('podcast-filter')?.value || 'all';
  document.querySelectorAll('.mobile-podcast').forEach(row => {
   row.hidden=!row.dataset.title.includes(query) || (filter==='downloaded' && !Number(row.dataset.episodes)) || (filter==='processing' && !Number(row.dataset.processing));
  });
 };
 document.addEventListener('input', event=>{ if(event.target.id==='podcast-search') filterMobile(); });
 document.addEventListener('change', event=>{ if(event.target.id==='podcast-filter') filterMobile(); });
})();
