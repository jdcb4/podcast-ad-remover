(() => {
 'use strict';
 const root=document.getElementById('podcast-settings-workspace');if(!root)return;
 const form=document.getElementById('podcast-settings-form'), mobile=matchMedia('(max-width: 700px)');
 const groups=[...root.querySelectorAll('.podcast-setting-group')],tabs=[...root.querySelectorAll('[role=tab]')],panels=[...root.querySelectorAll('[data-panel]')];
 let active='processing';
 const snapshot=()=>JSON.stringify([...form.elements].filter(e=>e.name).map(e=>[e.name,e.type==='checkbox'?e.checked:e.value,e.disabled]));
 const original=snapshot();
 function summaries(){
  groups.forEach(group=>{
   const toggle=group.querySelector('[data-inheritance-toggle]'),summary=group.querySelector('[data-group-summary]');
   if(!toggle)return;
   const inherited=toggle.checked;
   group.querySelector('[data-override-fields]').hidden=inherited;
   group.querySelector('.inherited-summary').hidden=!inherited;
   const values=[...group.querySelectorAll('[data-inherited-control]')].map(e=>{
    const label=e.closest('label').querySelector('span').textContent;
    return e.type==='checkbox'?(e.checked?label:null):label+': '+e.value;
   }).filter(Boolean);
   group.querySelector('[data-effective-summary]').textContent=values.join(' · ') || 'All additions off';
   summary.textContent=group.querySelector('[name=keep_whole_show]')?.checked?'Keep whole show':inherited?'Global defaults':'Custom settings';
  });
  root.querySelector('.podcast-save-bar').hidden=snapshot()===original;
 }
 function layout(){
  tabs.forEach(t=>{t.setAttribute('aria-selected',String(t.dataset.tab===active));t.tabIndex=t.dataset.tab===active?0:-1;});
  panels.forEach(p=>{p.hidden=!mobile.matches&&p.dataset.panel!==active;if(mobile.matches){p.removeAttribute('role');p.removeAttribute('aria-labelledby');}else{p.setAttribute('role','tabpanel');p.setAttribute('aria-labelledby','settings-tab-'+p.dataset.panel);}});
  if(mobile.matches){groups.forEach(g=>g.open=false);}else{groups.forEach(g=>g.open=g.dataset.group!=='3');}
 }
 tabs.forEach((tab,index)=>{
  tab.addEventListener('click',()=>{active=tab.dataset.tab;layout();});
  tab.addEventListener('keydown',e=>{let target;if(e.key==='ArrowRight')target=(index+1)%tabs.length;if(e.key==='ArrowLeft')target=(index+tabs.length-1)%tabs.length;if(e.key==='Home')target=0;if(e.key==='End')target=tabs.length-1;if(target!==undefined){e.preventDefault();tabs[target].click();tabs[target].focus();}});
 });
 groups.forEach(group=>{
  group.addEventListener('toggle',()=>{if(mobile.matches&&group.open)groups.forEach(other=>{if(other!==group)other.open=false;});});
  group.querySelector('[data-customize]')?.addEventListener('click',()=>{const toggle=group.querySelector('[data-inheritance-toggle]');toggle.checked=false;toggle.dispatchEvent(new Event('change',{bubbles:true}));});
 });
 root.addEventListener('input',summaries);root.addEventListener('change',summaries);
 root.addEventListener('invalid',event=>{const group=event.target.closest('.podcast-setting-group');if(group){active=group.dataset.settingsTab;layout();group.open=true;}},true);
 mobile.addEventListener('change',layout);
 root.classList.add('settings-enhanced');layout();summaries();
})();
