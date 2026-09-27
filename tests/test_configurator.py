import json
import subprocess
from pathlib import Path
import yaml


def generate(format='compose', key="key'with$dollar", gpu=False, shell='posix', media=''):
    script = r'''
const {JSDOM}=require('jsdom'),fs=require('fs');
(async()=>{
const dom=new JSDOM(fs.readFileSync('configurator/index.html','utf8'),{runScripts:'outside-only',url:'file:///offline/index.html'});
const w=dom.window;w.eval(fs.readFileSync('configurator/release.js','utf8'));w.eval(fs.readFileSync('configurator/configurator.js','utf8'));
let blob;w.URL.createObjectURL=b=>{blob=b;return 'blob:test';};w.URL.revokeObjectURL=()=>{};w.HTMLAnchorElement.prototype.click=()=>{};
const f=w.document.getElementById('config');
if(!f.elements.provider.closest('details')||f.elements.provider.closest('details').open)throw Error('Credentials must start collapsed');
f.elements.format.value=__FORMAT__;f.elements.shell.value=__SHELL__;f.elements.provider.value='OPENAI_API_KEY';f.elements.provider.dispatchEvent(new w.Event('change'));f.elements.key.value=__KEY__;f.elements.gpu.checked=__GPU__;
if(__MEDIA__){f.elements.separate_media.checked=true;f.elements.separate_media.dispatchEvent(new w.Event('change'));f.elements.media_path.value=__MEDIA__;}
const submit=()=>f.dispatchEvent(new w.Event('submit',{cancelable:true}));
const env=async()=>{w.document.getElementById('download-env').click();return await new Promise(resolve=>{const reader=new w.FileReader();reader.onload=()=>resolve(reader.result);reader.readAsText(blob);});};
submit();const first=await env();f.elements.port.value='8111';submit();const second=await env();
if(first!==second)throw Error('Editing port rotated secret');
w.document.getElementById('regenerate').click();const third=await env();if(third===second)throw Error('Explicit regeneration did not rotate secret');
console.log(JSON.stringify({output:w.document.getElementById('output').textContent,env:third}));
w.document.getElementById('clear').click();if(f.elements.key.value||w.document.getElementById('output').textContent)throw Error('Clear failed');
dom.window.close();
})().catch(e=>{console.error(e);process.exit(1)});
'''.replace('__FORMAT__', json.dumps(format)).replace('__SHELL__',json.dumps(shell)).replace('__KEY__', json.dumps(key)).replace('__GPU__', json.dumps(gpu)).replace('__MEDIA__', json.dumps(media))
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_compose_separates_credentials_and_preserves_volume(tmp_path):
    result=generate(gpu=True)
    data=yaml.safe_load(result['output'])
    service=data['services']['podcast-ad-remover']
    assert service['env_file'] == [{'path':'./install.env','format':'raw'}]
    assert "OPENAI_API_KEY=key'with$dollar" in result['env']
    assert 'SESSION_SECRET_KEY' not in result['output']
    assert service['volumes'][0]['source'] == 'podcast-data'
    assert service['deploy']['resources']['reservations']['devices'][0]['capabilities'] == ['gpu']
    assert generate()['env'] != generate()['env']
    (tmp_path/'compose.yaml').write_text(result['output'])
    (tmp_path/'install.env').write_text(result['env'])
    # Parsing above is portable; Docker qualification also validates this fixture.


def test_shell_output_keeps_credentials_out_of_arguments():
    import shlex
    key="a'$(touch /tmp/nope)`echo bad`$"
    result=generate('docker',key)
    tokens=shlex.split(result['output'].replace('\\\n',''))
    assert key not in result['output'] and key in result['env']
    assert '--env-file' in tokens and '--mount' in tokens
    assert '-e' not in tokens
    ps=generate('docker',key,shell='powershell')['output']
    assert '$PSScriptRoot' in ps and '--env-file ./install.env' in ps and key not in ps


def test_configurator_has_no_network_or_persistence_code():
    script=Path('configurator/configurator.js').read_text(encoding='utf-8-sig')
    for forbidden in ('fetch(', 'XMLHttpRequest', 'localStorage', 'sessionStorage', 'sendBeacon', 'innerHTML'):
        assert forbidden not in script
    assert "connect-src 'none'" in Path('configurator/index.html').read_text(encoding='utf-8-sig')


def test_offline_archive_contains_only_static_assets(tmp_path):
    from scripts.build_configurator import build
    from zipfile import ZipFile
    import shutil
    for name in ('index.html','style.css','configurator.js','release.js'):
        shutil.copyfile(Path('configurator')/name,tmp_path/name)
    build(tmp_path)
    with ZipFile(tmp_path/'offline.zip') as archive:
        assert set(archive.namelist()) == {'index.html','style.css','configurator.js','release.js','podcast-ad-remover-skill.zip'}


def test_separate_media_mount_and_environment():
    result = generate(media='/mnt/nas/par audio')
    service = yaml.safe_load(result['output'])['services']['podcast-ad-remover']
    assert service['volumes'][1] == {'type':'bind','source':'/mnt/nas/par audio','target':'/media','bind':{'create_host_path':False}}
    assert 'MEDIA_DIR=/media' in result['env']
    result = generate('docker', shell='powershell', media='D:\\Media Store')
    assert 'target=/media' in result['output']
    assert 'MEDIA_DIR=/media' in result['env']
