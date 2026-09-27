import json
import subprocess
from pathlib import Path

import yaml


def generate(format='compose', key="key'with$dollar", gpu=False):
    script = '''
const {JSDOM}=require('jsdom'),fs=require('fs');
const dom=new JSDOM(fs.readFileSync('configurator/index.html','utf8'),{runScripts:'outside-only',url:'https://example.test/dev/'});
const w=dom.window;w.eval(fs.readFileSync('configurator/release.js','utf8'));w.eval(fs.readFileSync('configurator/configurator.js','utf8'));
const f=w.document.getElementById('config');
f.elements.format.value=__FORMAT__;f.elements.provider.value='OPENAI_API_KEY';f.elements.provider.dispatchEvent(new w.Event('change'));f.elements.key.value=__KEY__;f.elements.gpu.checked=__GPU__;
f.dispatchEvent(new w.Event('submit',{cancelable:true}));
console.log(JSON.stringify({output:w.document.getElementById('output').textContent,stored:w.localStorage.length}));
w.document.getElementById('clear').click();if(f.elements.key.value||w.document.getElementById('output').textContent)throw Error('Clear failed');
dom.window.close();
'''.replace('__FORMAT__', json.dumps(format)).replace('__KEY__', json.dumps(key)).replace('__GPU__', json.dumps(gpu))
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_compose_escapes_credentials_and_preserves_volume():
    result=generate(gpu=True)
    data=yaml.safe_load(result['output'])
    service=data['services']['podcast-ad-remover']
    assert service['environment']['OPENAI_API_KEY'] == "key'with$$dollar"
    assert len(service['environment']['SESSION_SECRET_KEY']) == 64
    assert service['volumes'] == ['podcast-data:/data']
    assert service['deploy']['resources']['reservations']['devices'][0]['capabilities'] == ['gpu']
    assert result['stored'] == 0
    assert generate()['output'] != generate()['output']


def test_shell_output_quotes_untrusted_input():
    import shlex
    result=generate('docker', "a'$(touch /tmp/nope)`echo bad`$")['output']
    tokens=shlex.split(result.replace('\\\n',''))
    assert "OPENAI_API_KEY=a'$(touch /tmp/nope)`echo bad`$" in tokens
    assert '--mount' in tokens


def test_configurator_has_no_network_or_persistence_code():
    script=Path('configurator/configurator.js').read_text(encoding='utf-8-sig')
    for forbidden in ('fetch(', 'XMLHttpRequest', 'localStorage', 'sessionStorage', 'sendBeacon', 'innerHTML'):
        assert forbidden not in script
    assert "connect-src 'none'" in Path('configurator/index.html').read_text(encoding='utf-8-sig')
