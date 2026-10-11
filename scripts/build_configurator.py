"""Bundle the static configurator for local-file use without user configuration."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import base64
import hashlib
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_agent_skill import build as build_agent_skill


def build_standalone(directory):
    """Inline the reviewed assets with CSP hashes for a single-file download."""
    directory = Path(directory)
    html = (directory / 'index.html').read_text(encoding='utf-8-sig')
    style = (directory / 'style.css').read_text(encoding='utf-8-sig')
    scripts = [(directory / name).read_text(encoding='utf-8-sig')
               for name in ('release.js', 'configurator.js')]
    # Inline scripts execute after the form exists; defer has no effect inline.
    for name in ('release.js', 'configurator.js'):
        tag = f'<script src="{name}" defer></script>'
        assert html.count(tag) == 1, f'Missing expected asset: {name}'
        html = html.replace(tag, '')
    assert '</script' not in ''.join(scripts).lower()
    assert '</style' not in style.lower()
    html = html.replace('<link rel="stylesheet" href="style.css">',
                        '<style>' + style + '</style>')
    html = html.replace('</body>', ''.join('<script>' + s + '</script>' for s in scripts) + '</body>')

    def csp_hash(value):
        return "'sha256-" + base64.b64encode(hashlib.sha256(value.encode('utf-8')).digest()).decode('ascii') + "'"

    html = html.replace("script-src 'self'", 'script-src ' + ' '.join(map(csp_hash, scripts)))
    html = html.replace("style-src 'self'", 'style-src ' + csp_hash(style))
    # Adjacent ZIP files aren't required or available in the single-file copy.
    html = re.sub(r'<footer>.*?</footer>', '<footer><p>This wizard is self-contained; no server or additional files are needed. '
                  'Content removal, retention, accounts, feeds, speech, prompts and notifications can all be configured after installation. '
                  'No paid speech provider is enabled automatically.</p></footer>', html, flags=re.S)
    (directory / 'install.html').write_text(html, encoding='utf-8', newline='\n')


def build(directory):
    directory = Path(directory)
    build_standalone(directory)
    build_agent_skill(directory / 'podcast-ad-remover-skill.zip')
    with ZipFile(directory / 'offline.zip', 'w', ZIP_DEFLATED) as archive:
        for name in ('index.html', 'install.html', 'style.css', 'configurator.js', 'release.js', 'podcast-ad-remover-skill.zip'):
            archive.write(directory / name, name)

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'configurator')
