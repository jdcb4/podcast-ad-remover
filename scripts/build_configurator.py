"""Bundle the static configurator for local-file use without user configuration."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_agent_skill import build as build_agent_skill

def build(directory):
    directory = Path(directory)
    build_agent_skill(directory / 'podcast-ad-remover-skill.zip')
    with ZipFile(directory / 'offline.zip', 'w', ZIP_DEFLATED) as archive:
        for name in ('index.html', 'style.css', 'configurator.js', 'release.js', 'podcast-ad-remover-skill.zip'):
            archive.write(directory / name, name)

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'configurator')
