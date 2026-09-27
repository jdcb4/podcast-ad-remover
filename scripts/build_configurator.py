"""Bundle the static configurator for local-file use without user configuration."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys

def build(directory):
    directory = Path(directory)
    with ZipFile(directory / 'offline.zip', 'w', ZIP_DEFLATED) as archive:
        for name in ('index.html', 'style.css', 'configurator.js', 'release.js'):
            archive.write(directory / name, name)

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'configurator')
