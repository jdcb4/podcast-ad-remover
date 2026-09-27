"""Package the portable agent skill with the current canonical API guide."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'skills' / 'podcast-ad-remover'
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for file in sorted(source.rglob('*')):
            if file.is_file() and file.suffix in {'.md', '.yaml'}:
                archive.write(file, str(Path(source.name) / file.relative_to(source)))
        # Ship the exact guide, not a separately maintained copy that can drift.
        for name in ('API.md', 'Agent_Skill.md', 'COMPLETE_TIMELINE.md'):
            archive.write(ROOT / 'Documentation' / name, 'podcast-ad-remover/references/' + name)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Output ZIP path')
    print(build(parser.parse_args().output))
