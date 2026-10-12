"""Dispatch Pages only after an authorized, successful Docker publication."""
import shutil
import subprocess


def dispatch(channel: str, revision: str, image: str) -> None:
    if channel != 'stable':
        raise ValueError('Only stable installer publication is supported')
    gh = shutil.which('gh')
    if not gh:
        raise RuntimeError('Docker image published, but Pages dispatch needs GitHub CLI (gh). Rerun publish-configurator.yml for this revision/image.')
    subprocess.run([
        gh, 'workflow', 'run', 'publish-configurator.yml',
        '--repo', 'jdcb4/podcast-ad-remover', '--ref', 'main',
        '-f', f'revision={revision}', '-f', f'image={image}',
    ], check=True)
