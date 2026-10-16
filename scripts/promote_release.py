"""Validate and copy a qualified Dev image to release tags without rebuilding."""
import json
import re
import subprocess
from pathlib import Path

DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def descriptor(docker, reference, *, missing_ok=False):
    result = capture([docker, 'buildx', 'imagetools', 'inspect', reference,
                      '--format', '{{json .Manifest}}'])
    if result.returncode:
        # Fail closed on access/transport errors; only an explicit missing manifest
        # permits creating a new immutable version tag.
        if missing_ok and result.stderr.strip().endswith(f'{reference}: not found'):
            return None
        raise SystemExit(f'Cannot inspect registry image {reference}; check Docker access.')
    try:
        value = json.loads(result.stdout)['digest']
    except (ValueError, KeyError, TypeError) as exc:
        raise SystemExit('Registry did not return an image digest.') from exc
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise SystemExit('Registry returned an invalid image digest.')
    return value


def validate_ci(gh, revision):
    repo = capture([gh, 'repo', 'view', '--json', 'nameWithOwner'])
    if repo.returncode:
        raise SystemExit('Cannot resolve the GitHub repository for release verification.')
    try:
        name = json.loads(repo.stdout)['nameWithOwner']
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', name):
            raise ValueError('Invalid repository')
        checks = capture([gh, 'api', f'repos/{name}/commits/{revision}/check-runs?per_page=100'])
        if checks.returncode:
            raise ValueError('Cannot read checks')
        runs = [r for r in json.loads(checks.stdout)['check_runs']
                if isinstance(r, dict) and r.get('name') == 'verify'
                and isinstance(r.get('app'), dict) and r['app'].get('slug') == 'github-actions']
        if not runs or any(r.get('head_sha') != revision or r.get('status') != 'completed'
                           or r.get('conclusion') != 'success' for r in runs):
            raise ValueError('Verification incomplete or failed')
    except (ValueError, TypeError, KeyError) as exc:
        raise SystemExit('The exact release commit needs successful GitHub verify checks.') from exc


def promote(source, repository, version, revision, *, docker, git, gh, push=False):
    if not re.fullmatch(r'[a-z0-9][a-z0-9./:_-]*', repository) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise SystemExit('Invalid release repository or commit SHA.')
    prefix = repository + '@'
    if not source.startswith(prefix) or not DIGEST.fullmatch(source[len(prefix):]):
        raise SystemExit('Select an immutable source image from the release repository: repository@sha256:...')
    digest = source[len(prefix):]
    # The approved commit may precede today's Dev tip, but must belong to Dev and
    # be exactly the commit already promoted to the remote production branch.
    subprocess.run([git, 'fetch', 'origin', 'main', 'dev'], cwd=ROOT, check=True)
    remote_main = capture([git, 'rev-parse', 'origin/main'])
    ancestor = capture([git, 'merge-base', '--is-ancestor', revision, 'origin/dev'])
    if remote_main.returncode or remote_main.stdout.strip() != revision or ancestor.returncode:
        raise SystemExit('main must match the exact promoted Dev commit on origin.')
    validate_ci(gh, revision)
    if descriptor(docker, source) != digest:
        raise SystemExit('Source image digest mismatch.')
    subprocess.run([docker, 'pull', '--platform', 'linux/amd64', source], cwd=ROOT, check=True)
    inspected = capture([docker, 'image', 'inspect', source])
    try:
        if inspected.returncode:
            raise ValueError('Inspection failed')
        image = json.loads(inspected.stdout)[0]
        labels = image['Config']['Labels'] or {}
        if not isinstance(labels, dict):
            raise ValueError('Invalid image labels')
        if (source not in image['RepoDigests'] or image['Os'] != 'linux'
                or image['Architecture'] != 'amd64'
                or labels.get('org.opencontainers.image.revision') != revision
                or labels.get('org.opencontainers.image.version') != version
                or labels.get('io.par.channel') != 'dev'):
            raise ValueError('Image provenance mismatch')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise SystemExit('Qualified Dev image must match the exact commit, package version and Linux AMD64 platform.') from exc
    version_tag, latest_tag = f'{repository}:{version}', f'{repository}:latest'
    existing = descriptor(docker, version_tag, missing_ok=True)
    if existing is not None and existing != digest:
        raise SystemExit('This version already names another image. Bump the version before building and qualifying a new Dev image.')
    print(f'Qualified source: {source}\nRevision: {revision}\nRelease tags: {version_tag}, {latest_tag}')
    if push:
        subprocess.run([docker, 'buildx', 'imagetools', 'create', '--prefer-index=false',
                        '--tag', version_tag, '--tag', latest_tag, source], cwd=ROOT, check=True)
        for tag in (version_tag, latest_tag):
            if descriptor(docker, tag) != digest:
                raise SystemExit(f'Publication digest mismatch for {tag}; installer dispatch blocked. Inspect registry tags before retrying.')
    return version_tag
