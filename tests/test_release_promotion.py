import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import promote_release as promotion

REVISION = 'a' * 40
DIGEST = 'sha256:' + 'b' * 64
REPOSITORY = 'jdcb4/podcast-ad-remover'
SOURCE = REPOSITORY + '@' + DIGEST


@pytest.fixture
def registry(monkeypatch):
    state = {'calls': [], 'published': False, 'version': None, 'checks': 'success',
             'main': REVISION, 'ancestor': 0, 'labels': {
                 'org.opencontainers.image.revision': REVISION,
                 'org.opencontainers.image.version': '2.0.1', 'io.par.channel': 'dev'},
             'tag_digest': DIGEST, 'registry_error': False, 'publish_failure': False}

    def execute(command, **kwargs):
        state['calls'].append(command)
        output, error, code = '', '', 0
        if command[:2] == ['git', 'rev-parse']:
            output = state['main']
        elif command[:2] == ['git', 'merge-base']:
            code = state['ancestor']
        elif command[:3] == ['gh', 'repo', 'view']:
            output = json.dumps({'nameWithOwner': REPOSITORY})
        elif command[:2] == ['gh', 'api']:
            output = json.dumps({'check_runs': [{'name': 'verify', 'app': {'slug': 'github-actions'},
                'head_sha': REVISION, 'status': 'completed', 'conclusion': state['checks']}]})
        elif command[:3] == ['docker', 'image', 'inspect']:
            output = json.dumps([{'Config': {'Labels': state['labels']}, 'Os': 'linux',
                                 'Architecture': 'amd64', 'RepoDigests': [SOURCE]}])
        elif command[:4] == ['docker', 'buildx', 'imagetools', 'inspect']:
            reference = command[4]
            if state['registry_error']:
                code, error = 1, 'Registry access denied'
            elif reference == SOURCE:
                output = json.dumps({'digest': DIGEST})
            elif reference.endswith(':2.0.1') and not state['published']:
                if state['version']:
                    output = json.dumps({'digest': state['version']})
                else:
                    code, error = 1, f'ERROR: docker.io/{reference}: not found'
            else:
                output = json.dumps({'digest': state['tag_digest']})
        elif command[:4] == ['docker', 'buildx', 'imagetools', 'create']:
            if state['publish_failure']:
                raise subprocess.CalledProcessError(1, command)
            state['published'] = True
        return SimpleNamespace(returncode=code, stdout=output, stderr=error)

    monkeypatch.setattr(promotion.subprocess, 'run', execute)
    return state


def promote(push=True, source=SOURCE):
    return promotion.promote(source, REPOSITORY, '2.0.1', REVISION,
                             docker='docker', git='git', gh='gh', push=push)


def test_promotion_copies_exact_digest_without_building(registry):
    assert promote() == REPOSITORY + ':2.0.1'
    creates = [c for c in registry['calls'] if 'create' in c]
    assert creates == [['docker', 'buildx', 'imagetools', 'create', '--prefer-index=false',
                        '--tag', REPOSITORY + ':2.0.1', '--tag', REPOSITORY + ':latest', SOURCE]]
    assert not any('build' in c for c in registry['calls'])
    assert any(c[4] == REPOSITORY + ':latest' for c in registry['calls'] if 'inspect' in c and 'imagetools' in c)


def test_promotion_plan_does_not_publish(registry):
    promote(push=False)
    assert not registry['published']


@pytest.mark.parametrize('source', [REPOSITORY + ':dev', REPOSITORY + ':latest', 'other/repo@' + DIGEST])
def test_mutable_or_other_repository_source_is_rejected(registry, source):
    with pytest.raises(SystemExit, match='immutable'):
        promote(source=source)
    assert not registry['calls']


@pytest.mark.parametrize('field,value', [
    ('org.opencontainers.image.revision', 'c' * 40),
    ('org.opencontainers.image.version', '2.0.0'), ('io.par.channel', 'stable'),
])
def test_image_provenance_mismatch_blocks_publication(registry, field, value):
    registry['labels'][field] = value
    with pytest.raises(SystemExit, match='Qualified Dev image'):
        promote()
    assert not registry['published']


@pytest.mark.parametrize('key,value', [('checks', 'failure'), ('checks', None),
                                      ('main', 'c' * 40), ('ancestor', 1),
                                      ('registry_error', True), ('version', 'sha256:' + 'c' * 64)])
def test_unsafe_release_conditions_fail_before_tagging(registry, key, value):
    registry[key] = value
    with pytest.raises(SystemExit):
        promote()
    assert not registry['published']


def test_existing_identical_version_can_be_reconciled(registry):
    registry['version'] = DIGEST
    promote()
    assert registry['published']


def test_published_digest_mismatch_is_failure(registry):
    registry['tag_digest'] = 'sha256:' + 'c' * 64
    with pytest.raises(SystemExit, match='digest mismatch'):
        promote()


def test_tagging_failure_propagates(registry):
    registry['publish_failure'] = True
    with pytest.raises(subprocess.CalledProcessError):
        promote()


def test_release_push_requires_source_and_never_builds(monkeypatch):
    from scripts import publish_docker as release
    monkeypatch.setattr(release.sys, 'argv', ['publish_docker.py', '--push', '--skip-verify'])
    monkeypatch.setattr(release, 'git_output', lambda *a: 'main' if a[0] == 'branch' else '')
    monkeypatch.setattr(release, 'build_image', lambda *a, **k: pytest.fail('Release push rebuilt an image'))
    with pytest.raises(SystemExit, match='requires --source-image'):
        release.main()


def test_failed_promotion_never_dispatches_installer(monkeypatch):
    from scripts import publish_docker as release, publish_configurator
    monkeypatch.setattr(release.sys, 'argv', ['publish_docker.py', '--push', '--skip-verify', '--source-image', SOURCE])
    monkeypatch.setattr(release, 'git_output', lambda *a: 'main' if a[0] == 'branch' else '' if a[0] == 'status' else REVISION)
    def fail(*args, **kwargs):
        raise SystemExit('Mismatch')
    monkeypatch.setattr(promotion, 'promote', fail)
    monkeypatch.setattr(publish_configurator, 'dispatch', lambda *a: pytest.fail('Premature installer publication'))
    with pytest.raises(SystemExit, match='Mismatch'):
        release.main()
