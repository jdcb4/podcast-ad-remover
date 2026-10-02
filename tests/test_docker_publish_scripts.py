import json
from pathlib import Path

import pytest

from scripts.publish_dev_docker import dev_tags, validate_dev_checkout
from scripts.publish_docker import validate_release_checkout
from scripts.publish_experimental_docker import validate_tags


def test_validate_experimental_tags_deduplicates_tags():
    assert validate_tags(["experimental", "audit-work", "experimental"]) == [
        "experimental",
        "audit-work",
    ]


def test_validate_experimental_tags_rejects_latest():
    with pytest.raises(SystemExit):
        validate_tags(["latest"])


def test_validate_experimental_tags_rejects_semver_release_tag():
    with pytest.raises(SystemExit):
        validate_tags(["1.3.1"])


def test_package_exposes_arm64_experimental_api_speech_build():
    package_json = json.loads(Path("package.json").read_text(encoding="utf-8"))

    script = package_json["scripts"]["docker:experimental:arm64"]

    assert "--platform linux/arm64" in script
    assert "--no-tts" not in script
    assert "--tag experimental-arm64" in script


def test_dev_tags_include_rolling_and_immutable_sha_tags():
    assert dev_tags("example/podcast-ad-remover", "abc1234") == [
        "example/podcast-ad-remover:dev",
        "example/podcast-ad-remover:dev-abc1234",
    ]


def test_validate_dev_checkout_requires_dev_branch():
    with pytest.raises(SystemExit, match="must be built from branch 'dev'"):
        validate_dev_checkout("feature/test", True)


def test_validate_dev_checkout_requires_clean_tree():
    with pytest.raises(SystemExit, match="clean committed checkout"):
        validate_dev_checkout("dev", False)


@pytest.mark.parametrize("branch", ["dev", "master", "codex/test", ""])
def test_validate_release_checkout_requires_main_branch(branch):
    with pytest.raises(SystemExit, match="must be built from branch 'main'"):
        validate_release_checkout(branch, True)


def test_validate_release_checkout_accepts_clean_main():
    validate_release_checkout("main", True)


def test_validate_release_checkout_requires_clean_tree():
    with pytest.raises(SystemExit, match="clean committed checkout"):
        validate_release_checkout("main", False)


def test_package_exposes_separate_dev_build_and_publish_commands():
    package_json = json.loads(Path("package.json").read_text(encoding="utf-8"))

    assert package_json["scripts"]["docker:dev"] == "python scripts/publish_dev_docker.py"
    assert package_json["scripts"]["docker:dev:publish"].endswith("--push")


@pytest.mark.parametrize("available,push", [(True,False),(True,True),(False,False),(False,True)])
def test_build_commands(monkeypatch, available, push):
    from types import SimpleNamespace
    from scripts import publish_experimental_docker as build
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0 if available or command[1] != "buildx" else 1,
                               stdout="linux/x86_64")
    monkeypatch.setattr(build.subprocess, "run", execute)
    build.build_image("linux/amd64", ["repo:test", "repo:test2"], ["X=1"], push)
    command = next(c for c in calls if "--platform" in c)
    assert "--build-arg" in command and "X=1" in command
    assert "repo:test2" in command
    assert ("--push" in command) == (available and push)
    assert len([c for c in calls if c[1] == "push"]) == (2 if push and not available else 0)


@pytest.mark.parametrize("target,host", [("linux/arm64","linux/x86_64"),("linux/amd64","linux/aarch64"),("linux/amd64","")])
def test_fallback_rejects_wrong_architecture(monkeypatch, target, host):
    from types import SimpleNamespace
    from scripts import publish_experimental_docker as build
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=host))
    with pytest.raises(SystemExit):
        build.build_image(target, ["repo:test"], [], False)


def test_build_failure_does_not_fallback(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    from scripts import publish_experimental_docker as build
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if "--platform" in command:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(build.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        build.build_image("linux/amd64", ["repo:test"], [], True)
    assert len(calls) == 2
