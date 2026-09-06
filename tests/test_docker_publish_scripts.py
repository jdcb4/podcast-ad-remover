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


def test_package_exposes_arm64_experimental_no_tts_build():
    package_json = json.loads(Path("package.json").read_text(encoding="utf-8"))

    script = package_json["scripts"]["docker:experimental:arm64"]

    assert "--platform linux/arm64" in script
    assert "--no-tts" in script
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
