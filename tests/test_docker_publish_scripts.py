import json
from pathlib import Path

import pytest

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
