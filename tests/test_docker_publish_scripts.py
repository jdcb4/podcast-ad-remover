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
