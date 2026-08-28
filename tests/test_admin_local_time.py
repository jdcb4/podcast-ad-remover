from pathlib import Path

import pytest

# (template path relative to app/web/templates, expected count of `| local_time` uses)
ADMIN_LOCAL_TIME_SITES = [
    ("admin/users.html", 5),
    ("admin/feed_access.html", 2),
    ("admin/system.html", 1),
    ("admin/access_requests.html", 2),
]


@pytest.mark.parametrize("template_path, expected_count", ADMIN_LOCAL_TIME_SITES)
def test_admin_template_uses_local_time_and_not_compact_datetime(template_path, expected_count):
    source = Path("app/web/templates", template_path).read_text(encoding="utf-8")

    assert "compact_datetime" not in source
    assert source.count("| local_time") == expected_count
