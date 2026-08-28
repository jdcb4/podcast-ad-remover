import shutil
import subprocess
from pathlib import Path

import pytest


def test_local_time_script_exists_and_defines_the_hydrator():
    source = Path("app/web/static/js/local-time.js").read_text(encoding="utf-8")

    assert "window.AppLocalTime" in source
    assert "time[data-lt]" in source
    assert "toLocaleDateString" in source
    assert "formatIso" in source


def test_base_template_loads_local_time_script():
    template_source = Path("app/web/templates/base.html").read_text(encoding="utf-8")

    assert 'src="/static/js/local-time.js?v={{ static_asset_version(\'js/local-time.js\') }}"' in template_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_local_time_js_has_valid_syntax():
    subprocess.run(["node", "--check", "app/web/static/js/local-time.js"], check=True)
