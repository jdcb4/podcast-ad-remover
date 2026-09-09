import json
import os
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.infra.database import get_db_connection, init_db
from app.main import app
from app.web.auth_utils import hash_password
from app.web.template_filters import local_time


@pytest.mark.parametrize("zone,published", [
    ("Australia/Sydney", "2026-09-01 18:30:00"),
    ("America/Los_Angeles", "2026-09-01 02:30:00"),
])
def test_dashboard_view_switches_keep_browser_local_dates(isolated_data_dir, zone, published):
    init_db()
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users(id,username,password_hash,is_admin) VALUES(1,'viewer',?,1)", (hash_password("test-password"),))
        conn.execute("UPDATE app_settings SET auth_enabled=1 WHERE id=1")
        conn.execute("INSERT INTO subscriptions(id,feed_url,title,slug) VALUES(1,'https://example.com/feed','Show','show')")
        conn.execute("INSERT INTO user_subscriptions(user_id,subscription_id) VALUES(1,1)")
        conn.execute(
            """INSERT INTO episodes(id,subscription_id,guid,title,pub_date,original_url,status,local_filename)
               VALUES(7,1,'episode','Episode',?,'https://example.com/audio','completed','existing.mp3')""",
            (published,),
        )
        conn.commit()
    client = TestClient(app)
    assert client.post("/login", data={"username": "viewer", "password": "test-password"}, follow_redirects=False).status_code == 302
    views = {}
    for view in ("mine", "library"):
        response = client.get(f"/?view={view}")
        assert response.status_code == 200
        views[view] = response.text
    payload = {
        "views": views,
        "zone": zone,
        "legacy": str(local_time("2026-09-01 12:00:00", "datetime", False)),
    }
    result = subprocess.run(
        ["node", "tests/local_time_dom.cjs"], input=json.dumps(payload), text=True,
        capture_output=True, env=dict(os.environ, TZ=zone), timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
