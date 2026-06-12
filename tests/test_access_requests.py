from pathlib import Path

import pytest

from app.infra.database import get_db_connection, init_db
from app.web.auth_utils import hash_password, verify_password
from app.web.router import approve_access_request


def test_access_request_submission_hashes_requested_password():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")
    request_template = Path("app/web/templates/request_access.html").read_text(encoding="utf-8")
    admin_template = Path("app/web/templates/admin/access_requests.html").read_text(encoding="utf-8")

    assert "hash_password(password)" in router_source
    assert "INSERT INTO access_requests (username, email, reason, password_hash, ip_address)" in router_source
    assert 'name="password"' in request_template
    assert 'name="confirm_password"' in request_template
    assert "Temporary Password" not in admin_template
    assert "password they chose" in admin_template


@pytest.mark.asyncio
async def test_approving_access_request_uses_stored_password_hash(isolated_data_dir):
    init_db()
    chosen_hash = hash_password("chosen-password")

    with get_db_connection() as conn:
        request_id = conn.execute(
            """
            INSERT INTO access_requests (username, email, reason, password_hash, ip_address)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("newuser", "newuser@example.com", "testing", chosen_hash, "127.0.0.1"),
        ).lastrowid
        conn.commit()

    response = await approve_access_request(request=None, request_id=request_id, admin_user=object())

    with get_db_connection() as conn:
        user = conn.execute("SELECT username, password_hash FROM users WHERE username = ?", ("newuser",)).fetchone()
        access_request = conn.execute("SELECT status FROM access_requests WHERE id = ?", (request_id,)).fetchone()

    assert response.status_code == 303
    assert "approved=newuser" in response.headers["location"]
    assert user["username"] == "newuser"
    assert user["password_hash"] == chosen_hash
    assert verify_password("chosen-password", user["password_hash"]) is True
    assert access_request["status"] == "approved"


@pytest.mark.asyncio
async def test_legacy_access_request_without_password_hash_is_not_approved(isolated_data_dir):
    init_db()

    with get_db_connection() as conn:
        request_id = conn.execute(
            """
            INSERT INTO access_requests (username, email, reason, ip_address)
            VALUES (?, ?, ?, ?)
            """,
            ("legacyuser", "legacy@example.com", "old request", "127.0.0.1"),
        ).lastrowid
        conn.commit()

    response = await approve_access_request(request=None, request_id=request_id, admin_user=object())

    with get_db_connection() as conn:
        user = conn.execute("SELECT id FROM users WHERE username = ?", ("legacyuser",)).fetchone()
        access_request = conn.execute("SELECT status FROM access_requests WHERE id = ?", (request_id,)).fetchone()

    assert response.status_code == 303
    assert "password+capture" in response.headers["location"]
    assert user is None
    assert access_request["status"] == "pending"
