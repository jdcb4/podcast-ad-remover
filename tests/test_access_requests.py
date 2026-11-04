from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from app.infra.database import get_db_connection, init_db
from app.web import router as web
from app.web.auth_utils import hash_password, verify_password
from app.web.error_handlers import configure_error_handlers
from app.web.router import approve_access_request


@pytest.fixture
def client(isolated_data_dir):
    init_db()
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-only")
    configure_error_handlers(app)
    app.include_router(web.router)
    return TestClient(app, raise_server_exceptions=False)


def set_enabled(value):
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET access_requests_enabled=? WHERE id=1", (value,))
        conn.commit()


def signup(client):
    return client.post("/submit-access-request", data={
        "username": "newuser", "password": "chosen-password", "confirm_password": "chosen-password",
    })


def request_count():
    with get_db_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM access_requests").fetchone()[0]


def enable_login():
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET auth_enabled=1 WHERE id=1")
        conn.execute("INSERT INTO users (username,password_hash,is_admin) VALUES ('admin',?,1)", (hash_password("admin-password"),))
        conn.commit()


def test_access_requests_are_on_by_default(client):
    enable_login()
    assert client.get("/request-access").status_code == 200
    assert 'href="/request-access"' in client.get("/login").text
    assert signup(client).status_code == 200
    assert request_count() == 1


def test_disabling_access_requests_removes_the_signup_flow(client):
    enable_login()
    set_enabled(0)
    assert client.get("/request-access").status_code == 404
    assert signup(client).status_code == 404
    assert client.post("/submit-access-request").status_code == 404
    assert request_count() == 0
    assert "/request-access" not in client.get("/login").text
    failed_login = client.post("/login", data={"username": "admin", "password": "wrong"})
    assert "/request-access" not in failed_login.text


def test_error_page_links_to_signup_only_when_enabled(client):
    enable_login()
    assert 'href="/request-access"' in client.get("/change-password").text
    set_enabled(0)
    response = client.get("/change-password")
    assert response.status_code == 401
    assert "/request-access" not in response.text


def test_disabling_keeps_pending_requests_for_the_admin(client):
    assert signup(client).status_code == 200
    set_enabled(0)
    assert request_count() == 1
    assert "newuser" in client.get("/admin/users").text


def stored_setting():
    with get_db_connection() as conn:
        return conn.execute("SELECT access_requests_enabled FROM app_settings WHERE id=1").fetchone()[0]


def test_access_form_saves_the_setting_and_other_forms_leave_it_alone(client):
    assert 'name="access_requests_enabled"' in client.get("/admin/users").text
    client.post("/admin/system/update", data={"section": "access"})
    assert stored_setting() == 0
    client.post("/admin/system/update", data={"section": "system"})
    assert stored_setting() == 0
    client.post("/admin/system/update", data={"section": "api"})
    assert stored_setting() == 0
    client.post("/admin/system/update", data={"section": "access", "access_requests_enabled": "on"})
    assert stored_setting() == 1
    client.post("/admin/system/update", data={"section": "system"})
    assert stored_setting() == 1


def test_access_request_submission_hashes_requested_password():
    router_source = Path("app/web/router.py").read_text(encoding="utf-8")
    request_template = Path("app/web/templates/request_access.html").read_text(encoding="utf-8")

    assert "hash_password(password)" in router_source
    assert "INSERT INTO access_requests (username, email, reason, password_hash, ip_address)" in router_source
    assert 'name="password"' in request_template
    assert 'name="confirm_password"' in request_template


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
