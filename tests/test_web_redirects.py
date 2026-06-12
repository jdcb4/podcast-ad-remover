from app.web.router import _safe_local_redirect


def test_safe_local_redirect_preserves_local_paths():
    assert _safe_local_redirect("/admin/access", "/admin/system") == "/admin/access"
    assert _safe_local_redirect("/admin/access?tab=users", "/admin/system") == "/admin/access?tab=users"


def test_safe_local_redirect_rejects_external_targets():
    assert _safe_local_redirect("https://evil.example", "/admin/system") == "/admin/system"
    assert _safe_local_redirect("//evil.example", "/admin/system") == "/admin/system"
    assert _safe_local_redirect("\\\\evil.example\\share", "/admin/system") == "/admin/system"
    assert _safe_local_redirect(None, "/admin/system") == "/admin/system"
