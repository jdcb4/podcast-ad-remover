from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from app.core.reports import render_ad_report
from app.web.security_headers import SecurityHeadersMiddleware


def test_every_report_text_field_is_escaped():
    attack = '<img src=x onerror="alert(1)">'
    ep = SimpleNamespace(id=1, title=attack, guid=attack)
    segment = dict(start=attack, end=attack, label=attack, sources=[attack], reason=attack, text=attack)
    output = render_ad_report(ep, [segment])
    assert attack not in output
    assert output.count('&lt;img') == 9


def test_legacy_report_cannot_execute_scripts_or_submit_forms():
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)
    app.get('/artifacts/report/1')(lambda: HTMLResponse('<script>alert(1)</script>'))
    response = TestClient(app).get('/artifacts/report/1')
    policy = response.headers['content-security-policy']
    assert "sandbox;" in policy
    assert "default-src 'none'" in policy
    assert 'allow-scripts' not in policy
    assert "form-action 'none'" in policy
