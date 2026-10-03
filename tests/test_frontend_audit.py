import pytest
from scripts.audit_frontend import ADVISORY, audit_failures


def fixture():
    return ({"auditReportVersion": 2, "vulnerabilities": {
        "braces": {"severity": "high", "nodes": ["node_modules/braces"], "via": [{"url": ADVISORY}]},
        "tailwindcss": {"severity": "high", "nodes": ["node_modules/tailwindcss"], "via": ["braces"]}
    }}, {"packages": {"node_modules/braces": {"dev": True}, "node_modules/tailwindcss": {"dev": True}}})


def test_only_approved_dev_advisory_is_excepted():
    report, lock = fixture()
    assert audit_failures(report, lock) == []
    lock["packages"]["node_modules/braces"]["dev"] = False
    assert set(audit_failures(report, lock)) == {"braces", "tailwindcss"}


def test_additional_advisory_is_not_hidden_by_transitive_exception():
    report, lock = fixture()
    report["vulnerabilities"]["braces"]["via"].append({"url": "https://example.test/new"})
    assert set(audit_failures(report, lock)) == {"braces", "tailwindcss"}


def test_failed_audit_and_unknown_dependency_fail_closed():
    with pytest.raises(ValueError):
        audit_failures({"error": {}}, {})
    report, lock = fixture()
    report["vulnerabilities"]["tailwindcss"]["via"] = ["unknown"]
    assert audit_failures(report, lock) == ["tailwindcss"]

