"""Audit npm dependencies; one explicitly approved build-only exception."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADVISORY = "https://github.com/advisories/GHSA-vfj7-8cjw-p6xm"
ALLOWED = {"braces", "chokidar", "micromatch", "fast-glob", "tailwindcss"}


def audit_failures(report, lock):
    if report.get("auditReportVersion") != 2 or "error" in report:
        raise ValueError("Invalid or failed npm audit response")
    vulnerabilities = report["vulnerabilities"]
    packages = lock["packages"]

    def excepted(name, seen=frozenset()):
        if name not in ALLOWED or name in seen or name not in vulnerabilities:
            return False
        item = vulnerabilities[name]
        nodes = item.get("nodes", [])
        if not nodes or not all(packages.get(node, {}).get("dev") is True for node in nodes):
            return False
        via = item.get("via", [])
        return bool(via) and all(
            excepted(cause, seen | {name}) if isinstance(cause, str) else
            name == "braces" and cause.get("url") == ADVISORY
            for cause in via
        )

    return [name for name, item in vulnerabilities.items()
            if item["severity"] in {"moderate", "high", "critical"} and not excepted(name)]


def main():
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    result = subprocess.run([npm, "audit", "--json"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("npm audit failed to execute")
    report = json.loads(result.stdout)
    failures = audit_failures(report, json.loads((ROOT / "package-lock.json").read_text()))
    if "braces" in report["vulnerabilities"] and "braces" not in failures:
        print("WARNING: approved build-only exception: " + ADVISORY + " (see SECURITY.md)")
    if failures:
        print("Unexcepted dependency advisories: " + ", ".join(failures))
        return 1
    print("Dependency audit passed with the explicitly documented exception, if reported above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
