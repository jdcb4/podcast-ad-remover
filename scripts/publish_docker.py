#!/usr/bin/env python3
"""Build locally or promote an exact qualified Dev image to production tags."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


try:
    from .docker_build import build_image
except ImportError:
    from docker_build import build_image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "jdcb4/podcast-ad-remover"
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
PRODUCTION_BRANCH = "main"


def executable(name: str) -> str:
    if os.name == "nt":
        resolved = shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
        if resolved:
            return resolved
    return shutil.which(name) or name


def package_version() -> str:
    package_json = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    version = package_json["version"]
    if not SEMVER_RE.fullmatch(version):
        raise SystemExit(f"package.json version must be SemVer MAJOR.MINOR.PATCH, got {version!r}")
    lock = json.loads((ROOT / 'package-lock.json').read_text(encoding='utf-8'))
    if lock.get('version') != version or lock.get('packages', {}).get('', {}).get('version') != version:
        raise SystemExit('package.json and package-lock.json versions must match.')
    return version


def git_output(*args: str) -> str:
    result = subprocess.run(
        [executable("git"), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_release_checkout(branch: str, is_clean: bool) -> None:
    if branch != PRODUCTION_BRANCH:
        raise SystemExit(
            f"Release images must be built from branch {PRODUCTION_BRANCH!r}; current branch is {branch!r}"
        )
    if not is_clean:
        raise SystemExit("Release images must be built from a clean committed checkout")


def run(command: list[str], label: str) -> None:
    print(f"\n==> {label}")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build locally or promote a qualified immutable Dev image.")
    parser.add_argument("--push", action="store_true", help="Push tags to Docker Hub.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip the local verification checks.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="Docker image repository.")
    parser.add_argument("--platform", default="linux/amd64", help="Docker build platform.")
    parser.add_argument('--source-image', help='Qualified immutable Dev image to promote without rebuilding.')
    args = parser.parse_args()

    branch = git_output("branch", "--show-current")
    is_clean = not git_output("status", "--porcelain")
    validate_release_checkout(branch, is_clean)

    version = package_version()
    repository = args.repository
    version_tag = f"{repository}:{version}"
    latest_tag = f"{repository}:latest"

    if args.push and not args.source_image:
        raise SystemExit('Release publication requires --source-image repository@sha256:...; rebuilds cannot publish releases.')
    if args.source_image and args.platform != 'linux/amd64':
        raise SystemExit('Release promotion currently requires linux/amd64.')

    if not args.skip_verify:
        run([sys.executable, "scripts/verify.py"], "Pre-build verification")

    if args.source_image:
        try:
            from .promote_release import promote
        except ImportError:
            from promote_release import promote
        promote(args.source_image, repository, version, git_output('rev-parse', 'HEAD'),
                docker=executable('docker'), git=executable('git'), gh=executable('gh'), push=args.push)
    else:
        build_image(args.platform, [version_tag, latest_tag], [], False)
    if args.push and args.repository == DEFAULT_REPOSITORY:
        from publish_configurator import dispatch
        dispatch('stable', git_output('rev-parse', 'HEAD'), version_tag)
    print(f"\n{'Promoted' if args.push else 'Validated promotion for' if args.source_image else 'Built locally'}: {version_tag}, {latest_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
