#!/usr/bin/env python3
"""Build and optionally push non-release Docker image tags."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "jdcb4/podcast-ad-remover"
TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def executable(name: str) -> str:
    if os.name == "nt":
        resolved = shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
        if resolved:
            return resolved
    return shutil.which(name) or name


def run(command: list[str], label: str) -> None:
    print(f"\n==> {label}")
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def git_short_sha() -> str:
    result = subprocess.run(
        [executable("git"), "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_tags(tags: list[str]) -> list[str]:
    cleaned = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag == "latest" or SEMVER_RE.match(tag):
            raise SystemExit(f"Refusing non-release Docker tag {tag!r}")
        if not TAG_RE.match(tag):
            raise SystemExit(f"Invalid Docker tag {tag!r}")
        if tag not in cleaned:
            cleaned.append(tag)
    if not cleaned:
        raise SystemExit("At least one non-release tag is required")
    return cleaned


def default_tags() -> list[str]:
    sha = git_short_sha()
    return ["experimental", "audit-work", f"audit-work-{sha}"]


def build_image(platform: str, tags: list[str], build_args: list[str], push: bool) -> None:
    docker = executable("docker")
    available = subprocess.run([docker, "buildx", "version"], capture_output=True).returncode == 0
    if available:
        command = [docker, "buildx", "build", "--platform", platform]
    else:
        host = subprocess.run([docker, "info", "--format", "{{.OSType}}/{{.Architecture}}"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if platform != "linux/amd64" or host not in {"linux/amd64", "linux/x86_64"}:
            raise SystemExit("Buildx is required unless target and Docker daemon are both Linux AMD64")
        command = [docker, "build", "--platform", "linux/amd64"]
    for arg in build_args:
        command.extend(["--build-arg", arg])
    for tag in tags:
        command.extend(["-t", tag])
    if available:
        command.append("--push" if push else "--load")
    command.append(".")
    run(command, "Docker experimental build")
    if push and not available:
        for tag in tags:
            run([docker, "push", tag], "Push experimental tag")
    print(("Pushed tags: " if push else "Built locally: ") + ", ".join(tags))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or publish non-release Docker tags.")
    parser.add_argument("--push", action="store_true", help="Push tags to Docker Hub.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification checks.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="Docker image repository.")
    parser.add_argument("--platform", default="linux/amd64", help="Docker build platform.")
    parser.add_argument("--tag", action="append", dest="tags", help="Non-release tag to build. Repeatable.")
    parser.add_argument("--build-arg", action="append", default=[], help="Docker build argument. Repeatable.")
    args = parser.parse_args()

    tags = validate_tags(args.tags or default_tags())
    full_tags = [f"{args.repository}:{tag}" for tag in tags]

    if not args.skip_verify:
        run([sys.executable, "scripts/verify.py"], "Pre-build verification")

    build_image(args.platform, full_tags, args.build_arg, args.push)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
