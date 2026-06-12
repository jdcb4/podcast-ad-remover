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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or publish non-release Docker tags.")
    parser.add_argument("--push", action="store_true", help="Push tags to Docker Hub.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification checks.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="Docker image repository.")
    parser.add_argument("--platform", default="linux/amd64", help="Docker build platform.")
    parser.add_argument("--tag", action="append", dest="tags", help="Non-release tag to build. Repeatable.")
    parser.add_argument("--build-arg", action="append", default=[], help="Docker build argument. Repeatable.")
    parser.add_argument("--no-tts", action="store_true", help="Skip Piper TTS dependencies for images where Piper is unavailable.")
    args = parser.parse_args()

    tags = validate_tags(args.tags or default_tags())
    full_tags = [f"{args.repository}:{tag}" for tag in tags]

    if not args.skip_verify:
        run([sys.executable, "scripts/verify.py"], "Pre-build verification")

    command = [
        executable("docker"),
        "buildx",
        "build",
        "--platform",
        args.platform,
    ]
    build_args = list(args.build_arg)
    if args.no_tts:
        build_args.append("INSTALL_TTS=0")
    for build_arg in build_args:
        command.extend(["--build-arg", build_arg])
    for full_tag in full_tags:
        command.extend(["-t", full_tag])
    command.append("--push" if args.push else "--load")
    command.append(".")

    run(command, "Docker experimental publish" if args.push else "Docker experimental build")
    print("\nBuilt tags: " + ", ".join(full_tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
