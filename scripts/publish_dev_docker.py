#!/usr/bin/env python3
"""Build and optionally push Docker images from the dev branch."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "jdcb4/podcast-ad-remover"
DEV_BRANCH = "dev"


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


def git_output(*args: str) -> str:
    result = subprocess.run(
        [executable("git"), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_dev_checkout(branch: str, is_clean: bool) -> None:
    if branch != DEV_BRANCH:
        raise SystemExit(
            f"Dev images must be built from branch {DEV_BRANCH!r}; current branch is {branch!r}"
        )
    if not is_clean:
        raise SystemExit("Dev images must be built from a clean committed checkout")


def dev_tags(repository: str, short_sha: str) -> list[str]:
    return [f"{repository}:dev", f"{repository}:dev-{short_sha}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or publish Docker images from dev.")
    parser.add_argument("--push", action="store_true", help="Push tags to Docker Hub.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification checks.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="Docker image repository.")
    parser.add_argument("--platform", default="linux/amd64", help="Docker build platform.")
    args = parser.parse_args()

    branch = git_output("branch", "--show-current")
    is_clean = not git_output("status", "--porcelain")
    validate_dev_checkout(branch, is_clean)

    short_sha = git_output("rev-parse", "--short", "HEAD")
    tags = dev_tags(args.repository, short_sha)

    if not args.skip_verify:
        run([sys.executable, "scripts/verify.py"], "Pre-build verification")

    command = [
        executable("docker"),
        "buildx",
        "build",
        "--platform",
        args.platform,
        "--label",
        f"org.opencontainers.image.revision={git_output('rev-parse', 'HEAD')}",
    ]
    for tag in tags:
        command.extend(["-t", tag])
    command.append("--push" if args.push else "--load")
    command.append(".")

    run(command, "Docker dev publish" if args.push else "Docker dev build")
    print("\nBuilt tags: " + ", ".join(tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
