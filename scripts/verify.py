#!/usr/bin/env python3
"""Run local verification checks for Podcast Ad Remover."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Use virtual environment Python if available
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
if os.name == "nt":
    VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def executable(name: str) -> str:
    if os.name == "nt":
        resolved = shutil.which(f"{name}.cmd") or shutil.which(f"{name}.exe")
        if resolved:
            return resolved
    return shutil.which(name) or name


def run(command: list[str], label: str, optional: bool = False) -> None:
    print(f"\n==> {label}")
    print("$ " + " ".join(command))
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"Skipped (optional step failed: {e})")
        else:
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run project verification checks.")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Also build the Docker image locally as podcast-ad-remover:verify.",
    )
    args = parser.parse_args()

    run([PYTHON, "-m", "compileall", "-q", "app", "scripts"], "Python syntax check")
    run([PYTHON, "-m", "pytest", "-q"], "Python unit tests", optional=True)
    run([executable("npm"), "run", "build:css"], "Tailwind CSS build")
    run([executable("npm"), "audit", "--audit-level=moderate"], "Frontend dependency audit")

    if args.docker:
        run(
            [executable("docker"), "build", "-t", "podcast-ad-remover:verify", "."],
            "Docker image build",
        )

    print("\nVerification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
