"""Buildx with a strictly native Linux AMD64 fallback."""
import shutil
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def executable(name):
    return shutil.which(name + '.exe') or shutil.which(name) or name

def run(command, label):
    print(label)
    subprocess.run(command, cwd=ROOT, check=True)

def build_image(platform: str, tags: list[str], build_args: list[str], push: bool, labels: list[str] | None = None) -> None:
    docker = executable("docker")
    available = subprocess.run([docker, "buildx", "version"], capture_output=True).returncode == 0
    if available:
        command = [docker, "buildx", "build", "--platform", platform]
    else:
        try:
            host = subprocess.run([docker, "info", "--format", "{{.OSType}}/{{.Architecture}}"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SystemExit('Buildx is unavailable and the Docker daemon architecture could not be verified.') from error
        if platform != "linux/amd64" or host not in {"linux/amd64", "linux/x86_64"}:
            raise SystemExit("Buildx is required unless target and Docker daemon are both Linux AMD64")
        command = [docker, "build", "--platform", "linux/amd64"]
    for label in labels or []:
        command.extend(["--label", label])
    for arg in build_args:
        command.extend(["--build-arg", arg])
    for tag in tags:
        command.extend(["-t", tag])
    if available:
        command.append("--push" if push else "--load")
    command.append(".")
    run(command, "Docker build")
    if push and not available:
        for tag in tags:
            run([docker, "push", tag], "Push image tag")
    print(("Pushed tags: " if push else "Built locally: ") + ", ".join(tags))

