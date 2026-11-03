from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import requests

MODEL_CACHE = Path(os.getenv("AI_IMAGE_MODEL_CACHE", Path.home() / ".cache" / "ai-image-enhancement")).expanduser()
DEOLDIFY_REPOSITORY = "https://github.com/jantic/DeOldify.git"
DEOLDIFY_COMMIT = "5f86c28923799036ad4b1bf7af8c629ac65efd75"

ASSETS = (
    (
        "https://huggingface.co/TJ-27/Deblur-GAN-V2/resolve/main/deblurgan_generator.keras",
        MODEL_CACHE / "deblurgan_generator.keras",
        "7b6ae0af8bc6b18b8bd761f03d30b58f0287aec0cd4a17d71287d1ae238b3bcf",
    ),
    (
        "https://huggingface.co/databuzzword/deoldify-stable/resolve/main/ColorizeStable_gen.pth",
        MODEL_CACHE / "models" / "ColorizeStable_gen.pth",
        "ca9cd7f43fb8b222c9a70f7b292e305a000694b0ff9d2ae4a6747b1a2e1ee5af",
    ),
)


def download_asset(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing_digest = hashlib.sha256()
        with destination.open("rb") as existing:
            for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                existing_digest.update(chunk)
        existing_hash = existing_digest.hexdigest()
        if existing_hash == expected_sha256:
            print(f"Already verified: {destination}")
            return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(20, 600)) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    digest.update(chunk)

    actual_hash = digest.hexdigest()
    if actual_hash != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch downloading {destination.name}: {actual_hash}")
    temporary_path.replace(destination)
    print(f"Downloaded and verified: {destination}")


def setup_deoldify_source() -> None:
    destination = MODEL_CACHE / "DeOldify"
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise RuntimeError(f"Refusing to overwrite the existing model source directory: {destination}")
        result = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != DEOLDIFY_COMMIT:
            raise RuntimeError(f"DeOldify source at {destination} is not the pinned revision; choose a different AI_IMAGE_MODEL_CACHE.")
        return

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", DEOLDIFY_REPOSITORY], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth=1", "origin", DEOLDIFY_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
    print(f"Installed DeOldify source at {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GAN weights.")
    parser.add_argument("--only", choices=("deblur", "colorize"), help="Download a single model (default: all).")
    args = parser.parse_args()
    for url, destination, expected_sha256 in ASSETS:
        if args.only == "deblur" and "deblurgan" not in destination.name:
            continue
        if args.only == "colorize" and "deblurgan" in destination.name:
            continue
        download_asset(url, destination, expected_sha256)
    if args.only != "deblur":
        setup_deoldify_source()
    print(f"GAN models are ready in {MODEL_CACHE}")


if __name__ == "__main__":
    main()
