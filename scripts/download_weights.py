#!/usr/bin/env python3
"""Download and verify the three GeoDSSOP-PDB W3 release assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
from urllib.request import Request, urlopen

from geodssop.io import sha256_file


REPOSITORY = "eagleccnu/GeoDSSOP"
TAG = "v0.1.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="environment-variable name containing an optional GitHub token",
    )
    return parser.parse_args(argv)


def request_headers(token: str | None, *, binary: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": "GeoDSSOP-weight-downloader",
        "Accept": "application/octet-stream" if binary else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def release_assets(token: str | None) -> dict[str, str]:
    url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{TAG}"
    with urlopen(Request(url, headers=request_headers(token))) as response:
        payload = json.load(response)
    return {str(asset["name"]): str(asset["url"]) for asset in payload["assets"]}


def download(api_url: str, destination: Path, token: str | None) -> None:
    request = Request(api_url, headers=request_headers(token, binary=True))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        with urlopen(request) as response, open(temporary_name, "wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        os.replace(temporary_name, destination)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(__file__).resolve().parents[1] / "checkpoints" / "weights-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token = os.environ.get(args.token_env)
    assets = release_assets(token)
    for member in manifest["members"]:
        destination = args.output_dir / member["filename"]
        if destination.exists():
            if sha256_file(destination) == member["sha256"]:
                print(f"verified existing {destination.name}")
                continue
            raise RuntimeError(f"existing file has the wrong SHA-256: {destination}")
        api_url = assets.get(member["filename"])
        if api_url is None:
            raise RuntimeError(f"release asset is missing: {member['filename']}")
        download(api_url, destination, token)
        if sha256_file(destination) != member["sha256"]:
            raise RuntimeError(f"downloaded file has the wrong SHA-256: {destination}")
        print(f"downloaded and verified {destination.name}")
    destination_manifest = args.output_dir / "weights-manifest.json"
    if not destination_manifest.exists():
        destination_manifest.write_bytes(manifest_path.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
