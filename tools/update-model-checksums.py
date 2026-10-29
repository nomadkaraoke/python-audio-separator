#!/usr/bin/env python3
"""
Regenerate audio_separator/model-checksums.json: the SHA-256 of every file published in the
release repos that Separator downloads models from. Separator refuses any download (and any
cached file) whose hash doesn't match, so rerun this after uploading new model files.

Files are streamed and hashed, never written to disk, but this reads every asset (~110 GB for
a full run), so run it on the build host. Hashes already in the manifest are kept unless
--rehash is given, so routine runs only download new files.

Usage: tools/update-model-checksums.py [--rehash] [--jobs N]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# Same order Separator tries them in: our mirror first, then the upstream repos.
RELEASES = [
    ("nomadkaraoke/python-audio-separator", "model-configs"),
    ("TRvlvr/model_repo", "all_public_uvr_models"),
    ("Anjok0109/ai_magic", "v5"),
]
# Demucs checkpoints are downloaded straight from Meta's CDN when the mirror doesn't have them.
EXTRA_URLS = [
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th",
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th",
]
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "audio_separator", "model-checksums.json")


def list_assets():
    headers = {"Accept": "application/vnd.github+json"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    for repo, tag in RELEASES:
        response = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers=headers, timeout=60)
        response.raise_for_status()
        for asset in response.json()["assets"]:
            yield asset["name"], asset["browser_download_url"]
    for url in EXTRA_URLS:
        yield url.split("/")[-1], url


def sha256_of_url(url, attempts=5):
    for attempt in range(1, attempts + 1):
        digest = hashlib.sha256()
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    digest.update(chunk)
            return digest.hexdigest()
        except requests.RequestException as e:
            if attempt == attempts:
                raise
            print(f"Retrying {url} after error: {e}", file=sys.stderr)
            time.sleep(5 * attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rehash", action="store_true", help="re-download and re-hash files already in the manifest")
    parser.add_argument("--jobs", type=int, default=6, help="parallel downloads")
    args = parser.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    known = {} if args.rehash else manifest["sha256"]

    # The first source listing a filename is the one Separator tries first.
    # .json assets are snapshots of mutable metadata lists, which Separator fetches live and doesn't verify.
    sources = {}
    for name, url in list_assets():
        if not name.endswith(".json"):
            sources.setdefault(name, []).append(url)

    to_hash = [(name, url) for name, urls in sources.items() for url in (urls if args.rehash or name not in known else [])]
    print(f"{len(sources)} published files, hashing {len(to_hash)} downloads")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        hashes = list(pool.map(lambda item: (item[0], item[1], sha256_of_url(item[1])), to_hash))

    found = dict(known)
    conflicts = []
    for name, url, digest in hashes:
        if name in found and found[name] != digest:
            conflicts.append(f"{name}: {url} has {digest}, expected {found[name]}")
            continue
        found[name] = digest

    if conflicts:
        # Separator accepts one hash per filename, so a source serving different bytes would always fail verification.
        print("Sources disagree on file contents:\n  " + "\n  ".join(conflicts), file=sys.stderr)
        sys.exit(1)

    manifest["sha256"] = dict(sorted(found.items()))
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(found)} checksums to {os.path.normpath(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
