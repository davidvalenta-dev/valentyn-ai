"""Fetch only the pinned, audited public source files. No weights or source execution."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("research/frontier_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path(".valentyn/frontier-sources"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    count = 0
    for repository in manifest["repositories"]:
        name, commit = repository["requested_repo"], repository.get("commit", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise SystemExit("Invalid pinned repository metadata")
        for entry in repository["files"]:
            if "error" in entry:
                continue
            path = PurePosixPath(entry["path"])
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit("Invalid source path")
            destination = args.output / name.replace("/", "__") / str(path)
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == entry["sha256"]:
                count += 1
                continue
            url = f"https://raw.githubusercontent.com/{name}/{commit}/{path}"
            with urllib.request.urlopen(url, timeout=30) as response:
                source = response.read(2 * 1024 * 1024 + 1)
            if len(source) > 2 * 1024 * 1024 or hashlib.sha256(source).hexdigest() != entry["sha256"]:
                raise SystemExit(f"Pinned source checksum mismatch: {name}/{path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source)
            count += 1
    print(f"Verified {count} pinned source/license files. No model weights downloaded.")


if __name__ == "__main__":
    main()
