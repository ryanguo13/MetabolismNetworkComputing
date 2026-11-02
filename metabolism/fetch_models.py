"""Fetch pinned upstream artifacts, or verify existing ones, using only stdlib."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true", help="Verify local files without network access")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "models"
    entries = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for entry in entries:
        path = directory / entry['file']
        if not path.exists() and not args.check:
            with urlopen(entry['download_url'], timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError(f"Upstream content changed: {entry['file']}; review before updating pin")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Missing or changed artifact: {path}")
        print(f"OK {entry['file']}")


if __name__ == "__main__":
    main()
