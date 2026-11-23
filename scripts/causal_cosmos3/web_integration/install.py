"""Install reviewed Cobot web integration, preserving originals and rejecting drift."""

import hashlib
import json
import shutil
import time
from pathlib import Path


def main():
    source = Path(__file__).resolve().parent
    destination = Path("/home/agilex/cobot_magic/collect_data")
    hashes = json.loads((source / "original_sha256.json").read_text())
    changed = []
    for name, digest in hashes.items():
        current = (destination / name).read_bytes()
        if current == (source / name).read_bytes():
            continue
        if hashlib.sha256(current).hexdigest() != digest:
            raise SystemExit("File changed since review; refusing overwrite: " + name)
        changed.append(name)
    backup = destination.parent / "outputs" / ("veomni-piper-web-" + time.strftime("%Y%m%d-%H%M%S"))
    for name in changed:
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination / name, saved)
    for name in changed:
        target = destination / name
        temporary = target.with_name(target.name + ".veomni-new")
        shutil.copy2(source / name, temporary)
        temporary.replace(target)
    print("Installed", len(changed), "files; backup:", backup if changed else "unchanged")


if __name__ == "__main__":
    main()
