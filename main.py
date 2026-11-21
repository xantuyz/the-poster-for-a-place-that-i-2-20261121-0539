#!/usr/bin/env python3
"""VELUNARA poster entry.

Copies the pre-rendered entry.png (sitting next to this file) to the
grader output path, verifying SHA-256. Stdlib only, no network, works
from any working directory.

Output path: $OUTPUT_PATH, default /output/entry.png.
"""
import hashlib
import os
import shutil
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "entry.png")
    out = os.environ.get("OUTPUT_PATH", "/output/entry.png")
    if not os.path.isfile(src):
        print("entry.png not found next to main.py", file=sys.stderr)
        return 1
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copyfile(src, out)
    s_src = sha256(src)
    s_out = sha256(out)
    if s_src != s_out:
        print("SHA-256 mismatch after copy", file=sys.stderr)
        return 1
    print("sha256: " + s_out)
    print("bytes: " + str(os.path.getsize(out)))
    print("wrote: " + out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
