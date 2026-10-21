"""Download the photo set used by the benchmarks into data/ (about 740 MB).

    python tools/fetch_photos.py

Photos: iPhone 14 Plus, full resolution, metadata removed losslessly
(tools/strip_metadata.py), so decoded pixels match what the recorded results
were measured on. Licensed CC BY 4.0 (see LICENSE-CONTENT).
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/parallaxmetrology/spatial-kernel/releases/download/photos-v1/spatial-kernel-photos-v1.zip"
SHA256 = "7ba3755d6e6d55523fe8da4c440c54cd98fc6ff9620b6a68e97848f26ef558e5"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    zpath = data / "spatial-kernel-photos-v1.zip"
    if not zpath.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
    h = hashlib.sha256()
    with open(zpath, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        print("checksum mismatch; delete the zip and retry", file=sys.stderr)
        return 1
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data)
    print("photos in", data, ":", ", ".join(sorted(p.name for p in data.iterdir() if p.is_dir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
