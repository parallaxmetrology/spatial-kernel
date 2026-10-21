"""Copy JPEGs with every APPn/COM segment removed (EXIF, maker notes, thumbnails,
ICC, XMP). Lossless: the compressed image data is copied byte for byte, so decoded
pixels are identical. Usage: python tools/strip_metadata.py SRC_DIR DST_DIR"""
import sys
from pathlib import Path


def strip(data: bytes) -> bytes:
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    out, i = bytearray(b"\xff\xd8"), 2
    while i < len(data):
        assert data[i] == 0xFF, f"bad marker at {i}"
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: copy the rest verbatim
            out += data[i:]
            break
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if not (0xE0 <= marker <= 0xEF or marker == 0xFE):
            out += data[i:i + 2 + length]
        i += 2 + length
    return bytes(out)


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg"}:
            (dst / f.name).write_bytes(strip(f.read_bytes()))
