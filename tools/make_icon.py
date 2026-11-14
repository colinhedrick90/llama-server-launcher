from pathlib import Path
import struct
import zlib


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def make_png(size: int = 64) -> bytes:
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            border = x < 5 or y < 5 or x >= size - 5 or y >= size - 5
            chip = 15 <= x < 49 and 15 <= y < 49
            core = 24 <= x < 40 and 24 <= y < 40
            if border:
                color = (24, 31, 42, 255)
            elif core:
                color = (248, 250, 252, 255)
            elif chip:
                color = (44, 163, 99, 255)
            else:
                color = (24, 31, 42, 255)
            row.extend(color)
        rows.append(bytes(row))
    raw = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )


def main() -> None:
    target = Path(__file__).parent.parent / "assets" / "llama-server-launcher.ico"
    target.parent.mkdir(parents=True, exist_ok=True)
    png = make_png()
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", 64, 64, 0, 0, 1, 32, len(png), 22)
    target.write_bytes(header + entry + png)


if __name__ == "__main__":
    main()

