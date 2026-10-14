"""
Sandboxed NIfTI validator — runs in a SUBPROCESS with hard memory/CPU limits so that a
malicious file (gzip bomb, absurd dimensions, crafted header) can only kill this child.

    python -m api.mri_engine.validate_nifti <path> <max_decompressed_bytes> <max_dim> <max_voxels> <mem_mb> <cpu_s>

Deliberately imports nothing from the app (no config / secrets); the parent also passes a
scrubbed environment. Prints one JSON line: {"ok": true, ...} or {"ok": false, "error": "..."}.
Exit code is always 0 for a clean verdict; any crash / limit-kill is treated as REJECT by the parent.
"""
import gzip
import json
import struct
import sys


def _fail(msg: str):
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(0)


def _apply_limits(mem_mb: int, cpu_s: int) -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))        # cannot write files
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0)) if hasattr(resource, "RLIMIT_NPROC") else None
    except Exception:
        pass


def parse_header(raw: bytes):
    """Return (dims, bitpix) from a NIfTI-1/2 header or raise ValueError."""
    if len(raw) < 348:
        raise ValueError("file too small to be NIfTI")
    for endian in ("<", ">"):
        (sizeof_hdr,) = struct.unpack(endian + "i", raw[:4])
        if sizeof_hdr == 348:
            if raw[344:348] not in (b"n+1\0", b"ni1\0"):
                raise ValueError("bad NIfTI-1 magic")
            dim = struct.unpack(endian + "8h", raw[40:56])
            (bitpix,) = struct.unpack(endian + "h", raw[72:74])
            break
        if sizeof_hdr == 540:
            if len(raw) < 540 or raw[4:12] != b"n+2\0\r\n\x1a\n":
                raise ValueError("bad NIfTI-2 magic")
            dim = struct.unpack(endian + "8q", raw[16:80])
            (bitpix,) = struct.unpack(endian + "h", raw[14:16])
            break
    else:
        raise ValueError("sizeof_hdr is not 348 or 540")
    ndim = dim[0]
    if not 3 <= ndim <= 4:
        raise ValueError(f"unsupported number of dimensions ({ndim})")
    dims = list(dim[1:1 + ndim])
    if any(d < 1 for d in dims):
        raise ValueError("non-positive dimension")
    return dims, bitpix


def main() -> None:
    path = sys.argv[1]
    max_decomp, max_dim, max_vox, mem_mb, cpu_s = (int(x) for x in sys.argv[2:7])
    _apply_limits(mem_mb, cpu_s)

    with open(path, "rb") as f:
        magic = f.read(2)
    is_gz = magic == b"\x1f\x8b"
    opener = gzip.open if is_gz else open

    try:
        with opener(path, "rb") as f:
            head = f.read(540)
        dims, bitpix = parse_header(head)
    except (OSError, EOFError, ValueError, struct.error, gzip.BadGzipFile) as e:
        _fail(f"invalid NIfTI header: {e}")

    if max(dims) > max_dim:
        _fail(f"dimension too large ({max(dims)} > {max_dim})")
    voxels = 1
    for d in dims:
        voxels *= d
    if voxels > max_vox:
        _fail(f"too many voxels ({voxels} > {max_vox})")
    if bitpix not in (8, 16, 32, 64):
        _fail(f"unsupported bitpix {bitpix}")
    if voxels * bitpix // 8 > max_decomp:
        _fail("declared data size exceeds the decompressed-size limit")

    # Count the REAL decompressed size (a bomb can declare tiny dims and append gigabytes).
    total = 0
    try:
        with opener(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_decomp:
                    _fail("decompressed size exceeds limit (possible gzip bomb)")
    except (OSError, EOFError, gzip.BadGzipFile, zlib_error()) as e:
        _fail(f"corrupt or truncated file: {e}")

    # Finally let nibabel really parse it (limits above bound memory/CPU).
    try:
        import nibabel as nib
        import numpy as np
        img = nib.load(path)
        shape = tuple(int(x) for x in img.shape)
        data = np.asanyarray(img.dataobj)
        if data.size != voxels:
            _fail("data size does not match header")
    except MemoryError:
        _fail("file needs too much memory")
    except Exception as e:                      # nibabel raises many types for bad files
        _fail(f"nibabel could not load file: {type(e).__name__}")

    print(json.dumps({"ok": True, "shape": list(shape), "decompressed_bytes": total}))


def zlib_error():
    import zlib
    return zlib.error


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except MemoryError:
        _fail("file needs too much memory")
    except Exception as e:
        _fail(f"validator error: {type(e).__name__}")
