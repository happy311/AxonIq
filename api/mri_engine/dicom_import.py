"""
Sandboxed DICOM-zip → NIfTI converter. Runs in a SUBPROCESS (hard memory/CPU/file-size limits,
scrubbed env) so a hostile archive can only kill this child:

    python -m api.mri_engine.dicom_import <zip> <out.nii.gz> <max_unzipped_bytes> <max_dim> <max_voxels> <mem_mb> <cpu_s>

Prints one JSON line: {"ok": true, "slices": N, "shape": [...]} or {"ok": false, "error": "..."}.

Zip handling: nothing is ever extracted to disk and member names are never used as paths (no
zip-slip); encrypted archives, too many entries, oversize members, absurd compression ratios and
an unzipped total over the limit are all rejected. Only MR image series are accepted; the series
with the most slices is converted. Compressed DICOM transfer syntaxes (JPEG etc.) need extra
decoder plugins and are reported as unsupported rather than guessed at.
"""
import io
import json
import sys
import zipfile
from collections import Counter

MAX_ENTRIES = 5000
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_RATIO = 250            # per-member uncompressed/compressed for members > 1 MB
MIN_SLICES = 8


class Reject(Exception):
    pass


def _emit(obj):
    print(json.dumps(obj))
    sys.stdout.flush()


def _limits(mem_mb: int, cpu_s: int, max_out: int) -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_out, max_out))   # may write ONLY the output file, capped
    except Exception:
        pass


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with zf.open(info) as f:
        data = f.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise Reject("a file inside the zip is too large")
    return data


def convert(zip_path: str, out_path: str, max_unzipped: int, max_dim: int, max_vox: int) -> dict:
    import numpy as np
    import nibabel as nib
    import pydicom
    from pydicom.errors import InvalidDicomError

    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise Reject("not a valid zip file")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if not infos:
            raise Reject("the zip is empty")
        if len(infos) > MAX_ENTRIES:
            raise Reject(f"too many files in the zip (max {MAX_ENTRIES})")
        total = 0
        for i in infos:
            if i.flag_bits & 0x1:
                raise Reject("password-protected zips are not supported")
            total += i.file_size
            if i.file_size > MAX_MEMBER_BYTES:
                raise Reject("a file inside the zip is too large")
            if i.file_size > 1_000_000 and i.compress_size and i.file_size / i.compress_size > MAX_RATIO:
                raise Reject("suspicious compression ratio (possible zip bomb)")
        if total > max_unzipped:
            raise Reject("zip contents are too large when unzipped")

        # ── pass 1: headers only → group image slices by series ──────────────
        series = {}                                   # uid -> list[(info, hdr)]
        for i in infos:
            raw = _read_member(zf, i)
            try:
                ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
            except (InvalidDicomError, Exception):
                continue                              # DICOMDIR, README, .DS_Store, … — ignored
            if getattr(ds, "Modality", None) != "MR":
                continue
            if not all(hasattr(ds, a) for a in ("Rows", "Columns", "SeriesInstanceUID")):
                continue
            if int(getattr(ds, "SamplesPerPixel", 1)) != 1 or int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
                continue
            series.setdefault(str(ds.SeriesInstanceUID), []).append((i, ds))
        if not series:
            raise Reject("no MR image slices found in the zip (is it a DICOM export of an MRI?)")
        uid, items = max(series.items(), key=lambda kv: len(kv[1]))
        # keep the dominant in-plane size
        (rows, cols), _ = Counter((int(h.Rows), int(h.Columns)) for _, h in items).most_common(1)[0]
        items = [(i, h) for i, h in items if (int(h.Rows), int(h.Columns)) == (rows, cols)]
        n = len(items)
        if n < MIN_SLICES:
            raise Reject(f"the largest series has only {n} slices; a 3D FLAIR volume is needed")
        if max(rows, cols, n) > max_dim or rows * cols * n > max_vox:
            raise Reject("the volume is too large")

        # ── geometry ─────────────────────────────────────────────────────────
        h0 = items[0][1]
        iop = [float(x) for x in getattr(h0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
        xd, yd = np.array(iop[:3]), np.array(iop[3:])
        nrm = np.cross(xd, yd)
        if not np.isfinite(nrm).all() or np.linalg.norm(nrm) < 0.5:
            raise Reject("invalid image orientation in the DICOM headers")

        def position(h, idx):
            ipp = getattr(h, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                return float(np.dot(np.array([float(v) for v in ipp]), nrm))
            return float(getattr(h, "InstanceNumber", idx))
        keyed = sorted(((position(h, k), k, i, h) for k, (i, h) in enumerate(items)), key=lambda t: (t[0], t[1]))
        pos = np.array([k[0] for k in keyed])
        if len(set(np.round(pos, 3))) != n:
            raise Reject("the series has duplicate slice positions (multiple volumes/echoes) — upload one 3D series")
        gaps = np.diff(pos)
        dz = float(np.median(np.abs(gaps))) if n > 1 else 1.0
        if getattr(h0, "ImagePositionPatient", None) is None:
            dz = float(getattr(h0, "SliceThickness", 1.0) or 1.0)
        if not dz > 0 or (gaps.size and np.max(np.abs(gaps)) > 3 * dz + 1e-6):
            raise Reject("the series has missing slices (irregular spacing)")
        ps = [float(x) for x in getattr(h0, "PixelSpacing", [1.0, 1.0])]
        dy, dx = ps[0], ps[1]                              # row spacing, column spacing

        # ── pass 2: pixels for the chosen series only ────────────────────────
        slabs = []
        scaled = False
        for _, _, info, _h in keyed:
            ds = pydicom.dcmread(io.BytesIO(_read_member(zf, info)))
            try:
                arr = ds.pixel_array
            except Exception as e:
                raise Reject("could not decode pixel data (compressed DICOM transfer syntax is not supported "
                             f"without extra decoders): {type(e).__name__}")
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            icpt = float(getattr(ds, "RescaleIntercept", 0) or 0)
            if slope != 1.0 or icpt != 0.0:
                arr = arr.astype(np.float32) * np.float32(slope) + np.float32(icpt)
                scaled = True
            slabs.append(arr.T)                            # [col, row]  → i, j
        vol = np.stack(slabs, axis=2)
        if scaled:
            vol = vol.astype(np.float32)
        elif vol.dtype == np.uint16 and vol.max() <= 32767:
            vol = vol.astype(np.int16)
        elif vol.dtype == np.uint8:
            vol = vol.astype(np.int16)

        # first slice position (LPS), then LPS → RAS for NIfTI
        first = keyed[0][3]
        ipp = getattr(first, "ImagePositionPatient", None)
        origin = np.array([float(v) for v in ipp]) if ipp is not None else np.zeros(3)
        A = np.eye(4)
        A[:3, 0], A[:3, 1], A[:3, 2], A[:3, 3] = xd * dx, yd * dy, nrm * dz, origin
        A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ A
        img = nib.Nifti1Image(vol, A)
        img.set_qform(A, code=1)
        img.set_sform(A, code=1)
        nib.save(img, out_path)
        return {"slices": n, "shape": [int(s) for s in vol.shape]}


def main() -> None:
    zip_path, out_path = sys.argv[1], sys.argv[2]
    max_unzipped, max_dim, max_vox, mem_mb, cpu_s = (int(x) for x in sys.argv[3:8])
    _limits(mem_mb, cpu_s, max_unzipped)
    try:
        res = convert(zip_path, out_path, max_unzipped, max_dim, max_vox)
        _emit({"ok": True, **res})
    except Reject as e:
        _emit({"ok": False, "error": str(e)})
    except MemoryError:
        _emit({"ok": False, "error": "the scan needs too much memory"})
    except Exception as e:
        _emit({"ok": False, "error": f"conversion failed ({type(e).__name__})"})


if __name__ == "__main__":
    main()
