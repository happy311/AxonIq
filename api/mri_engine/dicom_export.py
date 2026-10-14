"""
NIfTI → DICOM series → ZIP.

The platform stores patient scans as NIfTI (what the segmentation engine consumes). Doctors
always download DICOM, so each download converts the stored volume into an MR Image Storage
series (one .dcm per axial slice) packed in a zip.

Honest limits (also written into README.txt inside every zip): this is a *conversion* from NIfTI,
so scanner/sequence metadata (TR/TE, manufacturer, original patient demographics) does not exist
and is not invented. Geometry (spacing, orientation, position) comes from the NIfTI affine; the
patient is identified only by an opaque id, never by name or e-mail.
"""
from __future__ import annotations

import io
import os
import tempfile
import threading
import zipfile
from datetime import datetime

import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
_export_gate = threading.Semaphore(1)        # volumes can be hundreds of MB — one conversion at a time

README = (
    "AxonIQ DICOM export\r\n"
    "Converted from a NIfTI FLAIR volume uploaded by the patient. Geometry comes from the NIfTI header;\r\n"
    "original scanner metadata is not available and has not been invented. Patient is identified by an\r\n"
    "opaque id only. Intended for review/viewing; not an original acquisition. Open the DICOM folder in\r\n"
    "any DICOM viewer (e.g. Horos, 3D Slicer, RadiAnt, Weasis).\r\n"
)


def _to_uint16_with_rescale(vol: np.ndarray):
    """Return (pixel array int16/uint16 view-compatible, slope, intercept, signed)."""
    v = np.nan_to_num(np.asarray(vol, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(v.min()), float(v.max())
    if np.issubdtype(vol.dtype, np.integer) and vmin >= -32768 and vmax <= 32767:
        return vol.astype(np.int16), 1.0, 0.0, True
    span = vmax - vmin
    if span <= 0:
        return np.zeros(vol.shape, np.uint16), 1.0, vmin, False
    slope = span / 65535.0
    return np.round((v - vmin) / slope).astype(np.uint16), slope, vmin, False


def nifti_to_dicom_zip(nifti_path: str, zip_path: str, *, patient_id: str, description: str = "FLAIR",
                       study_dt: datetime | None = None) -> int:
    """Write the zip to zip_path; return number of slices."""
    with _export_gate:
        img = nib.as_closest_canonical(nib.load(nifti_path))
        data = np.asanyarray(img.dataobj)
        while data.ndim > 3:                                   # 4-D → first volume
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError("Scan is not a 3-D volume")
        pix, slope, intercept, signed = _to_uint16_with_rescale(data)
        affine = img.affine
        n_slices = pix.shape[2]

        # NIfTI/RAS+ → DICOM/LPS: negate x and y
        def lps(p):
            return [-float(p[0]), -float(p[1]), float(p[2])]
        col_vec = affine[:3, 0] / (np.linalg.norm(affine[:3, 0]) or 1)    # direction along a row (i)
        row_vec = affine[:3, 1] / (np.linalg.norm(affine[:3, 1]) or 1)    # direction down a column (j)
        iop = [round(x, 6) for x in (lps(col_vec) + lps(row_vec))]
        dx, dy, dz = (float(np.linalg.norm(affine[:3, k])) or 1.0 for k in range(3))

        dt = study_dt or datetime.utcnow()
        study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
        rows, cols = pix.shape[1], pix.shape[0]                # slice stored as pix[:,:,k].T → rows=j, cols=i

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("README.txt", README)
            for k in range(n_slices):
                meta = FileMetaDataset()
                meta.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
                sop_uid = generate_uid()
                meta.MediaStorageSOPInstanceUID = sop_uid
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.ImplementationClassUID = generate_uid()
                ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
                ds.SOPClassUID, ds.SOPInstanceUID = MR_IMAGE_STORAGE, sop_uid
                ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, frame_uid
                ds.PatientName, ds.PatientID = f"ANON^{patient_id}", patient_id
                ds.PatientBirthDate, ds.PatientSex = "", ""
                ds.StudyDate, ds.StudyTime = dt.strftime("%Y%m%d"), dt.strftime("%H%M%S")
                ds.ContentDate, ds.ContentTime = ds.StudyDate, ds.StudyTime
                ds.Modality, ds.StudyDescription = "MR", f"{description} (converted from NIfTI)"
                ds.SeriesDescription, ds.SeriesNumber, ds.StudyID = description, 1, "1"
                ds.AccessionNumber, ds.ReferringPhysicianName = "", ""
                ds.Manufacturer = "AxonIQ NIfTI-to-DICOM converter"
                ds.InstanceNumber = k + 1
                ds.ImageType = ["DERIVED", "SECONDARY"]
                ds.ImagePositionPatient = [round(x, 4) for x in lps((affine @ np.array([0, 0, k, 1.0]))[:3])]
                ds.ImageOrientationPatient = iop
                ds.PixelSpacing = [round(dy, 5), round(dx, 5)]
                ds.SliceThickness, ds.SpacingBetweenSlices = round(dz, 5), round(dz, 5)
                ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
                ds.Rows, ds.Columns = rows, cols
                ds.BitsAllocated = ds.BitsStored = 16
                ds.HighBit, ds.PixelRepresentation = 15, 1 if signed else 0
                ds.RescaleSlope, ds.RescaleIntercept = f"{slope:.10g}", f"{intercept:.10g}"
                ds.PixelData = np.ascontiguousarray(pix[:, :, k].T).tobytes()
                ds["PixelData"].VR = "OW"
                buf = io.BytesIO()
                ds.save_as(buf, enforce_file_format=True)
                zf.writestr(f"DICOM/IM{k + 1:04d}.dcm", buf.getvalue())
        return n_slices


def make_zip_from_stored(stored_path: str, suffix: str, patient_id: str, study_dt: datetime | None,
                         out_dir: str) -> str:
    """Decrypt the stored scan to a private temp file, convert, return path of the finished zip.
    Caller deletes the returned file after streaming it."""
    from api.core.crypto import decrypt_file
    os.makedirs(out_dir, exist_ok=True)
    fd, plain = tempfile.mkstemp(suffix=suffix, dir=out_dir)
    os.close(fd)
    zfd, zpath = tempfile.mkstemp(suffix=".zip", dir=out_dir)
    os.close(zfd)
    try:
        decrypt_file(stored_path, plain)
        nifti_to_dicom_zip(plain, zpath, patient_id=patient_id, study_dt=study_dt)
        return zpath
    except BaseException:
        try:
            os.unlink(zpath)
        except OSError:
            pass
        raise
    finally:
        try:
            os.unlink(plain)
        except OSError:
            pass
