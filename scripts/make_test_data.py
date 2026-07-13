"""
Generate synthetic prototype DICOM files that mimic the real Canon ultrasound
series structure, for pipeline development before the real 30 files arrive:

  - raw/rgb_single_XX.dcm   : (1, 960, 1280, 3) RGB, uncompressed (Explicit VR LE)
                              -- stand-in for the 29 real single-frame files.
  - raw/cine_ybr422.dcm     : (N, 960, 1280, 3) YBR_FULL_422, JPEG Baseline
                              -- stand-in for the 1 real 248-frame cine file.
                              (N kept small here for prototyping speed; the
                              real file has 248 frames but the same shape.)

Both carry a simulated top "banner" (rows 0:BANNER_H) with bright synthetic
"text-like" content, a SequenceOfUltrasoundRegions tag whose topmost
RegionLocationMinY0 == BANNER_H, and a distinguishable "live image" body
below so we can verify the redaction zeroes the banner and leaves the body
untouched.

Additionally generates real-OCR-able variants (using actual rendered glyphs,
not blocky rectangles) to exercise dicom_phi_scan's EasyOCR-based scanner:

  - raw/rgb_realtext_01.dcm : real rendered text, entirely within the banner
                              -- should be fully removed by redaction.
  - raw/rgb_realtext_leak.dcm : real rendered text in the banner PLUS one
                              extra text line placed just BELOW the banner
                              cutoff (simulating a corner/overlay PHI label
                              burned directly onto the image) -- exercises
                              ocr_verify.py's "PHI below cutoff" WARN path,
                              since the region-based blackout will not
                              remove it.

Not real patient data -- entirely synthetic pixel content and dummy UIDs.
"""
import os
import io
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    generate_uid,
)
from pydicom.encaps import encapsulate
from pydicom.pixels import convert_color_space
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"

ROWS, COLS = 960, 1280
BANNER_H = 120  # rows [0:BANNER_H) = banner/chrome, rows [BANNER_H:ROWS) = live image
OUT_DIR = "/home/elijah/dicom-redact-pixels/raw"


def make_banner_row_pattern(rows, cols):
    """Bright synthetic 'header chrome' -- bands + blocky 'text' rectangles,
    similar in spirit to a Canon ultrasound top banner (never all-black)."""
    banner = np.zeros((rows, cols, 3), dtype=np.uint8)
    banner[:, :] = (20, 40, 60)  # dark blue-grey background band
    rng = np.random.default_rng(42)
    # simulate a few rows of "text" as bright horizontal streaks
    for row_start in (10, 35, 60, 85):
        row_end = row_start + 14
        n_chars = 18
        char_w = cols // (n_chars * 2)
        for c in range(n_chars):
            x0 = 20 + c * char_w * 2
            if rng.random() > 0.15:
                banner[row_start:row_end, x0:x0 + char_w] = (230, 230, 230)
    return banner


def make_live_image_body(rows, cols, seed):
    """Distinguishable non-black 'ultrasound-like' content for the region
    below the banner: radial gradient + noise, never all-zero."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    cx, cy = cols / 2, 0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = (dist / dist.max() * 200).astype(np.uint8)
    noise = rng.integers(0, 40, size=(rows, cols), dtype=np.uint8)
    gray = np.clip(dist.astype(np.int16) + noise, 10, 255).astype(np.uint8)
    body = np.stack([gray, gray, gray], axis=-1)
    return body


def build_frame(seed):
    frame = np.zeros((ROWS, COLS, 3), dtype=np.uint8)
    frame[0:BANNER_H] = make_banner_row_pattern(BANNER_H, COLS)
    frame[BANNER_H:ROWS] = make_live_image_body(ROWS - BANNER_H, COLS, seed)
    return frame


def build_frame_realtext(seed, leak_below_cutoff=False):
    """Same layout as build_frame, but the banner carries actual rendered
    glyphs (via PIL ImageDraw + a real TTF) instead of blocky rectangles, so
    EasyOCR has real text to detect. Optionally also draws one extra text
    line a bit below the banner cutoff, simulating a corner/overlay PHI
    label burned directly onto the image -- the region-based blackout
    won't touch it, which is exactly what ocr_verify.py's pre-scan should
    flag."""
    frame = np.zeros((ROWS, COLS, 3), dtype=np.uint8)
    frame[0:BANNER_H] = (20, 40, 60)  # dark blue-grey banner background
    frame[BANNER_H:ROWS] = make_live_image_body(ROWS - BANNER_H, COLS, seed)

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, size=22)
    lines = [
        "PATIENT: TEST^SYNTHETIC   ID: SYN0001",
        "DOB: 01/01/1970   SEX: F",
        "SYNTHETIC MEDICAL CENTER   Acc: 999999",
    ]
    for i, line in enumerate(lines):
        draw.text((20, 8 + i * 30), line, fill=(230, 230, 230), font=font)

    if leak_below_cutoff:
        # Placed a few rows *into* the live-image region (below BANNER_H) --
        # a corner overlay label that the banner cutoff will NOT remove.
        draw.text((20, BANNER_H + 6), "OVERLAY ID: SYN0001", fill=(255, 255, 0), font=font)

    return np.array(img)


def add_region_tags(ds, min_y0=BANNER_H):
    item = Dataset()
    item.RegionSpatialFormat = 1  # 2D
    item.RegionDataType = 1  # tissue
    item.RegionLocationMinX0 = 0
    item.RegionLocationMinY0 = min_y0
    item.RegionLocationMaxX1 = COLS - 1
    item.RegionLocationMaxY1 = ROWS - 1
    ds.SequenceOfUltrasoundRegions = [item]


def base_dataset(sop_class_uid="1.2.840.10008.5.1.4.1.1.6.1"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "US"
    ds.Manufacturer = "SYNTHETIC-TEST"
    ds.ManufacturerModelName = "prototype-generator"
    ds.PatientName = "Test^Synthetic"
    ds.PatientID = "SYN0001"
    ds.Rows = ROWS
    ds.Columns = COLS
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 3
    return ds


def make_rgb_single(idx):
    ds = base_dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.NumberOfFrames = 1

    frame = build_frame(seed=1000 + idx)
    add_region_tags(ds)
    ds.PixelData = frame.tobytes()

    out_path = os.path.join(OUT_DIR, f"rgb_single_{idx:02d}.dcm")
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


def make_rgb_realtext(name, seed, leak_below_cutoff=False):
    """RGB single-frame file with real rendered banner text (see
    build_frame_realtext), for exercising the OCR-based verification."""
    ds = base_dataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.NumberOfFrames = 1

    frame = build_frame_realtext(seed=seed, leak_below_cutoff=leak_below_cutoff)
    add_region_tags(ds)
    ds.PixelData = frame.tobytes()

    out_path = os.path.join(OUT_DIR, f"{name}.dcm")
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


def make_cine_ybr422(n_frames=8):
    ds = base_dataset()
    ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.PlanarConfiguration = 0
    ds.NumberOfFrames = n_frames
    ds.LossyImageCompression = "01"
    ds.LossyImageCompressionMethod = "ISO_10918_1"
    add_region_tags(ds)

    encoded_frames = []
    for i in range(n_frames):
        rgb = build_frame(seed=2000 + i)
        ybr = convert_color_space(rgb, "RGB", "YBR_FULL_422")
        # Encode via Pillow: JPEG baseline w/ 4:2:0 chroma subsampling is what
        # Pillow's encoder actually supports for "YCbCr"; DICOM's YBR_FULL_422
        # nominally means 4:2:2, but for this synthetic round-trip prototype
        # what matters is exercising encapsulate/decapsulate + color handling,
        # not exact subsampling fidelity.
        img = Image.fromarray(ybr, mode="YCbCr")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        encoded_frames.append(buf.getvalue())

    ds.PixelData = encapsulate(encoded_frames)

    out_path = os.path.join(OUT_DIR, "cine_ybr422.dcm")
    ds.save_as(out_path, enforce_file_format=True)
    return out_path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = []
    for i in range(1, 4):  # a handful of RGB stand-ins (not all 29 -- just enough to exercise the batch driver)
        paths.append(make_rgb_single(i))
    paths.append(make_cine_ybr422(n_frames=8))
    paths.append(make_rgb_realtext("rgb_realtext_01", seed=3000))
    paths.append(make_rgb_realtext("rgb_realtext_leak", seed=3001, leak_below_cutoff=True))
    for p in paths:
        print("wrote", p)
