"""
Redact the top banner/header of a DICOM ultrasound series by zeroing pixels,
while preserving each file's original transfer syntax.

Design notes (see plan for full rationale):

  * The banner cutoff row (y0) is found automatically from
    SequenceOfUltrasoundRegions (0018,6011) -> RegionLocationMinY0 (0018,601A),
    the topmost edge of the scanned-image region. One series-wide cutoff
    (the min across all files) is used so every file gets a consistent
    boundary, with a fallback for any file missing the tag.

  * Zeroing is always done in RGB space, never in raw YBR. YBR "black" is
    Y=0,Cb=128,Cr=128 -- writing zeros directly into a YBR array produces a
    *green* bar, not black. So: decode -> RGB -> zero rows -> re-encode in
    the original photometric interpretation/transfer syntax.

    IMPORTANT pydicom-version-specific gotcha discovered while prototyping:
    with pydicom's default (3.0+) pixel backend, `ds.pixel_array` ALREADY
    converts YBR_FULL/YBR_FULL_422 data to RGB automatically (see
    `Dataset.pixel_array_options` docs: "if raw=False (the default) ...
    photometric interpretation YBR_FULL_422 will be converted to RGB").
    Calling `convert_color_space(arr, "YBR_FULL_422", "RGB")` again on top
    of that double-converts and corrupts the image (observed as a uniform
    green tint, e.g. RGB (0,0,0) -> (0,135,0), across the whole redacted
    banner during prototype testing). So on the READ side we trust
    ds.pixel_array's already-RGB output directly and do NOT call
    convert_color_space again. convert_color_space IS still needed and
    correct on the WRITE side (see encode_jpeg_baseline_frames below),
    since there we're manually building YBR JPEG data from scratch.

  * Output preserves each input's transfer syntax:
      - uncompressed inputs (the 29 real RGB files) are written back
        losslessly as RGB, same (uncompressed) transfer syntax.
      - JPEG Baseline / YBR_FULL_422 input (the 1 real cine file) is
        re-encoded to JPEG Baseline via Pillow + pydicom.encaps.encapsulate.
        NOTE: a raw-GDCM-Image-API encode path was prototyped as a
        standards-faithful alternative, but it SEGFAULTED the interpreter
        (gdcm::Exception / aborted process) on a minimal test case, and no
        gdcmconv CLI was available as a safer alternative. Given that
        instability, this script uses the Pillow-based encoder exclusively
        -- it round-trips correctly and does not risk crashing on real
        patient data. This re-encode is lossy for the *entire* frame (not
        just the banner) since it's a fresh JPEG pass -- expected and
        acceptable for de-identification, but worth knowing.

  * Scope is pixels only, per request: header tags (PatientName, etc.) are
    left untouched. Originals are never modified; output goes to a
    separate directory.
"""
import sys
import os
import glob
import io

import numpy as np
import pydicom
from pydicom.encaps import encapsulate
from pydicom.pixels import convert_color_space
from pydicom.uid import JPEGBaseline8Bit
from PIL import Image


def region_cutoff(ds):
    """Topmost RegionLocationMinY0 across all ultrasound regions in this
    file, or None if the tag is absent."""
    if "SequenceOfUltrasoundRegions" not in ds:
        return None
    ys = [
        item.get("RegionLocationMinY0")
        for item in ds.SequenceOfUltrasoundRegions
        if item.get("RegionLocationMinY0") is not None
    ]
    return min(ys) if ys else None


def compute_series_cutoff(paths, fallback=None):
    """One series-wide cutoff = min(region_cutoff) across all files that
    have the tag. Files missing the tag are reported so the fallback
    (series cutoff, or an explicit override) can be applied to them too."""
    cutoffs = []
    missing = []
    for p in paths:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        y0 = region_cutoff(ds)
        if y0 is None:
            missing.append(p)
        else:
            cutoffs.append(y0)

    if not cutoffs:
        if fallback is None:
            raise ValueError(
                "No SequenceOfUltrasoundRegions found in any file, and no "
                "fallback cutoff was provided."
            )
        return fallback, missing

    return min(cutoffs), missing


def to_rgb(ds):
    """Return this dataset's pixel data as RGB.

    pydicom's default (3.0+) backend already converts YBR_FULL /
    YBR_FULL_422 pixel data to RGB when accessing ds.pixel_array (raw=False
    is the default) -- so for RGB and YBR* inputs alike, ds.pixel_array is
    already what we want; no manual convert_color_space call here (see the
    module docstring for why that would double-convert and corrupt data).
    Anything else (e.g. PALETTE COLOR, MONOCHROME) isn't handled by this
    auto-conversion and is rejected rather than silently mishandled.
    """
    photometric = ds.PhotometricInterpretation
    if photometric != "RGB" and not str(photometric).startswith("YBR"):
        raise ValueError(f"Unsupported PhotometricInterpretation for redaction: {photometric}")
    return ds.pixel_array


def redact_pixels(rgb_arr, y0):
    """Zero rows [0:y0) in RGB space. Handles both single-frame
    (rows, cols, 3) and multi-frame (frames, rows, cols, 3) arrays."""
    out = rgb_arr.copy()
    if out.ndim == 3:
        out[0:y0, :, :] = 0
    elif out.ndim == 4:
        out[:, 0:y0, :, :] = 0
    else:
        raise ValueError(f"Unexpected pixel array ndim={out.ndim} (shape={out.shape})")
    return out


def encode_jpeg_baseline_frames(rgb_frames):
    """rgb_frames: (n_frames, rows, cols, 3) uint8 RGB -> list of raw JPEG
    byte-strings, one per frame, encoded as YBR_FULL_422 (4:2:2 chroma
    subsampling, matching the DICOM photometric interpretation)."""
    encoded = []
    for i in range(rgb_frames.shape[0]):
        ybr = convert_color_space(rgb_frames[i], "RGB", "YBR_FULL_422")
        img = Image.fromarray(ybr, mode="YCbCr")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, subsampling="4:2:2")
        encoded.append(buf.getvalue())
    return encoded


def save_same_encoding(ds, rgb_arr, out_path):
    """Write `ds` back out with `rgb_arr` (redacted, RGB) as PixelData,
    preserving the original transfer syntax."""
    orig_ts = ds.file_meta.TransferSyntaxUID
    n_frames = rgb_arr.shape[0] if rgb_arr.ndim == 4 else 1

    if not orig_ts.is_compressed:
        # Uncompressed path (the 29 real RGB files): write the redacted RGB
        # array straight back. Lossless -- no quality loss anywhere.
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelData = rgb_arr.tobytes()
        if "NumberOfFrames" in ds:
            ds.NumberOfFrames = n_frames

    elif orig_ts == JPEGBaseline8Bit:
        frames = rgb_arr if rgb_arr.ndim == 4 else rgb_arr[np.newaxis, ...]
        encoded = encode_jpeg_baseline_frames(frames)
        ds.PhotometricInterpretation = "YBR_FULL_422"
        ds.PlanarConfiguration = 0
        ds.PixelData = encapsulate(encoded)
        ds.NumberOfFrames = len(encoded)

    else:
        raise NotImplementedError(
            f"Re-encoding for transfer syntax {orig_ts.name} ({orig_ts}) is not "
            f"implemented in this script. Only uncompressed and JPEG Baseline "
            f"(1.2.840.10008.1.2.4.50) are handled -- extend save_same_encoding() "
            f"if the real series contains another compressed transfer syntax."
        )

    ds.save_as(out_path, enforce_file_format=True)


def redact_file(path, y0, out_dir):
    ds = pydicom.dcmread(path)
    rgb = to_rgb(ds)
    rgb_redacted = redact_pixels(rgb, y0)

    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_redacted.dcm")
    save_same_encoding(ds, rgb_redacted, out_path)
    return out_path, ds.file_meta.TransferSyntaxUID, ds.get("NumberOfFrames", 1)


def redact_series(paths, y0, out_dir, continue_on_error=True):
    """Redact every path in `paths` to `out_dir` at cutoff `y0`.

    Returns (results, errors):
      results = list of (in_path, out_path, transfer_syntax, n_frames) for
                files that redacted successfully.
      errors  = list of (in_path, exception) for files that raised.

    On a per-file exception: if continue_on_error (the default), the error
    is recorded and the loop keeps going -- one bad file (e.g. an unsupported
    transfer syntax hitting save_same_encoding's NotImplementedError, or a
    corrupt/unreadable file) must not abort a whole production batch before
    the remaining files are processed. If continue_on_error is False, the
    exception is re-raised immediately (fail-fast).
    """
    results = []
    errors = []
    for path in paths:
        try:
            out_path, ts, n_frames = redact_file(path, y0, out_dir)
        except Exception as e:
            print(f"{path} -> ERROR: {type(e).__name__}: {e}")
            errors.append((path, e))
            if not continue_on_error:
                raise
            continue
        print(f"{path} -> {out_path}  (frames={n_frames}, ts={ts.name})")
        results.append((path, out_path, ts, n_frames))
    return results, errors


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "raw"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "redacted"
    fallback_cutoff = None  # set to an int to tolerate files with no region tag

    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.dcm")))
    if not paths:
        print(f"No .dcm files found in {raw_dir!r}")
        return 1

    y0, missing = compute_series_cutoff(paths, fallback=fallback_cutoff)
    print(f"Series-wide redaction cutoff: y0={y0} (rows [0:{y0}) will be zeroed)")
    if missing:
        print(f"WARNING: {len(missing)} file(s) lacked SequenceOfUltrasoundRegions, "
              f"using the series-wide cutoff for them: {missing}")

    _results, errors = redact_series(paths, y0, out_dir, continue_on_error=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
