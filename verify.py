"""
Post-redaction verification: reload every redacted file and check that the
redaction actually did what it claims, rather than trusting redact.py's own
success message.

Checks per file:
  1. Transfer syntax preserved (matches the corresponding original).
  2. Frame count preserved.
  3. pixel_array still decodes.
  4. The banner region [0:y0) is TRUE BLACK once decoded to RGB -- this is
     the check that would catch the "green bar" bug (zeroing raw YBR values
     instead of converting to RGB first).
  5. The region below the banner is NOT all-zero (i.e. we didn't
     accidentally blank the whole image).

Also exports before/after PNG spot-checks for visual review.

Usage: python verify.py [raw_dir] [redacted_dir] [y0]
"""
import sys
import os
import glob

import numpy as np
import pydicom
from PIL import Image


def decode_to_rgb(ds):
    """pydicom's default (3.0+) backend already converts YBR_FULL /
    YBR_FULL_422 pixel data to RGB when reading ds.pixel_array -- see
    redact.py's module docstring / to_rgb() for the double-conversion bug
    this avoids. No manual convert_color_space call needed here."""
    photometric = ds.PhotometricInterpretation
    if photometric != "RGB" and not str(photometric).startswith("YBR"):
        raise ValueError(f"Unsupported PhotometricInterpretation: {photometric}")
    return ds.pixel_array


def first_frame(rgb_arr):
    return rgb_arr[0] if rgb_arr.ndim == 4 else rgb_arr


def banner_region(rgb_arr, y0):
    return rgb_arr[..., 0:y0, :, :] if rgb_arr.ndim == 4 else rgb_arr[0:y0, :, :]


def body_region(rgb_arr, y0):
    return rgb_arr[..., y0:, :, :] if rgb_arr.ndim == 4 else rgb_arr[y0:, :, :]


def verify_pair(orig_path, redacted_path, y0):
    problems = []

    ds_orig = pydicom.dcmread(orig_path)
    ds_red = pydicom.dcmread(redacted_path)

    orig_ts = ds_orig.file_meta.TransferSyntaxUID
    red_ts = ds_red.file_meta.TransferSyntaxUID
    if orig_ts != red_ts:
        problems.append(f"TransferSyntax changed: {orig_ts.name} -> {red_ts.name}")

    orig_frames = ds_orig.get("NumberOfFrames", 1)
    red_frames = ds_red.get("NumberOfFrames", 1)
    if int(orig_frames) != int(red_frames):
        problems.append(f"NumberOfFrames changed: {orig_frames} -> {red_frames}")

    try:
        rgb_red = decode_to_rgb(ds_red)
    except Exception as e:
        problems.append(f"redacted pixel_array/decode FAILED: {type(e).__name__}: {e}")
        return problems  # can't do pixel-level checks without a decode

    banner = banner_region(rgb_red, y0)
    max_banner = int(banner.max())
    if max_banner != 0:
        problems.append(
            f"banner region NOT fully black: max value={max_banner} "
            f"(this is the green-bar-style bug check -- redaction likely "
            f"happened in the wrong color space, or y0 is wrong)"
        )

    body = body_region(rgb_red, y0)
    if int(body.max()) == 0:
        problems.append(
            "body region (below banner) is entirely zero -- the whole "
            "image may have been blanked instead of just the banner"
        )

    return problems


def export_png_spotcheck(orig_path, redacted_path, out_dir, tag):
    ds_orig = pydicom.dcmread(orig_path)
    ds_red = pydicom.dcmread(redacted_path)
    rgb_orig = first_frame(decode_to_rgb(ds_orig))
    rgb_red = first_frame(decode_to_rgb(ds_red))

    side_by_side = np.concatenate([rgb_orig, np.full((rgb_orig.shape[0], 8, 3), 255, dtype=np.uint8), rgb_red], axis=1)
    out_path = os.path.join(out_dir, f"spotcheck_{tag}.png")
    Image.fromarray(side_by_side).save(out_path)
    return out_path


def verify_series(orig_paths, redacted_dir, y0, spotcheck_limit=2):
    """Verify every original/redacted pair, printing per-file OK/FAIL and an
    overall verdict. Exports up to `spotcheck_limit` before/after PNGs (into
    <redacted_dir>/spotchecks) as a representative visual sample -- not
    every file, to keep this cheap by default; raise the limit for broader
    review. Returns all_ok (bool)."""
    out_png_dir = os.path.join(redacted_dir, "spotchecks")
    os.makedirs(out_png_dir, exist_ok=True)

    all_ok = True
    spot_checked = 0
    for orig_path in orig_paths:
        base = os.path.splitext(os.path.basename(orig_path))[0]
        redacted_path = os.path.join(redacted_dir, f"{base}_redacted.dcm")
        if not os.path.exists(redacted_path):
            print(f"--- {base}: MISSING redacted output at {redacted_path}")
            all_ok = False
            continue

        problems = verify_pair(orig_path, redacted_path, y0)
        if problems:
            all_ok = False
            print(f"--- {base}: FAIL")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"--- {base}: OK (transfer syntax + frame count preserved, "
                  f"banner true-black, body intact)")

        if spot_checked < spotcheck_limit:
            png_path = export_png_spotcheck(orig_path, redacted_path, out_png_dir, base)
            print(f"    spot-check PNG: {png_path}")
            spot_checked += 1

    print()
    print("=== Overall:", "PASS" if all_ok else "FAIL", "===")
    return all_ok


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "raw"
    redacted_dir = sys.argv[2] if len(sys.argv) > 2 else "redacted"
    y0 = int(sys.argv[3]) if len(sys.argv) > 3 else None

    orig_paths = sorted(glob.glob(os.path.join(raw_dir, "*.dcm")))
    if not orig_paths:
        print(f"No .dcm files found in {raw_dir!r}")
        return 1

    if y0 is None:
        # recompute the same way redact.py does, for a self-contained check
        from redact import compute_series_cutoff
        y0, missing = compute_series_cutoff(orig_paths)
        print(f"(recomputed) series-wide cutoff y0={y0}")

    all_ok = verify_series(orig_paths, redacted_dir, y0, spotcheck_limit=2)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
