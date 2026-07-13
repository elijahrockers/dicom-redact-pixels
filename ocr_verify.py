"""
OCR-based verification layer, using the `dicom-phi-scan` project
(https://github.com/elijahrockers/dicom-phi-scan)'s EasyOCR-backed
`scan_pixels()` to VERIFY redaction and FLAG residual PHI text.

This script does NOT redact anything and does NOT drive redact.py's
behavior -- the deterministic, region-based banner blackout in redact.py
remains the sole redactor. OCR recall is imperfect (low-contrast/stylized
fonts, rotation, etc.), so a clean scan here means "no text was detected",
not "no text exists" -- see the Caveats section of the plan. What OCR adds
that pixel-level checks (verify.py) can't:

  1. Pre-scan of ORIGINALS: flags any detected PHI text whose bounding box
     extends AT OR BELOW the banner cutoff y0 -- i.e. text the region-based
     blackout will NOT remove (e.g. a corner/overlay label burned directly
     onto the ultrasound image itself, outside the chrome banner).
  2. Post-scan of REDACTED output: asserts no OCR finding intersects the
     banner region [0:y0) -- a stronger guarantee than "pixels are black"
     (verify.py), since it confirms nothing *readable* survives, and would
     catch e.g. a wrong y0 or JPEG artifacts leaving faint-but-OCRable text.

Usage: python ocr_verify.py [raw_dir] [redacted_dir] [y0]
Exit status: 0 if no below-cutoff PHI was found pre-scan AND no banner text
survived post-scan; 1 otherwise.
"""
import sys
import os
import glob

import pydicom
from dicom_phi_scan.pixel_scanner import scan_pixels


def finding_span(finding):
    """(top, bottom) row span of an OCR finding's bounding box."""
    top = finding.bbox.y
    bottom = finding.bbox.y + finding.bbox.height
    return top, bottom


def prescan_file(path, y0):
    """Scan an original file; return (findings, below_cutoff_findings).
    A finding counts as 'below cutoff' if any part of its box is at or
    below y0 (top >= y0) -- i.e. text the banner blackout won't reach.
    A finding straddling the boundary (top < y0 < bottom) is also flagged,
    since only the portion above y0 would be removed."""
    ds = pydicom.dcmread(path)
    findings = scan_pixels(ds)
    below_cutoff = []
    for f in findings:
        top, bottom = finding_span(f)
        if bottom > y0:  # any part of the box reaches into/past the live image
            below_cutoff.append(f)
    return findings, below_cutoff


def postscan_file(path, y0):
    """Scan a redacted file; return (findings, banner_findings) where
    banner_findings is any finding whose box overlaps [0:y0) at all --
    these should be empty after correct redaction."""
    ds = pydicom.dcmread(path)
    findings = scan_pixels(ds)
    banner_findings = [f for f in findings if finding_span(f)[0] < y0]
    return findings, banner_findings


def ocr_verify_series(orig_paths, redacted_dir, y0):
    """Pre-scan originals (flag PHI text extending at/below the banner
    cutoff) and post-scan redacted output (assert the banner is text-free).
    Prints both sections + an overall verdict.

    Returns (ok, flags) where flags is a dict with keys
    'any_below_cutoff', 'any_banner_text', 'any_missing' (all bool), and
    ok = not (any_banner_text or any_missing or any_below_cutoff) --
    i.e. a pre-scan below-cutoff WARNING fails the run, same as a survived
    banner text or a missing redacted file. This is a deliberately strict,
    PHI-safe default."""
    print(f"\n=== Pre-scan: originals (flagging PHI text the banner cutoff [0:{y0}) won't remove) ===\n")
    any_below_cutoff = False
    for path in orig_paths:
        base = os.path.basename(path)
        findings, below_cutoff = prescan_file(path, y0)
        print(f"--- {base}: {len(findings)} text region(s) detected")
        for f in findings:
            top, bottom = finding_span(f)
            flag = "  <-- WARN: extends to/below banner cutoff, will NOT be redacted" if bottom > y0 else ""
            print(f"    {f.text!r} conf={f.confidence:.2f} box=(x={f.bbox.x},y={f.bbox.y},"
                  f"w={f.bbox.width},h={f.bbox.height}) rows=[{top}:{bottom}){flag}")
        if below_cutoff:
            any_below_cutoff = True

    print(f"\n=== Post-scan: {redacted_dir} (asserting banner [0:{y0}) is text-free after redaction) ===\n")
    any_banner_text = False
    any_missing = False
    for path in orig_paths:
        base = os.path.splitext(os.path.basename(path))[0]
        redacted_path = os.path.join(redacted_dir, f"{base}_redacted.dcm")
        if not os.path.exists(redacted_path):
            print(f"--- {base}: MISSING redacted output at {redacted_path}")
            any_missing = True
            continue

        findings, banner_findings = postscan_file(redacted_path, y0)
        if banner_findings:
            any_banner_text = True
            print(f"--- {base}: FAIL -- {len(banner_findings)} text region(s) still detected in banner")
            for f in banner_findings:
                top, bottom = finding_span(f)
                print(f"    {f.text!r} conf={f.confidence:.2f} rows=[{top}:{bottom})")
        else:
            extra = f" ({len(findings)} finding(s) elsewhere, outside banner)" if findings else ""
            print(f"--- {base}: OK -- banner text-free{extra}")

    print()
    if any_below_cutoff:
        print("WARNING: one or more original files have PHI text that extends below the "
              "banner cutoff and will NOT be removed by region-based redaction alone.")
    if any_banner_text:
        print("FAIL: one or more redacted files still have OCR-detectable text in the banner region.")
    if any_missing:
        print("FAIL: one or more redacted outputs are missing.")

    ok = not (any_banner_text or any_missing or any_below_cutoff)
    print("\n=== Overall:", "PASS" if ok else "FAIL", "===")
    flags = {
        "any_below_cutoff": any_below_cutoff,
        "any_banner_text": any_banner_text,
        "any_missing": any_missing,
    }
    return ok, flags


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "raw"
    redacted_dir = sys.argv[2] if len(sys.argv) > 2 else "redacted"
    y0 = int(sys.argv[3]) if len(sys.argv) > 3 else None

    orig_paths = sorted(glob.glob(os.path.join(raw_dir, "*.dcm")))
    if not orig_paths:
        print(f"No .dcm files found in {raw_dir!r}")
        return 1

    if y0 is None:
        from redact import compute_series_cutoff
        y0, missing = compute_series_cutoff(orig_paths)
        print(f"(recomputed) series-wide cutoff y0={y0}")

    ok, _flags = ocr_verify_series(orig_paths, redacted_dir, y0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
