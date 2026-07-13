#!/usr/bin/env python3
"""
Single-command driver for the DICOM banner-redaction pipeline: probe ->
redact -> verify -> ocr_verify, run in one process.

This complements -- does not replace -- the standalone scripts and the
staged rollout runbook documented in README.md. For a first run against
real PHI, prefer the staged steps (probe, then a small-subset dry run,
then a full batch, then manual visual review); this command is the
convenience one-shot for repeat/known-good runs.

Run from the repo root: `python pipeline.py -i raw -o redacted`
(relies on probe.py/redact.py/verify.py/ocr_verify.py being importable
siblings, same as the existing lazy `from redact import ...` calls in
verify.py/ocr_verify.py).

IMPORTANT: `ocr_verify` is imported lazily, inside run(), only if the OCR
stage will actually execute (i.e. --skip-ocr was not passed). It's the only
module in this chain that pulls in dicom_phi_scan -> easyocr -> torch (a
GB-scale dependency, and EasyOCR downloads model weights on first use) --
so --skip-ocr runs never touch torch at all.
"""
import argparse
import glob
import os
import sys

import probe
import redact
import verify
# ocr_verify is NOT imported here -- see the module docstring.


def build_parser():
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Run the full DICOM banner-redaction pipeline "
                    "(probe -> redact -> verify -> OCR-verify) in one command.",
    )
    p.add_argument("-i", "--input", default="raw",
                    help="Directory of source .dcm files (default: raw)")
    p.add_argument("-o", "--output", default="redacted",
                    help="Directory for redacted output + spotchecks (default: redacted)")
    p.add_argument("--cutoff", "--y0", dest="cutoff", type=int, default=None,
                    help="Manual banner cutoff row y0. Overrides the value computed "
                         "from SequenceOfUltrasoundRegions, and is also used as the "
                         "fallback for files that lack the region tag. If omitted, "
                         "y0 is computed once as the series-wide min and threaded "
                         "through every stage.")
    p.add_argument("--spotcheck-limit", type=int, default=2,
                    help="How many before/after PNG spot-checks the verify stage "
                         "writes (default: 2; raise for broader visual review)")
    p.add_argument("--skip-probe", action="store_true",
                    help="Skip the read-only recon stage")
    p.add_argument("--skip-verify", action="store_true",
                    help="Skip the pixel-level verification stage")
    p.add_argument("--skip-ocr", action="store_true",
                    help="Skip the OCR-verify stage (avoids importing torch/easyocr entirely)")
    p.add_argument("--fail-fast", action="store_true",
                    help="Abort the redact stage on the first file error "
                         "(default: continue-on-error, report failures at the end)")
    return p


def summarize_and_exit(results, redact_errors, verify_ok, ocr_ok, ocr_flags, y0):
    print("\n=== Pipeline summary ===")
    print(f"cutoff y0 = {y0}")
    print(f"redacted OK: {len(results)}   redact errors: {len(redact_errors)}")
    for path, e in redact_errors:
        print(f"  - {path}: {type(e).__name__}: {e}")
    if verify_ok is not None:
        print(f"verify:      {'PASS' if verify_ok else 'FAIL'}")
    else:
        print("verify:      skipped")
    if ocr_ok is not None:
        print(f"ocr_verify:  {'PASS' if ocr_ok else 'FAIL'}  "
              f"(below_cutoff={ocr_flags['any_below_cutoff']}, "
              f"banner_text={ocr_flags['any_banner_text']}, "
              f"missing={ocr_flags['any_missing']})")
    else:
        print("ocr_verify:  skipped")

    failed = bool(redact_errors) or (verify_ok is False) or (ocr_ok is False)
    print("=== OVERALL:", "FAIL" if failed else "PASS", "===")
    return 1 if failed else 0


def run(args):
    paths = sorted(glob.glob(os.path.join(args.input, "*.dcm")))
    if not paths:
        print(f"No .dcm files found in {args.input!r}")
        return 1
    os.makedirs(args.output, exist_ok=True)

    # --- Stage 1: probe (non-gating recon) ---
    if not args.skip_probe:
        print("=== Stage 1/4: probe ===")
        probe.run_probe(paths)

    # --- Cutoff: computed ONCE, threaded through every later stage ---
    if args.cutoff is not None:
        y0, missing = args.cutoff, []
        print(f"Using manual cutoff y0={y0}")
    else:
        try:
            y0, missing = redact.compute_series_cutoff(paths, fallback=None)
        except ValueError as e:
            print(f"ERROR computing cutoff: {e}\n"
                  f"Pass --cutoff/--y0 to supply one explicitly.")
            return 1
        print(f"Series-wide redaction cutoff: y0={y0} (rows [0:{y0}) will be zeroed)")
    if missing:
        print(f"WARNING: {len(missing)} file(s) lacked SequenceOfUltrasoundRegions; "
              f"applying the series cutoff to them: {missing}")

    # --- Stage 2: redact (per-file resilient by default) ---
    print("=== Stage 2/4: redact ===")
    if args.fail_fast:
        try:
            results, redact_errors = redact.redact_series(
                paths, y0, args.output, continue_on_error=False)
        except Exception as e:
            print(f"ABORTED (--fail-fast): {type(e).__name__}: {e}")
            return 1
    else:
        results, redact_errors = redact.redact_series(
            paths, y0, args.output, continue_on_error=True)

    # --- Stage 3: verify ---
    verify_ok = None
    if not args.skip_verify:
        print("=== Stage 3/4: verify ===")
        verify_ok = verify.verify_series(
            paths, args.output, y0, spotcheck_limit=args.spotcheck_limit)

    # --- Stage 4: OCR-verify (lazy import so --skip-ocr never touches torch) ---
    ocr_ok, ocr_flags = None, None
    if not args.skip_ocr:
        print("=== Stage 4/4: ocr_verify ===")
        import ocr_verify  # deferred: pulls in dicom_phi_scan -> easyocr -> torch
        ocr_ok, ocr_flags = ocr_verify.ocr_verify_series(paths, args.output, y0)

    return summarize_and_exit(results, redact_errors, verify_ok, ocr_ok, ocr_flags, y0)


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
