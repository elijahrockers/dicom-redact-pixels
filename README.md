# dicom-redact-pixels

Redacts burned-in PHI in the top banner/header of Canon ultrasound DICOM
files by zeroing those pixels to black, while leaving the live ultrasound
image and DICOM header tags untouched.

Built for a 30-file series: 29 single-frame `(1,960,1280,3)` RGB files
(uncompressed) + 1 multi-frame `(248,960,1280,3)` YBR_FULL_422 cine file
(JPEG Baseline compressed). The pipeline generalizes to any series exposing
`SequenceOfUltrasoundRegions`.

## How it works

1. **Cutoff detection** (`redact.py: compute_series_cutoff`): the banner/live-image
   boundary is read from `SequenceOfUltrasoundRegions` (0018,6011) →
   `RegionLocationMinY0` (0018,601A) — the topmost edge of the scanned-image
   region. One series-wide cutoff (the min across all files) is used so every
   file gets a consistent boundary.
2. **Redaction** (`redact.py: redact_pixels`): rows `[0:y0)` are zeroed in
   **RGB space**, never in raw YBR. YBR "black" is `Y=0,Cb=128,Cr=128` —
   zeroing raw YBR values produces a *green* bar, not black.
3. **Output preserves each input's transfer syntax**: uncompressed RGB files
   are written back losslessly; the JPEG-Baseline YBR_FULL_422 file is
   re-encoded via Pillow's JPEG encoder + `pydicom.encaps.encapsulate()`.
4. **Verification**: `verify.py` reloads every redacted file and asserts the
   banner decodes to true black and the image body is untouched.
   `ocr_verify.py` adds an OCR-based check on top — using
   [`dicom-phi-scan`](https://github.com/elijahrockers/dicom-phi-scan)'s
   EasyOCR scanner purely to **verify and flag**, never to redact:
   - pre-scans originals and flags any detected text extending at/below the
     banner cutoff (PHI the region-based blackout can't reach, e.g. a
     corner/overlay label burned onto the image itself)
   - post-scans redacted output and asserts no OCR finding remains in the
     banner region

### Key gotchas (learned the hard way — see inline comments for detail)

- **pydicom 3.x auto-converts YBR_FULL_422 → RGB on read.** Calling
  `convert_color_space` again on `ds.pixel_array` output double-converts and
  corrupts the image (observed as a green tint). Conversion is only needed
  on the *write* side, when manually building YBR JPEG bytes.
- **Avoid GDCM's raw `Image`/`ImageChangeTransferSyntax` Python API** for
  JPEG re-encoding — it SIGABRT-crashed the interpreter during testing (not
  a catchable Python exception). Pillow's JPEG encoder + `encapsulate()` is
  used exclusively instead.
- If a file lacks `SequenceOfUltrasoundRegions`, `redact.py` raises rather
  than guessing (no silent fallback cutoff is assumed).
- If a file uses a transfer syntax other than uncompressed or JPEG Baseline
  (e.g. JPEG2000, RLE), `save_same_encoding()` raises `NotImplementedError`
  rather than mishandling it.

## Repo layout

```
pipeline.py             single-command driver: probe + redact + verify + ocr_verify
probe.py               read-only header/region inspection
redact.py               cutoff computation + redaction + batch driver
verify.py               post-redaction pixel-level checks + before/after PNGs
ocr_verify.py           OCR-based pre-scan (flag) + post-scan (assert) via dicom-phi-scan
scripts/make_test_data.py   synthetic prototype DICOM generator (for dev/testing only)
requirements.txt        pinned dependencies
```

`raw/` (originals) and `redacted/` (output) are gitignored — no DICOM files,
real or synthetic, are ever committed to this repo.

## Installation

```bash
git clone https://github.com/elijahrockers/dicom-redact-pixels.git
cd dicom-redact-pixels
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Two things worth doing right after install, before touching any real PHI:

- **Check torch's build.** `requirements.txt` pins whatever build was used
  during development (may be a CUDA build). If the target machine has no
  GPU, install the CPU wheel instead to avoid pulling several GB of unused
  CUDA libraries:
  ```bash
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
  ```
- **Pre-warm EasyOCR's model download.** It fetches ~64MB of detection/
  recognition weights on first use. Do this deliberately now (not mid-run):
  ```bash
  .venv/bin/python -c "import easyocr; easyocr.Reader(['en'])"
  ```

Sanity-check the install:
```bash
.venv/bin/python -c "
import pydicom, numpy, gdcm, PIL, easyocr, torch
print(pydicom.__version__, numpy.__version__, gdcm.Version.GetVersion(), PIL.__version__, torch.__version__)
"
```

## Single-command pipeline

`pipeline.py` runs all four stages (probe → redact → verify → ocr_verify) in
one process, in-place of running the four scripts manually:

```bash
.venv/bin/python pipeline.py -i raw -o redacted
```

| Flag | Default | Purpose |
|---|---|---|
| `-i, --input` | `raw` | Directory of source `.dcm` files |
| `-o, --output` | `redacted` | Directory for redacted output + spotchecks |
| `--cutoff, --y0` | auto-detect | Manual banner cutoff row override; also used as the fallback for files missing `SequenceOfUltrasoundRegions`. If omitted, computed once (series-wide min) and threaded through every stage |
| `--spotcheck-limit` | `2` | How many before/after PNGs `verify` exports — raise for broader visual review |
| `--skip-probe` | off | Skip the read-only recon stage |
| `--skip-verify` | off | Skip the pixel-level verification stage |
| `--skip-ocr` | off | Skip the OCR-verify stage — **avoids importing torch/easyocr entirely**, since that import is deferred until this stage actually runs |
| `--fail-fast` | off | Abort the redact stage on the first file error (default: continue past errors, redact the rest, and report failures at the end) |

**Exit-code contract:** `probe` never gates (informational only); `redact`
gates on any per-file error; `verify` gates on FAIL; `ocr_verify` gates on
FAIL — which, matching its standalone behavior, **includes** a pre-scan
"PHI text extends below the banner cutoff" warning, not just a survived
banner. Overall exit is `0` only if every stage that ran, passed. The final
printed summary breaks out exactly which condition failed.

For a first run against real PHI, prefer the staged steps below — running
each stage by hand on a small subset first has real safety value.
`pipeline.py` is the convenience one-shot for repeat/known-good runs.

## Running against a real series (staged rollout)

Originals are never modified — every script only reads `raw/` and writes to
a separate `redacted/`. Still, work on copies and keep a separate backup of
the real files until you trust the output.

**Step A — read-only recon (zero risk, do this first):**
```bash
mkdir -p raw redacted
cp -a /path/to/real/export/*.dcm raw/
sha256sum /path/to/real/export/*.dcm > /tmp/source.sha256
(cd raw && sha256sum *.dcm) > /tmp/copy.sha256   # confirm byte-identical copies
chmod -R a-w raw/                                 # optional safety net
.venv/bin/python probe.py raw
```
Confirm `probe.py` reports the expected photometric interpretations,
transfer syntaxes, region tags present, and that all files decode. If
anything differs from what you expect, stop and investigate before
redacting anything — this step touches nothing.

**Step B — small-subset dry run:** before running the full series, copy a
handful of representative files (including the trickiest one — any
JPEG/compressed multi-frame file) into a separate directory and run the
full chain against just those:
```bash
mkdir raw_test redacted_test
cp raw/<pick-a-few>.dcm raw_test/
.venv/bin/python redact.py raw_test redacted_test
.venv/bin/python verify.py raw_test redacted_test
.venv/bin/python ocr_verify.py raw_test redacted_test
```

**Step C — full batch**, once Step B looks right:
```bash
.venv/bin/python redact.py raw redacted
.venv/bin/python verify.py raw redacted       # must end "Overall: PASS"
.venv/bin/python ocr_verify.py raw redacted   # review WARNs, must end "Overall: PASS"
```
Read `ocr_verify.py`'s pre-scan output carefully — any WARN about text
extending below the banner cutoff means the region-based redaction won't
remove it, and needs separate handling.

**Step D — manual visual review:** spot-check a broad sample of
`redacted/*.dcm` in a real DICOM viewer (or inspect `verify.py`'s exported
PNGs) before considering the batch done. Automated checks reduce risk but
don't replace a final human look.
