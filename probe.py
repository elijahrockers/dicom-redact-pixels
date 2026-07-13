"""
Read-only reconnaissance over a directory of DICOM files.

For every .dcm file, prints the technical tags needed to plan pixel
redaction of the top banner: transfer syntax, photometric interpretation,
pixel array shape, and SequenceOfUltrasoundRegions (0018,6011) region
boundaries -- the region tag's topmost RegionLocationMinY0 tells us where
the "live" ultrasound image starts (i.e. everything above it is chrome/banner
that can be blacked out).

Does not modify any file. No PHI values are printed -- only tag presence
and technical/geometric values.

Usage: python probe.py [directory]   (default: ./raw)
"""
import sys
import os
import glob
import pydicom


def describe_regions(ds):
    if "SequenceOfUltrasoundRegions" not in ds:
        return None
    regions = []
    for item in ds.SequenceOfUltrasoundRegions:
        regions.append({
            "MinX0": item.get("RegionLocationMinX0"),
            "MinY0": item.get("RegionLocationMinY0"),
            "MaxX1": item.get("RegionLocationMaxX1"),
            "MaxY1": item.get("RegionLocationMaxY1"),
            "SpatialFormat": item.get("RegionSpatialFormat"),
            "DataType": item.get("RegionDataType"),
        })
    return regions


def probe_file(path):
    ds = pydicom.dcmread(path)
    ts = ds.file_meta.TransferSyntaxUID if "TransferSyntaxUID" in ds.file_meta else None

    info = {
        "path": path,
        "TransferSyntaxUID": str(ts) if ts else None,
        "TransferSyntaxName": ts.name if ts else None,
        "TransferSyntaxCompressed": ts.is_compressed if ts else None,
        "PhotometricInterpretation": ds.get("PhotometricInterpretation"),
        "SamplesPerPixel": ds.get("SamplesPerPixel"),
        "PlanarConfiguration": ds.get("PlanarConfiguration"),
        "BitsAllocated": ds.get("BitsAllocated"),
        "BitsStored": ds.get("BitsStored"),
        "Rows": ds.get("Rows"),
        "Columns": ds.get("Columns"),
        "NumberOfFrames": ds.get("NumberOfFrames"),
        "Manufacturer": ds.get("Manufacturer"),
        "ManufacturerModelName": ds.get("ManufacturerModelName"),
        "BurnedInAnnotation": ds.get("BurnedInAnnotation"),
        "Regions": describe_regions(ds),
    }

    try:
        arr = ds.pixel_array
        info["decode_ok"] = True
        info["array_shape"] = arr.shape
        info["array_dtype"] = str(arr.dtype)
    except Exception as e:
        info["decode_ok"] = False
        info["decode_error"] = f"{type(e).__name__}: {e}"

    return info


def print_info(info):
    print(f"--- {info['path']}")
    print(f"  TransferSyntax: {info['TransferSyntaxName']} ({info['TransferSyntaxUID']}) "
          f"compressed={info['TransferSyntaxCompressed']}")
    print(f"  PhotometricInterpretation: {info['PhotometricInterpretation']}")
    print(f"  SamplesPerPixel={info['SamplesPerPixel']} PlanarConfiguration={info['PlanarConfiguration']} "
          f"BitsAllocated={info['BitsAllocated']} BitsStored={info['BitsStored']}")
    print(f"  Rows={info['Rows']} Columns={info['Columns']} NumberOfFrames={info['NumberOfFrames']}")
    print(f"  Manufacturer={info['Manufacturer']!r} Model={info['ManufacturerModelName']!r}")
    print(f"  BurnedInAnnotation={info['BurnedInAnnotation']}")
    if info["Regions"]:
        for i, r in enumerate(info["Regions"]):
            print(f"  Region[{i}]: X0={r['MinX0']} Y0={r['MinY0']} X1={r['MaxX1']} Y1={r['MaxY1']} "
                  f"SpatialFormat={r['SpatialFormat']} DataType={r['DataType']}")
    else:
        print("  Regions: none (SequenceOfUltrasoundRegions absent)")
    if info["decode_ok"]:
        print(f"  pixel_array OK: shape={info['array_shape']} dtype={info['array_dtype']}")
    else:
        print(f"  pixel_array FAILED: {info['decode_error']}")
    print()


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "raw"
    paths = sorted(glob.glob(os.path.join(directory, "*.dcm")))
    if not paths:
        print(f"No .dcm files found in {directory!r}")
        return 1

    all_info = []
    for path in paths:
        try:
            info = probe_file(path)
        except Exception as e:
            print(f"--- {path}\n  FAILED TO READ: {type(e).__name__}: {e}\n")
            continue
        print_info(info)
        all_info.append(info)

    # Summary
    n_rgb = sum(1 for i in all_info if i["PhotometricInterpretation"] == "RGB")
    n_ybr = sum(1 for i in all_info if str(i["PhotometricInterpretation"]).startswith("YBR"))
    n_with_regions = sum(1 for i in all_info if i["Regions"])
    n_decode_fail = sum(1 for i in all_info if not i["decode_ok"])
    min_y0_values = [
        min(r["MinY0"] for r in i["Regions"])
        for i in all_info if i["Regions"]
    ]
    print("=== Summary ===")
    print(f"Total files: {len(all_info)}")
    print(f"RGB: {n_rgb}  YBR*: {n_ybr}  other: {len(all_info) - n_rgb - n_ybr}")
    print(f"Files with SequenceOfUltrasoundRegions: {n_with_regions}/{len(all_info)}")
    print(f"Decode failures: {n_decode_fail}")
    if min_y0_values:
        print(f"Per-file topmost RegionLocationMinY0 values: {min_y0_values}")
        print(f"Series-wide cutoff candidate (min): {min(min_y0_values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
