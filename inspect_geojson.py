"""
Step 1: Memory-efficient GeoJSON schema inspector
Supports both fiona (fast) and ijson (fallback) backends.
"""

import sys
import json
import os
from pathlib import Path

GEOJSON_PATH = r"C:\AI_RecycleBin\{A1BC5211-A7CC-498C-9C4A-1D655B883451}\1\Downloads\BGKY_Contours_2FT_2004_Depression.geojson"

def inspect_with_fiona():
    import fiona
    print(f"\n{'='*60}")
    print("INSPECTING WITH FIONA (streaming, memory-safe)")
    print(f"{'='*60}")

    with fiona.open(GEOJSON_PATH) as src:
        print(f"\n[CRS]")
        print(f"  {src.crs}")

        print(f"\n[Driver / Format]")
        print(f"  {src.driver}")

        print(f"\n[Geometry Type]")
        print(f"  {src.schema['geometry']}")

        print(f"\n[Attribute Fields]")
        for field, dtype in src.schema['properties'].items():
            print(f"  {field:30s}  {dtype}")

        print(f"\n[Bounding Box] (minx, miny, maxx, maxy)")
        print(f"  {src.bounds}")

        print(f"\n[Feature Count]")
        count = len(src)
        print(f"  {count:,}")

        # Sample first 5 features for value ranges
        print(f"\n[Sample Properties from first 5 features]")
        for i, feat in enumerate(src):
            if i >= 5:
                break
            print(f"  Feature {i}: {dict(feat['properties'])}")

    # Walk all features to find min/max of numeric fields (reopen to reset cursor)
    print(f"\n[Scanning all features for numeric field ranges...]")
    print("  (this may take a minute for 660 MB)")
    with fiona.open(GEOJSON_PATH) as src2:
        numeric_stats = {}
        geom_type_counts = {}
        total = 0

        for feat in src2:
            total += 1
            gtype = feat['geometry']['type'] if feat['geometry'] else 'Null'
            geom_type_counts[gtype] = geom_type_counts.get(gtype, 0) + 1

            for k, v in feat['properties'].items():
                if isinstance(v, (int, float)) and v is not None:
                    if k not in numeric_stats:
                        numeric_stats[k] = {'min': v, 'max': v, 'count': 0}
                    if v < numeric_stats[k]['min']:
                        numeric_stats[k]['min'] = v
                    if v > numeric_stats[k]['max']:
                        numeric_stats[k]['max'] = v
                    numeric_stats[k]['count'] += 1

            if total % 50000 == 0:
                print(f"  ...scanned {total:,} features")

    print(f"\n[Geometry Type Breakdown]")
    for gtype, cnt in geom_type_counts.items():
        print(f"  {gtype}: {cnt:,}")

    print(f"\n[Numeric Field Ranges]")
    for field, stats in numeric_stats.items():
        print(f"  {field:30s}  min={stats['min']:.4f}  max={stats['max']:.4f}  non-null={stats['count']:,}")

    print(f"\n[Total features scanned]: {total:,}")


def inspect_with_ijson():
    """Fallback: pure streaming with ijson — no geopandas/fiona needed."""
    import ijson

    print(f"\n{'='*60}")
    print("INSPECTING WITH IJSON (streaming, no fiona required)")
    print(f"{'='*60}")

    file_size_mb = os.path.getsize(GEOJSON_PATH) / 1_048_576
    print(f"\nFile size: {file_size_mb:.1f} MB")

    fields_seen = {}
    geom_types = {}
    numeric_stats = {}
    bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    count = 0
    sample_props = []

    with open(GEOJSON_PATH, 'rb') as f:
        # Check top-level CRS
        # ijson prefix 'crs' to pull it out
        pass

    # Stream features
    with open(GEOJSON_PATH, 'rb') as f:
        for feat in ijson.items(f, 'features.item'):
            count += 1

            # Geometry type
            geom = feat.get('geometry') or {}
            gtype = geom.get('type', 'Null')
            geom_types[gtype] = geom_types.get(gtype, 0) + 1

            # Bounding box from coordinates (first coordinate of each feature)
            coords = geom.get('coordinates')
            if coords:
                try:
                    # Flatten to get first point
                    def first_point(c):
                        if isinstance(c[0], (int, float)):
                            return c
                        return first_point(c[0])
                    pt = first_point(coords)
                    if len(pt) >= 2:
                        bbox[0] = min(bbox[0], pt[0])
                        bbox[1] = min(bbox[1], pt[1])
                        bbox[2] = max(bbox[2], pt[0])
                        bbox[3] = max(bbox[3], pt[1])
                except Exception:
                    pass

            # Properties
            props = feat.get('properties') or {}
            for k, v in props.items():
                fields_seen[k] = type(v).__name__
                if isinstance(v, (int, float)):
                    if k not in numeric_stats:
                        numeric_stats[k] = {'min': v, 'max': v}
                    else:
                        if v < numeric_stats[k]['min']:
                            numeric_stats[k]['min'] = v
                        if v > numeric_stats[k]['max']:
                            numeric_stats[k]['max'] = v

            if count <= 5:
                sample_props.append(props)

            if count % 50000 == 0:
                print(f"  ...{count:,} features scanned")

    print(f"\n[CRS]")
    print("  Not easily extractable via ijson stream — check top of file manually")
    print("  Tip: head -c 2000 of file to read CRS block")

    print(f"\n[Feature Count]")
    print(f"  {count:,}")

    print(f"\n[Geometry Types]")
    for gtype, cnt in geom_types.items():
        print(f"  {gtype}: {cnt:,}")

    print(f"\n[Approximate Bounding Box] (minx, miny, maxx, maxy)")
    print(f"  {bbox}")

    print(f"\n[Attribute Fields & Types]")
    for k, dtype in fields_seen.items():
        print(f"  {k:30s}  {dtype}")

    print(f"\n[Numeric Field Ranges]")
    for field, stats in numeric_stats.items():
        print(f"  {field:30s}  min={stats['min']}  max={stats['max']}")

    print(f"\n[Sample Properties (first 5 features)]")
    for i, p in enumerate(sample_props):
        print(f"  Feature {i}: {p}")


def peek_crs():
    """Read the first 4 KB to extract CRS/type from the GeoJSON header."""
    print(f"\n[Quick Header Peek — CRS and Type Block]")
    with open(GEOJSON_PATH, 'r', encoding='utf-8', errors='replace') as f:
        header = f.read(4096)
    # Print up to 1500 chars so we can see crs + first feature
    print(header[:1500])


if __name__ == '__main__':
    print(f"Target file: {GEOJSON_PATH}")
    if not Path(GEOJSON_PATH).exists():
        print("ERROR: File not found. Check the path and try again.")
        sys.exit(1)

    file_size_mb = Path(GEOJSON_PATH).stat().st_size / 1_048_576
    print(f"File size: {file_size_mb:.1f} MB")

    peek_crs()

    # Try fiona first (fastest), fall back to ijson
    try:
        import fiona
        inspect_with_fiona()
    except ImportError:
        print("\nfiona not found — trying ijson...")
        try:
            import ijson
            inspect_with_ijson()
        except ImportError:
            print("\nNeither fiona nor ijson is installed.")
            print("Install one with:")
            print("  pip install fiona")
            print("  -- or --")
            print("  pip install ijson")
            sys.exit(1)
