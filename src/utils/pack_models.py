"""
Packs models/saved + models/onnx into a zip file to share.
Generates: models_saved.tar.gz

Usage:
  uv run scripts/pack_models.py            # includes ONNX if they exist
  uv run scripts/pack_models.py --no-onnx  # only .pth / .pkl / .json
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

from src.config.paths import paths

MODELS_DIR = paths.MODELS
ONNX_DIR = paths.MODELS / "onnx"
OUTPUT_DIR = paths.MODELS_ROOT


def get_file_hash(filepath: Path) -> str:
    """SHA256 of the file to validate integrity."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def pack_models(include_onnx: bool = True):
    """
    Compresses models/saved (and optionally models/onnx) into a tar.gz file.

    Structure within the tar:
      saved/best_classifier.pth
      saved/best_tissue_classifier.pth
      saved/best_segmenter.pth
      saved/tabular_model.pkl
      saved/ensemble_config.json
      saved/onnx/best_classifier.onnx       ← si include_onnx
      saved/onnx/best_classifier.json
      ...
    """
    if not MODELS_DIR.exists():
        print(f"❌ Does not exist: {MODELS_DIR}")
        return

    all_files: list[tuple[Path, str]] = []

    for pattern in ("*.pth", "*.pkl", "*.json"):
        for f in sorted(MODELS_DIR.glob(pattern)):
            all_files.append((f, f"saved/{f.name}"))

    if include_onnx and ONNX_DIR.exists():
        onnx_files = sorted(ONNX_DIR.glob("*.onnx")) + sorted(ONNX_DIR.glob("*.json"))
        for f in onnx_files:
            all_files.append((f, f"saved/onnx/{f.name}"))

    if not all_files:
        print("❌ No files to package")
        return

    manifest: dict = {
        "created": datetime.utcnow().isoformat(),
        "includes_onnx": include_onnx and ONNX_DIR.exists(),
        "files": {},
    }

    print("📦 Files to package:")
    total_size = 0.0
    for filepath, arcname in all_files:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        total_size += size_mb
        manifest["files"][arcname] = {
            "size_mb": round(size_mb, 2),
            "sha256": get_file_hash(filepath),
        }
        print(f"  {arcname:55s} {size_mb:8.2f} MB")

    print(f"  {'TOTAL':55s} {total_size:8.2f} MB")

    manifest_path = MODELS_DIR / "manifest.json"
    with open(str(manifest_path), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    output_path = OUTPUT_DIR / "models_saved.tar.gz"
    print(f"\n🔄 Compressing → {output_path}")

    with tarfile.open(output_path, "w:gz", compresslevel=6) as tar:
        for filepath, arcname in all_files:
            tar.add(filepath, arcname=arcname)
            print(f"  ✅ {arcname}")
        tar.add(manifest_path, arcname="saved/manifest.json")
        print("  ✅ saved/manifest.json")

    manifest_path.unlink()

    compressed_size = output_path.stat().st_size / (1024 * 1024)
    ratio = (1 - compressed_size / total_size) * 100 if total_size > 0 else 0

    print("\n✅ Packaging completed:")
    print(f"  Original:   {total_size:.1f} MB")
    print(f"  Compressed: {compressed_size:.1f} MB")
    print(f"  Reduction:  {ratio:.1f}%")
    print(f"  File:       {output_path}")
    print("\n📤 Upload this file to Google Drive / OneDrive and share the link")


def main():
    parser = argparse.ArgumentParser(description="Packs trained models")
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Exclude ONNX models from the package",
    )
    args = parser.parse_args()
    pack_models(include_onnx=not args.no_onnx)


if __name__ == "__main__":
    main()
