"""
scripts/pack_models.py

Empaqueta models/saved en un archivo comprimido para compartir.
Genera: models_saved.tar.gz (~30-40% más pequeño que los .pth sueltos)
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile

from src.config.paths import paths

MODELS_DIR = paths.MODELS
OUTPUT_DIR = paths.MODELS_ROOT


def get_file_hash(filepath: Path) -> str:
    """SHA256 del archivo para verificar integridad."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def pack_models():
    """Comprime models/saved en un tar.gz con manifiesto de integridad."""

    if not MODELS_DIR.exists():
        print(f"❌ No existe: {MODELS_DIR}")
        return

    # ── Archivos a empaquetar ──
    model_files = list(MODELS_DIR.glob("*.pth")) + list(MODELS_DIR.glob("*.pkl"))
    json_files = list(MODELS_DIR.glob("*.json"))
    all_files = model_files + json_files

    if not all_files:
        print("❌ No hay modelos en models/saved")
        return

    # ── Generar manifiesto ──
    manifest = {
        "created": datetime.utcnow().isoformat(),
        "files": {},
    }

    print("📦 Archivos a empaquetar:")
    total_size = 0
    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        file_hash = get_file_hash(f)
        manifest["files"][f.name] = {
            "size_mb": round(size_mb, 2),
            "sha256": file_hash,
        }
        print(f"  {f.name:40s} {size_mb:8.2f} MB")

    print(f"  {'TOTAL':40s} {total_size:8.2f} MB")

    # ── Guardar manifiesto temporalmente ──
    manifest_path = MODELS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Comprimir ──
    output_path = OUTPUT_DIR / "models_saved.tar.gz"
    print(f"\n🔄 Comprimiendo → {output_path}")

    with tarfile.open(output_path, "w:gz", compresslevel=6) as tar:
        for filepath in all_files + [manifest_path]:
            arcname = f"saved/{filepath.name}"
            tar.add(filepath, arcname=arcname)
            print(f"  ✅ {filepath.name}")

    # Limpiar manifiesto temporal
    manifest_path.unlink()

    compressed_size = output_path.stat().st_size / (1024 * 1024)
    ratio = (1 - compressed_size / total_size) * 100

    print("\n✅ Empaquetado completado:")
    print(f"  Original:   {total_size:.1f} MB")
    print(f"  Comprimido: {compressed_size:.1f} MB")
    print(f"  Reducción:  {ratio:.1f}%")
    print(f"  Archivo:    {output_path}")
    print("\n📤 Sube este archivo a Google Drive / OneDrive y comparte el enlace")


if __name__ == "__main__":
    pack_models()
