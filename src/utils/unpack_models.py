"""
scripts/unpack_models.py

Descomprime models_saved.tar.gz en models/saved/ y models/onnx/.
Verifica integridad con SHA256.

Uso:
  uv run scripts/unpack_models.py                        # archivo local
  uv run scripts/unpack_models.py --url "https://..."    # desde URL
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from urllib.request import urlretrieve

import gdown

from src.config.paths import paths

MODELS_DIR = paths.MODELS
ONNX_DIR = paths.MODELS / "onnx"
ARCHIVE_PATH = paths.MODELS_ROOT / "models_saved.tar.gz"


def download_file(url: str, dest: Path):
    """Descarga archivo desde URL con soporte para Google Drive."""
    print(f"⬇️  Descargando desde: {url}")

    if "drive.google.com" in url:
        if gdown is None:
            print("❌ Instala gdown: uv pip install gdown")
            sys.exit(1)
        print("☁️  Google Drive detectado, usando gdown...")
        out = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)
        if out is None:
            print("❌ Descarga fallida. Verifica que el enlace sea público.")
            sys.exit(1)
    else:

        def _progress(count, block_size, total_size):
            total = total_size if total_size > 0 else 1
            pct = count * block_size * 100 / total
            print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

        try:
            urlretrieve(url, str(dest), reporthook=_progress)
            print()
        except Exception as e:
            print(f"\n❌ Error descargando: {e}")
            sys.exit(1)

    print(f"  ✅ Descargado: {dest}")


def verify_file(filepath: Path, expected_hash: str) -> bool:
    """Verifica SHA256 de un archivo."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def unpack_models(archive_path: Path) -> bool:
    """Descomprime y verifica integridad."""
    if not archive_path.exists():
        print(f"❌ No se encontró: {archive_path}")
        print("  Opciones:")
        print("    1. Coloca models_saved.tar.gz en models/")
        print('    2. Usa: uv run scripts/unpack_models.py --url "https://..."')
        return False

    print(f"📦 Descomprimiendo: {archive_path}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Seguridad: rechazar paths absolutos o con ..
                if member.name.startswith("/") or ".." in member.name:
                    print(f"  ⚠️  Path sospechoso ignorado: {member.name}")
                    continue

                member_name = member.name

                # Determinar destino según prefijo
                if member_name.startswith("saved/onnx/"):
                    # → models/onnx/
                    rel = member_name[len("saved/onnx/") :]
                    if not rel:
                        continue
                    dest_path = ONNX_DIR / rel
                elif member_name.startswith("saved/"):
                    # → models/saved/
                    rel = member_name[len("saved/") :]
                    if not rel:
                        continue
                    dest_path = MODELS_DIR / rel
                else:
                    continue

                source = tar.extractfile(member)
                if source:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(source.read())
                    print(f"  📄 {member_name}")

    except tarfile.ReadError:
        print("❌ Archivo .tar.gz inválido o corrupto.")
        archive_path.unlink(missing_ok=True)
        return False

    # Verificar integridad
    manifest_path = MODELS_DIR / "manifest.json"
    if manifest_path.exists():
        print("\n🔍 Verificando integridad...")
        with open(manifest_path) as f:
            manifest = json.load(f)

        all_ok = True
        for arcname, info in manifest.get("files", {}).items():
            # Reconstruir path local desde arcname
            if arcname.startswith("saved/onnx/"):
                filepath = ONNX_DIR / arcname[len("saved/onnx/") :]
            elif arcname.startswith("saved/"):
                filepath = MODELS_DIR / arcname[len("saved/") :]
            else:
                continue

            if not filepath.exists():
                print(f"  ❌ FALTA: {arcname}")
                all_ok = False
                continue

            if verify_file(filepath, info["sha256"]):
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✅ {arcname:55s} {size_mb:.1f} MB")
            else:
                print(f"  ❌ {arcname:55s} HASH INCORRECTO")
                all_ok = False

        manifest_path.unlink()

        if all_ok:
            print("\n✅ Todos los archivos verificados correctamente")
        else:
            print("\n⚠️  Algunos archivos tienen problemas")
            return False
    else:
        print("\n⚠️  Sin manifiesto (no se puede verificar integridad)")

    # Resumen
    print("\n📋 Modelos disponibles:")
    for directory, label in [(MODELS_DIR, "PyTorch"), (ONNX_DIR, "ONNX")]:
        if not directory.exists():
            continue
        files = [
            f
            for f in sorted(directory.iterdir())
            if f.is_file() and f.suffix in (".pth", ".pkl", ".onnx", ".json", ".pt")
        ]
        if files:
            print(f"\n  [{label}]")
            for item in files:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.name:45s} {size_mb:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Descomprime modelos pre-entrenados")
    parser.add_argument("--url", type=str, default=None, help="URL de descarga")
    parser.add_argument("--archive", type=str, default=None, help="Ruta al .tar.gz")
    args = parser.parse_args()

    archive = Path(args.archive) if args.archive else ARCHIVE_PATH

    if args.url:
        archive.parent.mkdir(parents=True, exist_ok=True)
        download_file(args.url, archive)

    unpack_models(archive)


if __name__ == "__main__":
    main()
