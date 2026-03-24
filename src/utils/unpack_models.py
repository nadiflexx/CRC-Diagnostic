"""
scripts/unpack_models.py

Descomprime models_saved.tar.gz en models/saved/.
Verifica integridad con SHA256.

Uso:
  uv run src/utils/unpack_models.py                          # desde archivo local
  uv run src/utils/unpack_models.py --url "https://..."      # desde URL (Drive, etc.)
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
ARCHIVE_PATH = paths.MODELS_ROOT / "models_saved.tar.gz"


def download_file(url: str, dest: Path):
    """Descarga archivo desde URL con soporte especial para Google Drive."""
    print(f"⬇️  Descargando desde: {url}")

    # Detectar si es un enlace de Google Drive
    if "drive.google.com" in url:
        if gdown is None:
            print(
                "\n❌ Error: Para descargar desde Google Drive necesitas instalar 'gdown'."
            )
            print("   Ejecuta en tu consola: uv pip install gdown")
            sys.exit(1)

        print("☁️  Detectado enlace de Google Drive. Usando gdown...")
        # fuzzy=True ayuda a que gdown entienda las URLs de visualización estándar de Drive
        output_path = gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)

        if output_path is None:
            print(
                "\n❌ Error al descargar con gdown. Verifica que el enlace sea público."
            )
            sys.exit(1)

    else:
        # Descarga estándar para otros enlaces
        def progress(count, block_size, total_size):
            # Prevenir división por cero si el servidor no reporta el tamaño
            total = total_size if total_size > 0 else 1
            pct = count * block_size * 100 / total
            print(f"\r  {pct:.1f}%", end="", flush=True)

        try:
            urlretrieve(url, str(dest), reporthook=progress)
        except Exception as e:
            print(f"\n❌ Error durante la descarga: {e}")
            sys.exit(1)

    print(f"\n  ✅ Descargado: {dest}")


def verify_file(filepath: Path, expected_hash: str) -> bool:
    """Verifica SHA256."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def unpack_models(archive_path: Path):
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
            # Seguridad: verificar que no hay paths maliciosos
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    print(f"  ⚠️  Path sospechoso ignorado: {member.name}")
                    continue

                # Extraer a models/saved/ (quitando el prefijo "saved/")
                member_name = member.name
                if member_name.startswith("saved/"):
                    member_name = member_name[6:]  # quitar "saved/"

                if not member_name:
                    continue

                dest_path = MODELS_DIR / member_name
                print(f"  📄 {member_name}")

                # Extraer
                source = tar.extractfile(member)
                if source:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest_path, "wb") as target_file:
                        target_file.write(source.read())
    except tarfile.ReadError:
        print("\n❌ Error: El archivo no es un .tar.gz válido.")
        print(
            "   Es probable que la descarga haya fallado o sea un archivo HTML corrupto."
        )
        print("   Elimina el archivo e inténtalo de nuevo.")
        archive_path.unlink(missing_ok=True)
        return False

    # ── Verificar integridad ──
    manifest_path = MODELS_DIR / "manifest.json"
    if manifest_path.exists():
        print("\n🔍 Verificando integridad...")
        with open(manifest_path) as f:
            manifest = json.load(f)

        all_ok = True
        for filename, info in manifest["files"].items():
            filepath = MODELS_DIR / filename
            if not filepath.exists():
                print(f"  ❌ FALTA: {filename}")
                all_ok = False
                continue

            if verify_file(filepath, info["sha256"]):
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✅ {filename:40s} {size_mb:.1f} MB  OK")
            else:
                print(f"  ❌ {filename:40s} HASH INCORRECTO")
                all_ok = False

        manifest_path.unlink()  # Limpiar manifiesto

        if all_ok:
            print("\n✅ Todos los modelos verificados correctamente")
        else:
            print("\n⚠️  Algunos archivos tienen problemas")
            return False
    else:
        print("\n⚠️  Sin manifiesto (no se puede verificar integridad)")

    # ── Mostrar resumen ──
    print("\n📋 Modelos disponibles:")
    for item in sorted(MODELS_DIR.iterdir()):
        if item.is_file() and item.suffix in (".pth", ".pkl", ".json", ".pt"):
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name:40s} {size_mb:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Descomprime modelos pre-entrenados")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL directa de descarga (Google Drive, OneDrive, etc.)",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Ruta al archivo .tar.gz",
    )
    args = parser.parse_args()

    archive = Path(args.archive) if args.archive else ARCHIVE_PATH

    if args.url:
        archive.parent.mkdir(parents=True, exist_ok=True)
        download_file(args.url, archive)

    unpack_models(archive)


if __name__ == "__main__":
    main()
