"""
Unzip pre-trained models from .zip or .tar.gz
Supports Google Drive downloads.

Usage:
  uv run src/utils/unpack_models.py                        # local file
  uv run src/utils/unpack_models.py --url "https://..."    # from URL
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
from urllib.parse import urlparse
from urllib.request import urlretrieve
import zipfile

import gdown

from src.config.paths import paths

GZIP_MAGIC = b"\x1f\x8b"
ZIP_MAGIC = b"PK\x03\x04"


MODELS_DIR = paths.MODELS
ONNX_DIR = paths.MODELS / "onnx"
ARCHIVE_PATH = paths.MODELS_ROOT / "models_saved.tar.gz"


# ─────────────────────────────────────────────────────────────────────────────
# FILE DETECTION
# ─────────────────────────────────────────────────────────────────────────────


def detect_format(filepath: Path) -> str:
    """Returns 'zip', 'targz', or 'unknown'."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(4)
    except OSError:
        return "unknown"

    if header[:2] == GZIP_MAGIC:
        return "targz"
    if header[:4] == ZIP_MAGIC:
        return "zip"
    return "unknown"


def diagnose_file(filepath: Path) -> str:
    """Human-readable description of what the file actually is."""
    if not filepath.exists():
        return "File does not exist"

    size = filepath.stat().st_size
    if size == 0:
        return "File is empty (0 bytes)"

    with open(filepath, "rb") as f:
        header = f.read(512)

    fmt = detect_format(filepath)
    size_mb = size / 1024 / 1024

    if fmt == "targz":
        return f"Valid gzip/tar.gz ({size_mb:.1f} MB)"
    if fmt == "zip":
        return f"Valid ZIP ({size_mb:.1f} MB)"
    if header[:5] in (b"<!DOC", b"<html"):
        snippet = header[:200].decode("utf-8", errors="replace")
        return f"HTML page (Google Drive error):\n    {snippet}"
    if header[:1] == b"{":
        snippet = header[:200].decode("utf-8", errors="replace")
        return f"JSON/API error response:\n    {snippet}"

    return f"Unknown format — hex: {header[:16].hex()} ({size_mb:.1f} MB)"


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────


def extract_gdrive_id(url: str) -> str | None:
    import re

    for pattern in [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def download_gdrive_gdown(file_id: str, dest: Path) -> bool:
    strategies = [
        {
            "url": f"https://drive.google.com/uc?id={file_id}",
            "output": str(dest),
            "quiet": False,
            "fuzzy": True,
        },
        {"id": file_id, "output": str(dest), "quiet": False},
        {
            "url": f"https://drive.google.com/uc?export=download&id={file_id}",
            "output": str(dest),
            "quiet": False,
            "use_cookies": False,
        },
    ]
    names = ["fuzzy URL", "direct ID", "no-cookies"]

    for i, (kwargs, name) in enumerate(zip(strategies, names, strict=True), 1):
        print(f"\n  🔄 Attempt {i}/{len(strategies)}: gdown {name}")
        try:
            if dest.exists():
                dest.unlink()
            result = gdown.download(**kwargs)
            if result and dest.exists():
                fmt = detect_format(dest)
                if fmt in ("zip", "targz"):
                    print(
                        f"  ✅ Valid {fmt} downloaded ({dest.stat().st_size / 1024 / 1024:.1f} MB)"
                    )
                    return True
                print(f"  ⚠️  Downloaded but invalid: {diagnose_file(dest)}")
                dest.unlink(missing_ok=True)
            else:
                print("  ⚠️  gdown returned None or file missing")
        except Exception as e:
            print(f"  ⚠️  Failed: {e}")

        if i < len(strategies):
            print("  ⏳ Waiting 3s...")
            time.sleep(3)

    return False


def download_file(url: str, dest: Path) -> bool:
    """Downloads file from URL. Returns True on success."""
    print(f"⬇️  Downloading from: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if "drive.google.com" in url:
        print("☁️  Google Drive detected...")
        file_id = extract_gdrive_id(url)
        if not file_id:
            print(f"❌ Cannot extract file ID from: {url}")
            return False
        print(f"  📋 File ID: {file_id}")
        if download_gdrive_gdown(file_id, dest):
            return True
        print("\n❌ Download failed. Manual steps:")
        print(f"   1. Open: https://drive.google.com/uc?export=download&id={file_id}")
        print(f"   2. Save to: {dest}")
        print(f'   3. Run: uv run src/utils/unpack_models.py --archive "{dest}"')
        return False

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            print(
                f"\r  {pct:.1f}% ({count * block_size / 1024 / 1024:.1f} MB)",
                end="",
                flush=True,
            )

    try:
        urlretrieve(url, str(dest), reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n❌ Download error: {e}")
        return False

    fmt = detect_format(dest)
    if fmt == "unknown":
        print(f"❌ Downloaded file is not zip or tar.gz: {diagnose_file(dest)}")
        return False

    print(f"  ✅ Downloaded: {dest} ({fmt})")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# DESTINATION RESOLVER
# ─────────────────────────────────────────────────────────────────────────────


def resolve_dest(member_name: str) -> Path | None:
    """
    Maps an archive member path to its extraction destination.

    Supported structures inside the archive:
      onnx/<file>            → ONNX_DIR/<file>
      saved/onnx/<file>      → ONNX_DIR/<file>
      saved/<file>           → MODELS_DIR/<file>
      <file>.onnx            → ONNX_DIR/<file>
      <file>.pth/pkl/pt/json → MODELS_DIR/<file>
    """
    name = member_name.replace("\\", "/")

    if name.startswith("onnx/"):
        rel = name[len("onnx/") :]
        return ONNX_DIR / rel if rel else None

    if name.startswith("saved/onnx/"):
        rel = name[len("saved/onnx/") :]
        return ONNX_DIR / rel if rel else None

    if name.startswith("saved/"):
        rel = name[len("saved/") :]
        return MODELS_DIR / rel if rel else None

    stem = Path(name).name
    suffix = Path(name).suffix.lower()
    if suffix == ".onnx":
        return ONNX_DIR / stem
    if suffix in (".pth", ".pkl", ".pt", ".json"):
        return MODELS_DIR / stem

    return None


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────


def extract_zip(archive_path: Path) -> tuple[int, int]:
    """Extracts a ZIP archive. Returns (extracted, skipped)."""
    extracted = skipped = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        entries = zf.infolist()
        print(f"  📋 ZIP contains {len(entries)} entries")

        for entry in entries:
            name = entry.filename

            if name.endswith("/") or name.endswith("\\"):
                continue

            if name.startswith("/") or ".." in name:
                print(f"  ⚠️  Suspicious path skipped: {name}")
                skipped += 1
                continue

            dest_path = resolve_dest(name)
            if dest_path is None:
                print(f"  ⏭️  Skipping (unknown path): {name}")
                skipped += 1
                continue

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(entry.filename)
            dest_path.write_bytes(data)
            size_mb = len(data) / 1024 / 1024
            print(f"  📄 {name:60s} ({size_mb:.1f} MB)")
            extracted += 1

    return extracted, skipped


def extract_targz(archive_path: Path) -> tuple[int, int]:
    """Extracts a tar.gz archive. Returns (extracted, skipped)."""
    extracted = skipped = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"  📋 tar.gz contains {len(members)} entries")

        for member in members:
            if member.isdir():
                continue

            if member.name.startswith("/") or ".." in member.name:
                print(f"  ⚠️  Suspicious path skipped: {member.name}")
                skipped += 1
                continue

            dest_path = resolve_dest(member.name)
            if dest_path is None:
                print(f"  ⏭️  Skipping (unknown path): {member.name}")
                skipped += 1
                continue

            source = tar.extractfile(member)
            if source:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                data = source.read()
                dest_path.write_bytes(data)
                size_mb = len(data) / 1024 / 1024
                print(f"  📄 {member.name:60s} ({size_mb:.1f} MB)")
                extracted += 1

    return extracted, skipped


def unpack_models(archive_path: Path) -> bool:
    """Main extraction entry point. Handles both .zip and .tar.gz."""

    if not archive_path.exists():
        print(f"❌ File not found: {archive_path}")
        print("  Options:")
        print("    1. Place the archive in models/")
        print('    2. Use: uv run src/utils/unpack_models.py --url "https://..."')
        return False

    fmt = detect_format(archive_path)
    diagnosis = diagnose_file(archive_path)
    print(f"\n📦 Archive : {archive_path}")
    print(f"   Format  : {diagnosis}")

    if fmt == "unknown":
        print("\n❌ Unrecognized format. Expected .zip or .tar.gz")
        print("   Run with --diagnose for details.")
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🔓 Extracting ({fmt})...")
    try:
        if fmt == "zip":
            extracted, skipped = extract_zip(archive_path)
        else:
            extracted, skipped = extract_targz(archive_path)
    except zipfile.BadZipFile as e:
        print(f"\n❌ Corrupt ZIP: {e}")
        archive_path.unlink(missing_ok=True)
        return False
    except tarfile.ReadError as e:
        print(f"\n❌ Corrupt tar.gz: {e}")
        archive_path.unlink(missing_ok=True)
        return False
    except EOFError:
        print("\n❌ Archive is truncated (incomplete download).")
        archive_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        return False

    print(f"\n  ✅ Extracted: {extracted} files  |  Skipped: {skipped}")

    if extracted == 0:
        print("\n⚠️  No files extracted! Check the archive structure with --list")
        return False

    manifest_path = MODELS_DIR / "manifest.json"
    if manifest_path.exists():
        print("\n🔍 Verifying integrity...")
        with open(manifest_path) as f:
            manifest = json.load(f)

        all_ok = True
        for arcname, info in manifest.get("files", {}).items():
            filepath = resolve_dest(arcname)
            if filepath is None:
                continue
            if not filepath.exists():
                print(f"  ❌ MISSING  : {arcname}")
                all_ok = False
                continue
            actual = _sha256(filepath)
            size_mb = filepath.stat().st_size / 1024 / 1024
            if actual == info["sha256"]:
                print(f"  ✅ {arcname:55s} {size_mb:.1f} MB")
            else:
                print(f"  ❌ {arcname:55s} HASH MISMATCH")
                all_ok = False

        manifest_path.unlink()
        if all_ok:
            print("\n✅ All files verified correctly")
        else:
            print("\n⚠️  Some files failed integrity check")
            return False
    else:
        print("\n⚠️  No manifest.json found (skipping integrity check)")

    _print_summary()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_summary():
    print("\n📋 Available models:")
    for directory, label in [(MODELS_DIR, "PyTorch/Pickle"), (ONNX_DIR, "ONNX")]:
        if not directory.exists():
            continue
        files = [
            f
            for f in sorted(directory.iterdir())
            if f.is_file() and f.suffix in (".pth", ".pkl", ".onnx", ".json", ".pt")
        ]
        if files:
            print(f"\n  [{label}] → {directory}")
            for item in files:
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"    {item.name:50s} {size_mb:.1f} MB")


def list_archive(archive_path: Path):
    """Lists archive contents without extracting."""
    if not archive_path.exists():
        print(f"❌ File not found: {archive_path}")
        return

    fmt = detect_format(archive_path)
    print(f"📋 Contents of: {archive_path}")
    print(f"   Format: {diagnose_file(archive_path)}\n")

    if fmt == "zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                entries = zf.infolist()
                print(f"  Total entries: {len(entries)}\n")
                for e in entries:
                    size_mb = e.file_size / 1024 / 1024
                    dest = resolve_dest(e.filename)
                    dest_str = str(dest) if dest else "⏭️  skipped"
                    print(f"  {e.filename:55s} {size_mb:6.2f} MB  →  {dest_str}")
        except Exception as e:
            print(f"❌ Cannot read ZIP: {e}")

    elif fmt == "targz":
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = tar.getmembers()
                print(f"  Total entries: {len(members)}\n")
                for m in members:
                    size_mb = m.size / 1024 / 1024
                    dest = resolve_dest(m.name)
                    dest_str = str(dest) if dest else "⏭️  skipped"
                    print(f"  {m.name:55s} {size_mb:6.2f} MB  →  {dest_str}")
        except Exception as e:
            print(f"❌ Cannot read tar.gz: {e}")
    else:
        print("❌ Not a recognized archive format")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unpack pre-trained models (.zip or .tar.gz)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run src/utils/unpack_models.py
  uv run src/utils/unpack_models.py --url "https://drive.google.com/..."
  uv run src/utils/unpack_models.py --archive models/onnx_models.zip
  uv run src/utils/unpack_models.py --archive models/onnx_models.zip --list
  uv run src/utils/unpack_models.py --diagnose
        """,
    )
    parser.add_argument(
        "--url", type=str, default=None, help="Download URL (Google Drive or direct)"
    )
    parser.add_argument(
        "--archive", type=str, default=None, help="Path to .zip or .tar.gz file"
    )
    parser.add_argument(
        "--list", action="store_true", help="List contents without extracting"
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Diagnose file format only"
    )
    args = parser.parse_args()

    if args.archive:
        archive = Path(args.archive)
    else:
        zip_candidate = ARCHIVE_PATH.with_suffix("").with_suffix(".zip")
        if zip_candidate.exists() and not ARCHIVE_PATH.exists():
            archive = zip_candidate
        else:
            archive = ARCHIVE_PATH

    if args.url:
        url_path = urlparse(args.url).path
        if url_path.endswith(".zip"):
            archive = archive.with_suffix(".zip")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not download_file(args.url, archive):
            sys.exit(1)

    if args.diagnose:
        print(f"🔍 {archive}")
        print(f"   {diagnose_file(archive)}")
        print(f"   Format: {detect_format(archive)}")
        return

    if args.list:
        list_archive(archive)
        return

    if not unpack_models(archive):
        sys.exit(1)


if __name__ == "__main__":
    main()
