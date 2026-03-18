from datetime import datetime
from pathlib import Path
import shutil

from fastapi import APIRouter, HTTPException, UploadFile

from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS
from src.config.paths import paths

router = APIRouter(prefix="/uploads", tags=["Uploads"])


@router.post("/colonoscopy/{patient_id}")
async def upload_colonoscopy(patient_id: int, file: UploadFile):
    """Upload a colonoscopy image for a specific patient.

    Args:
        patient_id (int): The ID of the patient.
        file (UploadFile): The colonoscopy image file.

    Returns:
        dict: A dictionary containing the filename, saved path, and patient ID.

    Raises:
        HTTPException: If the file format is not supported.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in UPLOAD_ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_dir = paths.UPLOAD_IMAGES / str(patient_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"colonoscopy_{ts}{ext}"

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "filename": file.filename,
        "saved_path": str(dest_path),
        "patient_id": patient_id,
    }
