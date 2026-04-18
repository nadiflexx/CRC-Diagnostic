# frontend/utils/video_processor.py
"""
Video processor utility for endoscopy video frame extraction.
Extracts frames every N seconds and returns them as PIL Images.
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile

import cv2
from PIL import Image

MAX_VIDEO_DURATION_SECONDS = 30
FRAME_INTERVAL_SECONDS = 3


def _get_video_duration(cap: cv2.VideoCapture) -> float:
    """Return video duration in seconds."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps > 0:
        return total_frames / fps
    return 0.0


def extract_frames_from_video(
    video_bytes: bytes,
    interval_seconds: float = FRAME_INTERVAL_SECONDS,
    max_duration: float = MAX_VIDEO_DURATION_SECONDS,
) -> list[dict]:
    """
    Extract frames from a video every `interval_seconds` seconds.

    Args:
        video_bytes: Raw video file bytes.
        interval_seconds: Time between extracted frames.
        max_duration: Maximum video duration to process (seconds).

    Returns:
        List of dicts with keys:
            - ``timestamp``: float seconds from start.
            - ``image``: PIL Image (RGB).
            - ``file_obj``: BytesIO ready for upload.
            - ``filename``: suggested filename.
    """
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        Path(tmp_path).unlink(missing_ok=True)
        raise ValueError("No se pudo abrir el archivo de vídeo.")

    duration = _get_video_duration(cap)
    effective_duration = min(duration, max_duration)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames: list[dict] = []
    timestamp = 0.0

    while timestamp <= effective_duration:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        buf.seek(0)

        frames.append(
            {
                "timestamp": round(timestamp, 2),
                "image": pil_img,
                "file_obj": buf,
                "filename": f"frame_{timestamp:.1f}s.jpg",
            }
        )
        timestamp += interval_seconds

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return frames


def pil_to_bytes_io(pil_img: Image.Image) -> io.BytesIO:
    """Convert a PIL Image to a fresh BytesIO (JPEG)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return buf
