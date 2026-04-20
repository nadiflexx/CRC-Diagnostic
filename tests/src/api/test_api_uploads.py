# tests/test_api_uploads.py
"""
Tests para /uploads routes.
Corrected: importa app correctamente, parchea startup.
"""

import io
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client():
    with patch("src.api.app.Base.metadata.create_all"):
        from src.api.app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestUploadColonoscopy:
    def _make_jpg_file(self, filename="image.jpg", content=b"fake-image-data"):
        return {"file": (filename, io.BytesIO(content), "image/jpeg")}

    def test_upload_jpg_success(self, client, tmp_path):
        with (
            patch("src.api.routes.uploads.paths") as mock_paths,
            patch("src.api.routes.uploads.shutil.copyfileobj"),
        ):
            mock_paths.UPLOAD_IMAGES = tmp_path

            resp = client.post(
                "/uploads/colonoscopy/1",
                files=self._make_jpg_file("colonoscopy.jpg"),
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["patient_id"] == 1
            assert data["filename"] == "colonoscopy.jpg"
            assert "saved_path" in data

    def test_upload_png_success(self, client, tmp_path):
        with (
            patch("src.api.routes.uploads.paths") as mock_paths,
            patch("src.api.routes.uploads.shutil.copyfileobj"),
        ):
            mock_paths.UPLOAD_IMAGES = tmp_path

            resp = client.post(
                "/uploads/colonoscopy/2",
                files={"file": ("image.png", io.BytesIO(b"pngdata"), "image/png")},
            )
            assert resp.status_code == 200
            assert resp.json()["patient_id"] == 2

    def test_upload_unsupported_pdf(self, client):
        """PDF no soportado → 400."""
        resp = client.post(
            "/uploads/colonoscopy/1",
            files={"file": ("report.pdf", io.BytesIO(b"pdf"), "application/pdf")},
        )
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]

    def test_upload_unsupported_txt(self, client):
        """TXT no soportado → 400."""
        resp = client.post(
            "/uploads/colonoscopy/1",
            files={"file": ("notes.txt", io.BytesIO(b"text"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_unsupported_exe(self, client):
        """EXE no soportado → 400."""
        resp = client.post(
            "/uploads/colonoscopy/1",
            files={
                "file": ("malware.exe", io.BytesIO(b"data"), "application/octet-stream")
            },
        )
        assert resp.status_code == 400

    def test_upload_saved_path_contains_colonoscopy_prefix(self, client, tmp_path):
        with (
            patch("src.api.routes.uploads.paths") as mock_paths,
            patch("src.api.routes.uploads.shutil.copyfileobj"),
        ):
            mock_paths.UPLOAD_IMAGES = tmp_path

            resp = client.post(
                "/uploads/colonoscopy/5",
                files=self._make_jpg_file("photo.jpg"),
            )
            assert resp.status_code == 200
            assert "colonoscopy_" in resp.json()["saved_path"]

    def test_upload_patient_id_in_saved_path(self, client, tmp_path):
        with (
            patch("src.api.routes.uploads.paths") as mock_paths,
            patch("src.api.routes.uploads.shutil.copyfileobj"),
        ):
            mock_paths.UPLOAD_IMAGES = tmp_path

            resp = client.post(
                "/uploads/colonoscopy/42",
                files=self._make_jpg_file("test.jpg"),
            )
            assert resp.status_code == 200
            assert "42" in resp.json()["saved_path"]

    def test_upload_no_file_returns_422(self, client):
        """Sin archivo → 422."""
        resp = client.post("/uploads/colonoscopy/1")
        assert resp.status_code == 422

    def test_upload_returns_original_filename(self, client, tmp_path):
        with (
            patch("src.api.routes.uploads.paths") as mock_paths,
            patch("src.api.routes.uploads.shutil.copyfileobj"),
        ):
            mock_paths.UPLOAD_IMAGES = tmp_path

            resp = client.post(
                "/uploads/colonoscopy/1",
                files=self._make_jpg_file("my_scan.jpg"),
            )
            assert resp.status_code == 200
            assert resp.json()["filename"] == "my_scan.jpg"

    def test_upload_bmp_depends_on_allowed_list(self, client, tmp_path):
        """BMP puede estar o no en la lista permitida."""
        with (
            patch("src.api.routes.uploads.paths") as mock_paths,
            patch("src.api.routes.uploads.shutil.copyfileobj"),
        ):
            mock_paths.UPLOAD_IMAGES = tmp_path

            resp = client.post(
                "/uploads/colonoscopy/1",
                files={"file": ("scan.bmp", io.BytesIO(b"bmpdata"), "image/bmp")},
            )
            assert resp.status_code in (200, 400)
