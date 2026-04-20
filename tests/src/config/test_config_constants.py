# tests/test_constants.py
"""
Tests for constants module.
"""


class TestUploadAllowedExtensions:
    def test_extensions_importable(self):
        from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS

        assert UPLOAD_ALLOWED_EXTENSIONS is not None

    def test_jpg_allowed(self):
        from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS

        assert ".jpg" in UPLOAD_ALLOWED_EXTENSIONS

    def test_png_allowed(self):
        from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS

        assert ".png" in UPLOAD_ALLOWED_EXTENSIONS

    def test_extensions_are_lowercase(self):
        from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS

        for ext in UPLOAD_ALLOWED_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} is not lowercase"

    def test_extensions_start_with_dot(self):
        from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS

        for ext in UPLOAD_ALLOWED_EXTENSIONS:
            assert ext.startswith("."), f"Extension {ext} doesn't start with dot"

    def test_extensions_is_iterable(self):
        from src.config.constants import UPLOAD_ALLOWED_EXTENSIONS

        assert hasattr(UPLOAD_ALLOWED_EXTENSIONS, "__iter__")
