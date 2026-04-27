# tests/test_database_connection.py
"""
Tests para src/database/connection.py
"""

from unittest.mock import MagicMock, patch

import pytest


class TestGetDb:
    def test_get_db_yields_session(self):
        """get_db debe hacer yield de una sesión."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db

            with get_db() as db:
                assert db is mock_session

    def test_get_db_commits_on_success(self):
        """get_db debe hacer commit si no hay excepción."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db

            with get_db() as db:
                pass
            mock_session.commit.assert_called_once()

    def test_get_db_rollback_on_exception(self):
        """get_db debe hacer rollback si hay excepción."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db

            with pytest.raises(ValueError), get_db() as db:
                raise ValueError("DB error")
            mock_session.rollback.assert_called_once()

    def test_get_db_always_closes_session(self):
        """get_db siempre cierra la sesión."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db

            with get_db() as db:
                pass
            mock_session.close.assert_called_once()

    def test_get_db_closes_on_exception(self):
        """get_db cierra la sesión incluso con excepción."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db

            with pytest.raises(RuntimeError), get_db() as db:
                raise RuntimeError("fail")
            mock_session.close.assert_called_once()

    def test_get_db_reraises_exception(self):
        """get_db relanza la excepción original."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db

            with pytest.raises(KeyError, match="test_key"), get_db() as db:
                raise KeyError("test_key")


class TestGetDbDependency:
    def test_yields_session(self):
        """get_db_dependency debe yield una sesión."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db_dependency

            gen = get_db_dependency()
            db = next(gen)
            assert db is mock_session

    def test_closes_session_after_use(self):
        """get_db_dependency cierra la sesión al finalizar."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db_dependency

            gen = get_db_dependency()
            next(gen)
            try:
                next(gen)
            except StopIteration:
                pass
            mock_session.close.assert_called_once()

    def test_does_not_commit(self):
        """get_db_dependency NO hace commit (responsabilidad del caller)."""
        mock_session = MagicMock()

        with patch("src.database.connection.SessionLocal", return_value=mock_session):
            from src.database.connection import get_db_dependency

            gen = get_db_dependency()
            next(gen)
            try:
                next(gen)
            except StopIteration:
                pass
            mock_session.commit.assert_not_called()

    def test_is_generator(self):
        """get_db_dependency debe ser un generador."""
        import inspect

        with patch("src.database.connection.SessionLocal", return_value=MagicMock()):
            from src.database.connection import get_db_dependency

            gen = get_db_dependency()
            assert inspect.isgenerator(gen)


class TestEngineConfig:
    def test_engine_importable(self):
        from src.database.connection import engine

        assert engine is not None

    def test_session_local_importable(self):
        from src.database.connection import SessionLocal

        assert SessionLocal is not None
