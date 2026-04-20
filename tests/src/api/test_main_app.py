# tests/test_main_app.py
"""
Tests para la aplicación FastAPI.
Corrected: CORS test usando middleware stack real.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client(fastapi_app):
    with TestClient(fastapi_app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_content_type_json(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]

    def test_health_post_not_allowed(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405


class TestAppConfiguration:
    def test_app_title(self, fastapi_app):
        assert fastapi_app.title == "Colon Cancer Diagnosis API"

    def test_app_version(self, fastapi_app):
        assert fastapi_app.version == "1.0.0"

    def test_app_description_mentions_cancer(self, fastapi_app):
        assert "cancer" in fastapi_app.description.lower()

    def test_app_has_patients_router(self, fastapi_app):
        routes = [r.path for r in fastapi_app.routes]
        assert any("/patients" in r for r in routes)

    def test_app_has_diagnosis_router(self, fastapi_app):
        routes = [r.path for r in fastapi_app.routes]
        assert any("/diagnosis" in r for r in routes)

    def test_app_has_uploads_router(self, fastapi_app):
        routes = [r.path for r in fastapi_app.routes]
        assert any("/uploads" in r for r in routes)

    def test_health_route_registered(self, fastapi_app):
        paths = [r.path for r in fastapi_app.routes]
        assert "/health" in paths

    def test_cors_middleware_present(self, fastapi_app):
        """
        Verifica CORS inspeccionando el middleware stack completo.
        FastAPI almacena middlewares en app.middleware_stack (Starlette) o
        en app.user_middleware como Middleware objects.
        """
        # Estrategia 1: buscar en user_middleware por clase
        mw_classes = []
        for mw in fastapi_app.user_middleware:
            cls_name = ""
            if hasattr(mw, "cls"):
                cls_name = getattr(mw.cls, "__name__", "")
            elif hasattr(mw, "kwargs"):
                cls_name = str(mw)
            mw_classes.append(cls_name)

        cors_in_user_mw = any("CORS" in c for c in mw_classes)

        # Estrategia 2: buscar en el stack de middleware construido
        cors_in_stack = False
        if fastapi_app.middleware_stack is not None:
            stack_repr = repr(fastapi_app.middleware_stack)
            cors_in_stack = "CORS" in stack_repr or "cors" in stack_repr.lower()

        # Estrategia 3: verificar mediante una petición OPTIONS real
        # Si CORS está activo, OPTIONS devuelve cabeceras CORS
        with TestClient(fastapi_app, raise_server_exceptions=False) as c:
            resp = c.options(
                "/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                },
            )
            cors_header_present = "access-control-allow-origin" in resp.headers

        assert cors_in_user_mw or cors_in_stack or cors_header_present, (
            "CORS middleware not found via user_middleware, middleware_stack, "
            "or OPTIONS response headers"
        )

    def test_static_reports_mounted(self, fastapi_app):
        route_names = [
            r.name for r in fastapi_app.routes if hasattr(r, "name") and r.name
        ]
        assert "reports" in route_names

    def test_static_uploads_mounted(self, fastapi_app):
        route_names = [
            r.name for r in fastapi_app.routes if hasattr(r, "name") and r.name
        ]
        assert "uploads" in route_names


class TestStartupEvent:
    def test_on_startup_calls_create_all(self):
        with patch("src.api.app.Base.metadata.create_all") as mock_create:
            from src.api import app as app_module

            app_module.on_startup()
            mock_create.assert_called()

    def test_startup_does_not_raise(self):
        with patch("src.api.app.Base.metadata.create_all"):
            from src.api import app as app_module

            try:
                app_module.on_startup()
            except Exception as exc:
                pytest.fail(f"on_startup raised: {exc}")


class TestNotFoundRoutes:
    def test_unknown_route_returns_404(self, client):
        resp = client.get("/this-route-does-not-exist")
        assert resp.status_code == 404

    def test_unknown_post_route(self, client):
        resp = client.post("/this-route-does-not-exist")
        assert resp.status_code in (404, 405)
