"""
FastAPI application entry point.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import diagnosis, patients, uploads
from src.config.paths import paths
from src.database.connection import engine
from src.database.models import Base

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Colon Cancer Diagnosis API",
    description="AI-powered colon cancer diagnostic simulator",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

paths.REPORTS.mkdir(parents=True, exist_ok=True)
paths.UPLOAD_IMAGES.mkdir(parents=True, exist_ok=True)

app.mount("/static/reports", StaticFiles(directory=str(paths.REPORTS)), name="reports")
app.mount(
    "/static/uploads", StaticFiles(directory=str(paths.UPLOAD_IMAGES)), name="uploads"
)

app.include_router(patients.router)
app.include_router(uploads.router)
app.include_router(diagnosis.router)


@app.get("/health")
def health():
    """Check the health of the API."""
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    """Create database tables on startup."""
    Base.metadata.create_all(bind=engine)
    logging.getLogger(__name__).info("✅ API ready")
