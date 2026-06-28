"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.services.model_loader import model_manager
from app.routers import predict, admin

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Initializing database...")
    init_db()

    logger.info("Loading models...")
    model_manager.load_all()

    logger.info("API ready")
    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Toxicity Detection API",
    description="Multilingual toxic comment detection using mBERT, SVM, and CNN with subword tokenization",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - open to all (no auth required)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# predict.py has NO prefix in its router, so add it here
app.include_router(predict.router, prefix="/predict")

# admin.py ALREADY has prefix="/admin" in its router, so DON'T add it here
app.include_router(admin.router)


@app.get("/health")
async def health_check():
    """System health and model status."""
    return {
        "status": "healthy",
        "models": model_manager.status,
    }


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Toxicity Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "batch": "POST /predict/batch",
            "feedback": "POST /admin/feedback",
            "stats": "GET /admin/stats",
            "recent": "GET /admin/recent",
            "health": "GET /health",
        }
    }