"""Admin and monitoring endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, save_feedback, get_daily_stats, get_feedback_stats
from app.services.drift import drift_tracker
from app.services.model_loader import model_manager
from app.services.cache import cache

router = APIRouter(prefix="/admin", tags=["admin"])


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="UUID from /predict response")
    correct_label: int = Field(..., ge=0, le=1, description="0=non-toxic, 1=toxic")
    reason: Optional[str] = Field(default=None, description="Why was the prediction wrong?")


class FeedbackResponse(BaseModel):
    status: str
    original_prediction: dict


class ThresholdUpdate(BaseModel):
    language: str = Field(..., description="ISO language code")
    threshold: float = Field(..., ge=0.0, le=1.0)


class StatsResponse(BaseModel):
    period: str
    requests: dict
    performance: dict
    quality: dict
    drift: dict


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    """Submit a correction for a prediction."""
    success = await save_feedback(
        session=db,
        request_id=request.request_id,
        correct_label=request.correct_label,
        reason=request.reason,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {
        "status": "recorded",
        "original_prediction": {
            "request_id": request.request_id,
            "your_feedback": request.correct_label,
        }
    }


@router.get("/stats", response_model=StatsResponse)
async def get_stats(days: int = 1, db: AsyncSession = Depends(get_db)):
    """Get system statistics and drift indicators."""
    stats = await get_daily_stats(db, days)
    feedback = await get_feedback_stats(db, days)
    drift = drift_tracker.check()

    total = stats["total"] or 0
    svm_count = stats["svm_count"] or 0
    mbert_count = stats["mbert_count"] or 0
    cnn_count = stats["cnn_count"] or 0

    feedback_count = feedback["feedback_count"] or 0
    fn = feedback["false_negatives"] or 0
    fp = feedback["false_positives"] or 0

    return {
        "period": f"{days}d",
        "requests": {
            "total": total,
            "svm_routed": svm_count,
            "mbert_routed": mbert_count,
            "cnn_routed": cnn_count,
        },
        "performance": {
            "avg_latency_ms": round(stats["avg_latency"] or 0, 2),
            "error_rate": 0.0,  # Would track from exception handling
        },
        "quality": {
            "feedback_count": feedback_count,
            "fp_rate": round(fp / feedback_count, 4) if feedback_count > 0 else 0.0,
            "fn_rate": round(fn / feedback_count, 4) if feedback_count > 0 else 0.0,
        },
        "drift": drift,
    }


@router.get("/models")
async def list_models():
    """List available models and their status."""
    return {
        "models": [
            {
                "name": "svm",
                "type": "sklearn_sgd",
                "description": "Linear SVM on mBERT [CLS] embeddings",
                "speed": "fast",
                "status": model_manager.status["models"].get("svm", "unknown"),
            },
            {
                "name": "mbert",
                "type": "transformers",
                "description": "Fine-tuned mBERT for sequence classification",
                "speed": "slow",
                "status": model_manager.status["models"].get("mbert", "unknown"),
            },
            {
                "name": "cnn",
                "type": "pytorch_cnn",
                "description": "CNN with max-pooling on mBERT hidden states",
                "speed": "medium",
                "status": model_manager.status["models"].get("cnn", "unknown"),
            },
        ]
    }


@router.post("/threshold")
async def update_threshold(update: ThresholdUpdate):
    """Update classification threshold for a language."""
    from app.config import settings
    settings.thresholds[update.language] = update.threshold
    return {
        "language": update.language,
        "new_threshold": update.threshold,
        "all_thresholds": settings.thresholds,
    }


@router.post("/cache/clear")
async def clear_cache():
    """Clear prediction cache."""
    cache.clear()
    return {"status": "cache cleared"}
