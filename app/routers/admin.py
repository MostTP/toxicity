"""Admin and monitoring endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, text, func

from app.db import (
    get_db, save_feedback, get_daily_stats, get_feedback_stats,
    save_general_feedback, get_general_feedback, Prediction, GeneralFeedback
)
from app.services.drift import drift_tracker
from app.services.model_loader import model_manager
from app.services.cache import cache

router = APIRouter(prefix="/admin", tags=["admin"])


# --- Existing: Prediction Correction Feedback ---

class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="UUID from /predict response")
    correct_label: int = Field(..., ge=0, le=1, description="0=non-toxic, 1=toxic")
    reason: Optional[str] = Field(default=None, description="Why was the prediction wrong?")


class FeedbackResponse(BaseModel):
    status: str
    original_prediction: dict


# --- NEW: General User Feedback ---

class GeneralFeedbackCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    type: str = Field(..., description="general|bug|feature|accuracy")
    text: str = Field(..., min_length=1, max_length=5000)
    email: Optional[str] = Field(default=None, max_length=255)


class GeneralFeedbackItem(BaseModel):
    id: int
    rating: int
    type: str
    text: str
    email: Optional[str]
    created_at: Optional[str]


class GeneralFeedbackListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[GeneralFeedbackItem]


class ThresholdUpdate(BaseModel):
    language: str = Field(..., description="ISO language code")
    threshold: float = Field(..., ge=0.0, le=1.0)


class StatsResponse(BaseModel):
    period: str
    requests: dict
    performance: dict
    quality: dict
    drift: dict


# --- Routes ---

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


@router.post("/general-feedback")
async def submit_general_feedback(
    request: GeneralFeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    """Submit general user feedback (ratings, bugs, features, etc.)."""
    feedback_id = await save_general_feedback(
        session=db,
        rating=request.rating,
        type=request.type,
        text=request.text,
        email=request.email,
    )

    return {
        "status": "recorded",
        "id": feedback_id,
        "received_at": datetime.utcnow().isoformat(),
    }


@router.get("/general-feedback", response_model=GeneralFeedbackListResponse)
async def list_general_feedback(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List general feedback submissions."""
    result = await get_general_feedback(db, limit=limit, offset=offset)

    return {
        "total": result["total"],
        "limit": result["limit"],
        "offset": result["offset"],
        "items": [
            {
                "id": item.id,
                "rating": item.rating,
                "type": item.type,
                "text": item.text,
                "email": item.email,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in result["items"]
        ],
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
            "error_rate": 0.0,
        },
        "quality": {
            "feedback_count": feedback_count,
            "fp_rate": round(fp / feedback_count, 4) if feedback_count > 0 else 0.0,
            "fn_rate": round(fn / feedback_count, 4) if feedback_count > 0 else 0.0,
        },
        "drift": drift,
    }


@router.get("/recent")
async def get_recent(
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get recent predictions with pagination."""
    count_result = await db.execute(select(func.count()).select_from(Prediction))
    total = count_result.scalar()
    
    result = await db.execute(
        select(Prediction)
        .order_by(desc(Prediction.created_at))
        .limit(limit)
        .offset(offset)
    )
    predictions = result.scalars().all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "predictions": [
            {
                "id": p.id,
                "model_used": p.model_used,
                "confidence": round(p.confidence, 4),
                "label": p.label,
                "language": p.language or "unknown",
                "source": p.source or "web",
                "latency_ms": p.latency_ms,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in predictions
        ]
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


@router.get("/model-stats")
async def get_model_stats(days: int = 7, db: AsyncSession = Depends(get_db)):
    """Get model usage and accuracy stats."""
    since = text(f"datetime('now', '-{days} days')")
    
    result = await db.execute(
        select(
            Prediction.model_used,
            func.count().label("count"),
            func.avg(Prediction.confidence).label("avg_confidence"),
        )
        .where(Prediction.created_at >= since)
        .group_by(Prediction.model_used)
    )
    
    model_stats = {}
    for row in result.mappings():
        model_stats[row["model_used"]] = {
            "count": row["count"],
            "avg_confidence": round(row["avg_confidence"] or 0, 4),
        }
    
    return {
        "period": f"{days}d",
        "models": model_stats,
    }


@router.get("/language-stats")
async def get_language_stats(days: int = 7, db: AsyncSession = Depends(get_db)):
    """Get language distribution stats."""
    since = text(f"datetime('now', '-{days} days')")
    
    result = await db.execute(
        select(
            Prediction.language,
            func.count().label("count"),
        )
        .where(Prediction.created_at >= since)
        .group_by(Prediction.language)
        .order_by(func.count().desc())
    )
    
    total = 0
    languages = []
    for row in result.mappings():
        count = row["count"]
        total += count
        languages.append({
            "language": row["language"] or "unknown",
            "count": count,
        })
    
    for lang in languages:
        lang["percentage"] = round(lang["count"] / total * 100, 1) if total > 0 else 0
    
    return {
        "period": f"{days}d",
        "total": total,
        "languages": languages,
    }