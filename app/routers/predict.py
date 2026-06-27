"""Prediction endpoints."""

from typing import List, Optional, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, save_prediction
from app.services.predictor import predictor

router = APIRouter()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Comment text to classify")
    model: str = Field(default="auto", pattern="^(auto|svm|mbert|cnn)$")
    source: Optional[str] = Field(default=None, description="Platform source (reddit, discord, etc.)")
    language: Optional[str] = Field(default=None, description="ISO language code (auto-detected if None)")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")
    return_probs: bool = Field(default=True, description="Include probability breakdown")


class PredictResponse(BaseModel):
    request_id: str
    toxic: bool
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    model_used: str
    inference_time_ms: int
    threshold: float
    language: str
    cached: bool


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    model: str = Field(default="auto", pattern="^(auto|svm|mbert|cnn)$")
    source: Optional[str] = None
    language: Optional[str] = None


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_processed: int
    avg_latency_ms: float


@router.post("", response_model=PredictResponse)
async def predict(request: PredictRequest, db: AsyncSession = Depends(get_db)):
    """Classify a single comment."""
    try:
        result = predictor.predict_single(
            text=request.text,
            model=request.model,
            source=request.source,
            language=request.language,
            thread_id=request.thread_id,
        )

        await save_prediction(
            session=db,
            text_hash=result["request_id"],
            model_used=result["model_used"],
            confidence=result["confidence"],
            label=int(result["toxic"]),
            latency_ms=result["inference_time_ms"],
            source=request.source,
            language=result["language"],
            thread_id=request.thread_id,
            threshold_used=result["threshold"],
        )

        if not request.return_probs:
            result["probabilities"] = None

        return result

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest, db: AsyncSession = Depends(get_db)):
    """Classify multiple comments."""
    try:
        results = predictor.predict_batch(
            texts=request.texts,
            model=request.model,
            source=request.source,
            language=request.language,
        )

        for result in results:
            await save_prediction(
                session=db,
                text_hash=result["request_id"],
                model_used=result["model_used"],
                confidence=result["confidence"],
                label=int(result["toxic"]),
                latency_ms=result["inference_time_ms"],
                source=request.source,
                language=result["language"],
            )

        avg_latency = sum(r["inference_time_ms"] for r in results) / len(results)

        return {
            "results": results,
            "total_processed": len(results),
            "avg_latency_ms": round(avg_latency, 2),
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")