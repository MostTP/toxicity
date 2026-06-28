import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, Text,
    select, func, and_, text
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

Base = declarative_base()


class Prediction(Base):
    """Stores every prediction for analytics and feedback loop."""
    __tablename__ = "predictions"

    id = Column(String, primary_key=True)
    text_hash = Column(String, index=True)
    model_used = Column(String, index=True)
    confidence = Column(Float)
    label = Column(Integer)
    latency_ms = Column(Integer)
    source = Column(String, nullable=True)
    language = Column(String, nullable=True)
    thread_id = Column(String, nullable=True)
    threshold_used = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    feedback_label = Column(Integer, nullable=True)
    feedback_reason = Column(String, nullable=True)


class GeneralFeedback(Base):
    """Stores general user feedback (ratings, bugs, features, etc.)."""
    __tablename__ = "general_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rating = Column(Integer, nullable=False)
    type = Column(String(20), nullable=False)
    text = Column(Text, nullable=False)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Sync engine for initialization
_sync_engine = create_engine(f"sqlite:///{settings.db_path}")

# Async engine for runtime
async_engine = create_async_engine(f"sqlite+aiosqlite:///{settings.db_path}")
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)


def init_db():
    """Create tables synchronously at startup."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=_sync_engine)


async def get_db():
    """Dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        yield session


async def save_prediction(
    session: AsyncSession,
    text_hash: str,
    model_used: str,
    confidence: float,
    label: int,
    latency_ms: int,
    source: Optional[str] = None,
    language: Optional[str] = None,
    thread_id: Optional[str] = None,
    threshold_used: float = 0.5,
) -> str:
    """Save a prediction and return its ID."""
    pred_id = str(uuid.uuid4())
    prediction = Prediction(
        id=pred_id,
        text_hash=text_hash,
        model_used=model_used,
        confidence=confidence,
        label=label,
        latency_ms=latency_ms,
        source=source,
        language=language,
        thread_id=thread_id,
        threshold_used=threshold_used,
    )
    session.add(prediction)
    await session.commit()
    return pred_id


async def save_feedback(
    session: AsyncSession,
    request_id: str,
    correct_label: int,
    reason: Optional[str] = None,
) -> bool:
    """Update a prediction with user feedback."""
    result = await session.execute(
        select(Prediction).where(Prediction.id == request_id)
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        return False

    prediction.feedback_label = correct_label
    prediction.feedback_reason = reason
    await session.commit()
    return True


async def save_general_feedback(
    session: AsyncSession,
    rating: int,
    type: str,
    text: str,
    email: Optional[str] = None,
) -> int:
    """Save general user feedback and return its ID."""
    entry = GeneralFeedback(
        rating=rating,
        type=type,
        text=text,
        email=email,
        created_at=datetime.utcnow(),
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry.id


async def get_general_feedback(
    session: AsyncSession,
    limit: int = 50,
    offset: int = 0,
):
    """List general feedback submissions."""
    count_result = await session.execute(
        select(func.count()).select_from(GeneralFeedback)
    )
    total = count_result.scalar()

    result = await session.execute(
        select(GeneralFeedback)
        .order_by(desc(GeneralFeedback.created_at))
        .limit(limit)
        .offset(offset)
    )
    items = result.scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


async def get_daily_stats(session: AsyncSession, days: int = 1):
    """Get aggregated stats for the last N days."""
    since = text(f"datetime('now', '-{days} days')")

    result = await session.execute(
        select(
            func.count().label("total"),
            func.avg(Prediction.latency_ms).label("avg_latency"),
            func.sum(func.case((Prediction.model_used == "svm", 1), else_=0)).label("svm_count"),
            func.sum(func.case((Prediction.model_used == "mbert", 1), else_=0)).label("mbert_count"),
            func.sum(func.case((Prediction.model_used == "cnn", 1), else_=0)).label("cnn_count"),
            func.avg(Prediction.confidence).label("avg_confidence"),
        ).where(Prediction.created_at >= since)
    )
    return result.mappings().first()


async def get_feedback_stats(session: AsyncSession, days: int = 1):
    """Calculate false positive/negative rates from feedback."""
    since = text(f"datetime('now', '-{days} days')")

    result = await session.execute(
        select(
            func.count().label("feedback_count"),
            func.sum(func.case(
                (and_(Prediction.label == 0, Prediction.feedback_label == 1), 1),
                else_=0
            )).label("false_negatives"),
            func.sum(func.case(
                (and_(Prediction.label == 1, Prediction.feedback_label == 0), 1),
                else_=0
            )).label("false_positives"),
        ).where(
            and_(
                Prediction.created_at >= since,
                Prediction.feedback_label.isnot(None)
            )
        )
    )
    return result.mappings().first()