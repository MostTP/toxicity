"""Simple caching layer - Redis if available, dict fallback."""

import hashlib
import time
from typing import Optional, Dict, Any

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from app.config import settings


class Cache:
    """Unified cache interface."""

    def __init__(self):
        self._local: Dict[str, Any] = {}
        self._redis: Optional[Any] = None

        if settings.redis_url and HAS_REDIS:
            try:
                self._redis = redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def _key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        h = hashlib.sha256(f"{text}:{model}".encode()).hexdigest()[:16]
        return f"pred:{h}"

    def get(self, text: str, model: str) -> Optional[dict]:
        """Get cached prediction if exists and not expired."""
        key = self._key(text, model)

        if self._redis:
            try:
                import json
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                pass

        # Fallback to in-memory
        if key in self._local:
            entry = self._local[key]
            if entry["expires"] > time.time():
                return entry["value"]
            del self._local[key]

        return None

    def set(self, text: str, model: str, value: dict, ttl: int = 3600):
        """Cache a prediction result."""
        key = self._key(text, model)

        if self._redis:
            try:
                import json
                self._redis.setex(key, ttl, json.dumps(value))
                return
            except Exception:
                pass

        # Fallback to in-memory
        self._local[key] = {
            "value": value,
            "expires": time.time() + ttl,
        }

    def clear(self):
        """Clear all cached entries."""
        self._local.clear()
        if self._redis:
            try:
                self._redis.flushdb()
            except Exception:
                pass


# Singleton
cache = Cache()
