"""Drift detection using rolling confidence statistics."""

import time
from collections import deque
from typing import Dict, Optional


class DriftTracker:
    """Track prediction confidence over time to detect distribution shift."""

    def __init__(self, window_seconds: int = 86400, alert_threshold: float = 0.12):
        """
        Args:
            window_seconds: Rolling window size (default 24h)
            alert_threshold: Confidence drop % to trigger alert
        """
        self.window = window_seconds
        self.alert_threshold = alert_threshold
        self._history: deque = deque()
        self._baseline: Optional[float] = None

    def record(self, confidence: float, timestamp: Optional[float] = None):
        """Record a prediction confidence score."""
        ts = timestamp or time.time()
        self._history.append((ts, confidence))

        # Remove old entries
        cutoff = ts - self.window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Set baseline after first 100 predictions
        if self._baseline is None and len(self._history) >= 100:
            self._baseline = self._current_avg()

    def _current_avg(self) -> float:
        """Current average confidence in window."""
        if not self._history:
            return 0.0
        return sum(c for _, c in self._history) / len(self._history)

    def check(self) -> Dict:
        """Check for drift and return status."""
        current = self._current_avg()

        if self._baseline is None or len(self._history) < 100:
            return {
                "status": "collecting",
                "message": "Collecting baseline data...",
                "confidence_trend": 0.0,
                "current_avg": round(current, 4),
                "baseline": None,
                "alert": None,
            }

        trend = (current - self._baseline) / self._baseline

        if trend < -self.alert_threshold:
            return {
                "status": "alert",
                "message": f"Confidence dropped {abs(trend)*100:.1f}% below baseline",
                "confidence_trend": round(trend, 4),
                "current_avg": round(current, 4),
                "baseline": round(self._baseline, 4),
                "alert": "Average confidence dropped significantly. Possible input distribution shift.",
                "recommended_action": "Review recent feedback and consider retraining.",
            }
        elif trend < -self.alert_threshold / 2:
            return {
                "status": "warning",
                "message": f"Confidence trending down ({trend*100:.1f}%)",
                "confidence_trend": round(trend, 4),
                "current_avg": round(current, 4),
                "baseline": round(self._baseline, 4),
                "alert": None,
            }

        return {
            "status": "normal",
            "message": "Confidence stable",
            "confidence_trend": round(trend, 4),
            "current_avg": round(current, 4),
            "baseline": round(self._baseline, 4),
            "alert": None,
        }

    def reset_baseline(self):
        """Reset baseline (e.g., after retraining)."""
        self._baseline = self._current_avg()


# Singleton
drift_tracker = DriftTracker()
