"""Core prediction logic with smart routing and drift tracking."""

import hashlib
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from langdetect import detect, LangDetectException

from app.config import settings
from app.services.cache import cache
from app.services.drift import drift_tracker
from app.services.model_loader import model_manager

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    """Create deterministic hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_language(text: str) -> str:
    """Detect language code, fallback to 'unknown'."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _get_threshold(language: str) -> float:
    """Get language-specific threshold."""
    return settings.thresholds.get(language, settings.thresholds["default"])


class Predictor:
    """Main prediction service."""

    def __init__(self):
        self.svm_fallback_threshold = 0.92

    def predict_single(
        self,
        text: str,
        model: str = "auto",
        source: Optional[str] = None,
        language: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict:
        """Classify a single text."""
        start_time = time.time()
        text_hash = _hash_text(text)

        # Check cache
        cached = cache.get(text, model)
        if cached:
            cached["cached"] = True
            return cached

        # Detect language
        detected_lang = language or _detect_language(text)
        threshold = _get_threshold(detected_lang)

        # Route to appropriate model
        result = self._run_inference(text, model, threshold)

        latency_ms = int((time.time() - start_time) * 1000)

        response = {
            "request_id": result["request_id"],
            "toxic": result["toxic"],
            "confidence": float(result["confidence"]),
            "probabilities": {
                "non_toxic": float(result["probabilities"]["non_toxic"]),
                "toxic": float(result["probabilities"]["toxic"]),
            },
            "model_used": result["model_used"],
            "inference_time_ms": latency_ms,
            "threshold": float(threshold),
            "language": detected_lang,
            "cached": False,
        }

        # Record for drift detection
        drift_tracker.record(float(result["confidence"]))

        # Cache result
        cache.set(text, model, response)

        return response

    def _run_inference(
        self, text: str, model: str, threshold: float
    ) -> Dict:
        """Run the actual model inference."""

        # AUTO routing: try SVM first, escalate if uncertain
        if model == "auto":
            if model_manager.svm is not None:
                svm_conf, svm_label = self._svm_predict([text])
                svm_conf = float(svm_conf[0])

                # Clear case: high confidence
                if svm_conf > self.svm_fallback_threshold or svm_conf < (1 - self.svm_fallback_threshold):
                    return self._format_result(svm_conf, "svm", threshold)

                # Uncertain: use mBERT
                if model_manager.mbert is not None:
                    mbert_conf, _ = self._mbert_predict([text])
                    return self._format_result(float(mbert_conf[0]), "mbert", threshold)

                # mBERT unavailable, use SVM anyway
                return self._format_result(svm_conf, "svm", threshold)

            # No SVM, try mBERT
            if model_manager.mbert is not None:
                mbert_conf, _ = self._mbert_predict([text])
                return self._format_result(float(mbert_conf[0]), "mbert", threshold)

            raise RuntimeError("No models available")

        # Explicit model selection
        if model == "svm":
            if model_manager.svm is None:
                raise RuntimeError("SVM model not loaded")
            conf, _ = self._svm_predict([text])
            return self._format_result(float(conf[0]), "svm", threshold)

        elif model == "mbert":
            if model_manager.mbert is None:
                raise RuntimeError("mBERT model not loaded")
            conf, _ = self._mbert_predict([text])
            return self._format_result(float(conf[0]), "mbert", threshold)

        elif model == "cnn":
            if model_manager.cnn is None:
                raise RuntimeError("CNN model not loaded")
            conf, _ = self._cnn_predict([text])
            return self._format_result(float(conf[0]), "cnn", threshold)

        else:
            raise ValueError(f"Unknown model: {model}")

    def _svm_predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """SVM prediction on mBERT embeddings."""
        embeddings = model_manager.get_embeddings(texts)

        # SGDClassifier: decision_function gives distance to hyperplane
        decisions = model_manager.svm.decision_function(embeddings)

        # Convert to probability-like score using sigmoid
        probs = 1 / (1 + np.exp(-decisions))

        labels = (decisions > 0).astype(int)
        return probs, labels

    def _mbert_predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Fine-tuned mBERT prediction."""
        enc = model_manager.tokenizer(
            texts, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(model_manager.device)
        attention_mask = enc["attention_mask"].to(model_manager.device)

        with torch.no_grad():
            outputs = model_manager.mbert(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        labels = (probs > 0.5).astype(int)
        return probs, labels

    def _cnn_predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """CNN + mBERT prediction."""
        enc = model_manager.tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=128, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(model_manager.device)
        attention_mask = enc["attention_mask"].to(model_manager.device)

        with torch.no_grad():
            outputs = model_manager.cnn(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=1)[:, 1].cpu().numpy()

        labels = (probs > 0.5).astype(int)
        return probs, labels

    def _format_result(self, confidence, model_used, threshold):
        """Format raw confidence into standardized response."""
        confidence = float(confidence)
        threshold = float(threshold)
        
        # More conservative: require very high confidence
        if confidence >= 0.90:
            toxic = True
        elif confidence <= 0.50:
            toxic = False
        else:
            return {
                "request_id": str(uuid.uuid4()),
                "toxic": False,
                "confidence": confidence,
                "probabilities": {"non_toxic": round(1-confidence,4), "toxic": round(confidence,4)},
                "model_used": str(model_used),
                "warning": "uncertain",
            }
        
        return {
            "request_id": str(uuid.uuid4()),
            "toxic": bool(toxic),
            "confidence": confidence,
            "probabilities": {"non_toxic": round(1-confidence,4), "toxic": round(confidence,4)},
            "model_used": str(model_used),
        }

    def predict_batch(
        self,
        texts: List[str],
        model: str = "auto",
        source: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Dict]:
        """Classify multiple texts."""
        results = []
        for text in texts:
            result = self.predict_single(text, model, source, language)
            results.append(result)
        return results


# Singleton
predictor = Predictor()