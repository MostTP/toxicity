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
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _get_threshold(language: str) -> float:
    return settings.thresholds.get(language, settings.thresholds["default"])


class Predictor:
    def __init__(self):
        # DISABLED: SVM model is corrupted (sklearn version mismatch)
        # Only mBERT and CNN are functional
        pass

    def predict_single(
        self,
        text: str,
        model: str = "auto",
        source: Optional[str] = None,
        language: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict:
        start_time = time.time()

        cached = cache.get(text, model)
        if cached:
            cached["cached"] = True
            return cached

        detected_lang = language or _detect_language(text)
        threshold = _get_threshold(detected_lang)

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

        drift_tracker.record(float(result["confidence"]))
        cache.set(text, model, response)
        return response

    def _run_inference(self, text: str, model: str, threshold: float) -> Dict:
        if model == "auto":
            # Try CNN first (fast, well-calibrated)
            if model_manager.cnn is not None:
                cnn_conf, _ = self._cnn_predict([text])
                cnn_conf = float(cnn_conf[0])
                # Use CNN if confident, otherwise fall back to mBERT
                if cnn_conf > 0.92 or cnn_conf < 0.20:
                    return self._format_result(cnn_conf, "cnn", threshold)
                
                if model_manager.mbert is not None:
                    mbert_conf, _ = self._mbert_predict([text])
                    return self._format_result(float(mbert_conf[0]), "mbert", threshold)
                
                return self._format_result(cnn_conf, "cnn", threshold)

            # No CNN, use mBERT
            if model_manager.mbert is not None:
                mbert_conf, _ = self._mbert_predict([text])
                return self._format_result(float(mbert_conf[0]), "mbert", threshold)

            raise RuntimeError("No models available")

        if model == "svm":
            # SVM is disabled due to corrupted model file
            raise RuntimeError("SVM model is corrupted (sklearn version mismatch). Please use 'mbert' or 'cnn'.")

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

    def _mbert_predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        enc = model_manager.tokenizer(
            texts, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(model_manager.device)
        attention_mask = enc["attention_mask"].to(model_manager.device)

        with torch.no_grad():
            outputs = model_manager.mbert(input_ids=input_ids, attention_mask=attention_mask)
            num_labels = outputs.logits.shape[-1]
            
            if num_labels == 6:
                probs_per_label = torch.sigmoid(outputs.logits)
                probs = probs_per_label.max(dim=1).values.cpu().numpy()
            else:
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        labels = (probs > 0.5).astype(int)
        return probs, labels

    def _cnn_predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        enc = model_manager.svm_tokenizer(
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
        confidence = float(confidence)
        threshold = float(threshold)
        
        if confidence >= threshold:
            toxic = True
        elif confidence <= 0.30:
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
        results = []
        for text in texts:
            result = self.predict_single(text, model, source, language)
            results.append(result)
        return results


predictor = Predictor()