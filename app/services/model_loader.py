"""Load and manage SVM, mBERT, and CNN models."""

import gc
import logging
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from app.config import settings

logger = logging.getLogger(__name__)

MAX_LENGTH = 128

MBERT_MODEL_NAME = "unitary/toxic-bert"
SVM_MODEL_NAME = "bert-base-multilingual-cased"


class CNNmBERTClassifier(nn.Module):
    """CNN on top of frozen mBERT hidden states."""

    def __init__(self, mbert_model, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.mbert = mbert_model
        for param in self.mbert.parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.mbert(input_ids=input_ids, attention_mask=attention_mask)
            x = outputs.last_hidden_state

        x = x.permute(0, 2, 1)
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x1 = torch.max(x1, dim=2)[0]
        x2 = torch.max(x2, dim=2)[0]
        x3 = torch.max(x3, dim=2)[0]
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)
        logits = self.fc(x)

        result = {"logits": logits}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            result["loss"] = loss
        return result


class ModelManager:
    """Manages loading and lifecycle of all three models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer: Optional[AutoTokenizer] = None
        self.mbert: Optional[AutoModelForSequenceClassification] = None
        self.mbert_base: Optional[AutoModel] = None
        self.cnn: Optional[CNNmBERTClassifier] = None

        self.svm_tokenizer: Optional[AutoTokenizer] = None
        self.svm_mbert_base: Optional[AutoModel] = None
        self.svm = None

        self._status: Dict[str, str] = {}

    def load_all(self):
        """Load all models at startup."""
        self._load_tokenizer()
        self._load_mbert_base()
        self._load_svm_models()
        self._load_mbert()
        self._load_cnn()
        logger.info(f"Model status: {self._status}")

    def _load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL_NAME)
            self._status["tokenizer"] = "loaded"
            logger.info("Main tokenizer loaded (toxic-bert)")
        except Exception as e:
            self._status["tokenizer"] = f"error: {e}"
            logger.error(f"Failed to load tokenizer: {e}")

    def _load_mbert_base(self):
        try:
            self.mbert_base = AutoModel.from_pretrained(MBERT_MODEL_NAME)
            self.mbert_base.to(self.device)
            self.mbert_base.eval()
            self._status["mbert_base"] = "loaded"
            logger.info("mBERT base loaded (toxic-bert)")
        except Exception as e:
            self._status["mbert_base"] = f"error: {e}"
            logger.error(f"Failed to load mBERT base: {e}")

    def _load_svm_models(self):
        try:
            self.svm_tokenizer = AutoTokenizer.from_pretrained(SVM_MODEL_NAME)
            self.svm_mbert_base = AutoModel.from_pretrained(SVM_MODEL_NAME)
            self.svm_mbert_base.to(self.device)
            self.svm_mbert_base.eval()
            self._status["svm_base"] = "loaded"
            logger.info("SVM base model loaded (bert-base-multilingual-cased)")
        except Exception as e:
            self._status["svm_base"] = f"error: {e}"
            logger.error(f"Failed to load SVM base: {e}")

        try:
            if settings.svm_path.exists():
                pkg = joblib.load(settings.svm_path)
                self.svm = pkg['svm_model']
                self.svm_scaler = pkg['scaler']  # EXTRACT SCALER
                self._status["svm"] = "loaded"
                logger.info("SVM classifier loaded")
            else:
                self._status["svm"] = "not_found"
                logger.warning(f"SVM model not found at {settings.svm_path}")
        except Exception as e:
            self._status["svm"] = f"error: {e}"
            logger.error(f"Failed to load SVM: {e}")

    def _load_mbert(self):
        try:
            if settings.mbert_path.exists() and any(settings.mbert_path.iterdir()):
                self.mbert = AutoModelForSequenceClassification.from_pretrained(
                    str(settings.mbert_path)
                )
                self.mbert.to(self.device)
                self.mbert.eval()
                self._status["mbert"] = "loaded"
                logger.info(f"mBERT loaded with {self.mbert.config.num_labels} labels")
            else:
                self._status["mbert"] = "not_found"
                logger.warning(f"mBERT model not found at {settings.mbert_path}")
        except Exception as e:
            self._status["mbert"] = f"error: {e}"
            logger.error(f"Failed to load mBERT: {e}")

    def _load_cnn(self):
        try:
            if settings.cnn_path.exists():
                # FIXED: Use svm_mbert_base (multilingual-cased, matches CNN checkpoint)
                if self.svm_mbert_base is None:
                    raise RuntimeError("SVM base model required for CNN not loaded")

                self.cnn = CNNmBERTClassifier(self.svm_mbert_base).to(self.device)
                checkpoint = torch.load(settings.cnn_path, map_location=self.device)

                if "model_state_dict" in checkpoint:
                    self.cnn.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.cnn.load_state_dict(checkpoint["state_dict"])
                else:
                    self.cnn.load_state_dict(checkpoint)

                self.cnn.eval()
                self._status["cnn"] = "loaded"
                logger.info("CNN loaded")
            else:
                self._status["cnn"] = "not_found"
                logger.warning(f"CNN model not found at {settings.cnn_path}")
        except Exception as e:
            self._status["cnn"] = f"error: {e}"
            logger.error(f"Failed to load CNN: {e}")

    def get_embeddings(self, texts: list) -> np.ndarray:
        if self.mbert_base is None:
            raise RuntimeError("mBERT base model not loaded")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._extract_embeddings(texts, self.tokenizer, self.mbert_base)

    def get_svm_embeddings(self, texts: list) -> np.ndarray:
        if self.svm_mbert_base is None:
            raise RuntimeError("SVM base model not loaded")
        if self.svm_tokenizer is None:
            raise RuntimeError("SVM tokenizer not loaded")
        return self._extract_embeddings(texts, self.svm_tokenizer, self.svm_mbert_base)

    def _extract_embeddings(self, texts: list, tokenizer, model) -> np.ndarray:
        model.eval()
        all_embeddings = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(cls)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.array(all_embeddings)

    @property
    def status(self) -> Dict:
        return {
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
            "models": self._status,
        }


model_manager = ModelManager()