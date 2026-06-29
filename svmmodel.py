#!/usr/bin/env python3
"""
FAST Toxic Comment Detector for VS Code (MacBook Air, CPU)
Optimized for speed and stability - no disconnects
"""

import numpy as np
import pandas as pd
import re
import warnings
import os
import joblib
import time

warnings.filterwarnings('ignore')

from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

MBERT_MODEL = 'bert-base-multilingual-cased'
MAX_LENGTH = 128
BATCH_SIZE = 32  # Lower for MacBook Air CPU
DEVICE = torch.device('cpu')  # Force CPU - stable, no quotas

print(f"Device: {DEVICE}")
print("Running on MacBook Air CPU - stable, no disconnects")

# ============================================================
# LOAD DATA
# ============================================================

def load_data_fast(sample_size=1000):
    """Load TextDetox with limited samples"""
    print(f"\n{'='*60}")
    print(f"Loading TextDetox (max {sample_size} samples)...")
    print(f"{'='*60}")
    
    start = time.time()
    
    try:
        ds = load_dataset("textdetox/multilingual_toxicity_dataset", "default")
        
        dfs = []
        for split_name in ds.keys():
            df = pd.DataFrame(ds[split_name])
            df['lang'] = split_name
            dfs.append(df)
            print(f"  {split_name}: {len(df)} samples")
        
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.dropna(subset=['text', 'toxic'])
        combined = combined.drop_duplicates(subset=['text'])
        combined = combined.rename(columns={'toxic': 'label'})
        
        # Stratified sampling
        if sample_size and len(combined) > sample_size:
            _, combined = train_test_split(
                combined, 
                train_size=sample_size, 
                random_state=42, 
                stratify=combined['label']
            )
            combined = combined.reset_index(drop=True)
        
        elapsed = time.time() - start
        print(f"\nLoaded: {len(combined)} samples in {elapsed:.1f}s")
        print(f"Languages: {combined['lang'].nunique()}")
        print(f"Class distribution:\n{combined['label'].value_counts()}")
        
        return combined[['text', 'label', 'lang']].reset_index(drop=True)
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# FEATURE EXTRACTION WITH PROGRESS
# ============================================================

class FastFeatureExtractor:
    def __init__(self):
        print(f"\n{'='*60}")
        print("Loading MBERT...")
        print(f"{'='*60}")
        
        start = time.time()
        self.tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
        self.model = BertModel.from_pretrained(MBERT_MODEL)
        self.model.to(DEVICE)
        self.model.eval()
        
        elapsed = time.time() - start
        print(f"MBERT loaded in {elapsed:.1f}s")
        print(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def extract(self, texts, desc="Extracting"):
        all_embeddings = []
        total = len(texts)
        start = time.time()
        
        print(f"\n{desc} features for {total} samples...")
        print(f"Batch size: {BATCH_SIZE}, Estimated batches: {total // BATCH_SIZE + 1}")
        
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(cls)
            
            # Progress update every 5 batches
            if (i // BATCH_SIZE + 1) % 5 == 0 or i + BATCH_SIZE >= total:
                progress = min(i + BATCH_SIZE, total)
                pct = progress / total * 100
                elapsed = time.time() - start
                rate = progress / elapsed if elapsed > 0 else 0
                eta = (total - progress) / rate if rate > 0 else 0
                print(f"  {progress}/{total} ({pct:.0f}%) | {elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")
        
        total_time = time.time() - start
        print(f"Done! {total_time:.1f}s total ({total/total_time:.1f} samples/sec)")
        
        return np.vstack(all_embeddings)


# ============================================================
# TRAINING
# ============================================================

def train_fast(df):
    print(f"\n{'='*60}")
    print("PREPROCESSING")
    print(f"{'='*60}")
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'].values, df['label'].values,
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Extract features
    extractor = FastFeatureExtractor()
    
    X_train_feat = extractor.extract(X_train.tolist(), "Training")
    X_test_feat = extractor.extract(X_test.tolist(), "Testing")
    
    # Scale
    print(f"\n{'='*60}")
    print("Scaling features...")
    print(f"{'='*60}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    
    # Train SVM
    print(f"\n{'='*60}")
    print("Training SVM (single, fast - no GridSearch)")
    print(f"{'='*60}")
    
    start = time.time()
    svm = SVC(
        C=1, kernel='rbf', gamma='scale',
        class_weight='balanced', probability=True
    )
    svm.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    print(f"SVM trained in {train_time:.1f}s")
    
    # Evaluate
    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    
    package = {
        'svm_model': svm,
        'scaler': scaler,
        'mbert_model_name': MBERT_MODEL,
        'max_length': MAX_LENGTH
    }
    save_path = 'svm_mbert_toxic_detector.pkl'
    joblib.dump(package, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved: {save_path}")
    print(f"File size: {size_mb:.1f} MB")
    
    return extractor, scaler, svm


# ============================================================
# PREDICTOR
# ============================================================

class ToxicPredictor:
    def __init__(self, model_path='svm_mbert_toxic_detector.pkl'):
        print(f"\n{'='*60}")
        print("Loading predictor...")
        print(f"{'='*60}")
        
        pkg = joblib.load(model_path)
        self.svm = pkg['svm_model']
        self.scaler = pkg['scaler']
        self.max_length = pkg['max_length']
        
        self.tokenizer = BertTokenizer.from_pretrained(pkg['mbert_model_name'])
        self.mbert = BertModel.from_pretrained(pkg['mbert_model_name'])
        self.mbert.to(DEVICE)
        self.mbert.eval()
        
        print("Predictor ready!")
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        cleaned = [clean_text(t) for t in texts]
        features = []
        
        for i in range(0, len(cleaned), BATCH_SIZE):
            batch = cleaned[i:i + BATCH_SIZE]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors='pt')
            
            with torch.no_grad():
                out = self.mbert(enc['input_ids'].to(DEVICE),
                                enc['attention_mask'].to(DEVICE))
                features.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        
        features = np.vstack(features)
        features = self.scaler.transform(features)
        
        preds = self.svm.predict(features)
        probs = self.svm.predict_proba(features)[:, 1]
        
        results = []
        for text, pred, prob in zip(texts, preds, probs):
            results.append({
                'text': text[:80] + '...' if len(text) > 80 else text,
                'prediction': int(pred),
                'confidence': float(prob if pred == 1 else 1 - prob),
                'label': 'TOXIC' if pred == 1 else 'NON-TOXIC'
            })
        return results


# ============================================================
# MAIN
# ============================================================

def main():
    total_start = time.time()
    
    print(f"{'='*60}")
    print("TOXIC COMMENT DETECTOR - VS CODE EDITION")
    print("MacBook Air CPU - Stable, No Disconnects")
    print(f"{'='*60}")
    
    # Adjust sample_size: 500=very fast, 1000=balanced, 2000=more accurate
    df = load_data_fast(sample_size=1000)
    
    # Train
    extractor, scaler, svm = train_fast(df)
    
    # Test
    print(f"\n{'='*60}")
    print("MULTILINGUAL TEST")
    print(f"{'='*60}")
    
    predictor = ToxicPredictor()
    
    tests = [
        ("EN toxic", "You are a complete moron and nobody likes you"),
        ("EN safe", "Thank you for sharing this helpful information"),
        ("ES toxic", "Eres un idiota estúpido, vete a la mierda"),
        ("ES safe", "Me encantó este artículo, muy bien escrito"),
        ("FR toxic", "T'es vraiment un connard fini"),
        ("FR safe", "C'est un excellent article, merci"),
        ("DE toxic", "Du bist ein dummer Idiot, verschwinde"),
        ("DE safe", "Sehr guter Artikel, danke für die Information"),
        ("AR toxic", "أنت غبي جداً، اخرس يا قذارة"),
        ("AR safe", "مقال رائع، شكرا على المعلومات المفيدة"),
        ("RU toxic", "Ты тупой идиот, проваливай отсюда"),
        ("RU safe", "Отличная статья, спасибо за информацию"),
    ]
    
    print(f"\n{'LANG':<10} {'RESULT':<10} {'CONF':<6} TEXT")
    print("-" * 70)
    for lang, text in tests:
        r = predictor.predict([text])[0]
        print(f"{lang:<10} {r['label']:<10} {r['confidence']:.3f}  {r['text']}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"COMPLETE! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Model saved: svm_mbert_toxic_detector.pkl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()