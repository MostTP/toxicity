#!/usr/bin/env python3
from pathlib import Path
import sys

try:
    from app.config import settings
    from app.services.model_loader import model_manager
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

print("=" * 50)
print("MODEL PATH CHECK")
print("=" * 50)

paths = {
    "SVM": settings.svm_path,
    "mBERT": settings.mbert_path,
    "CNN": settings.cnn_path,
}

for name, path in paths.items():
    exists = path.exists()
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {name}: {path}")
    if exists and path.is_dir():
        files = list(path.iterdir())
        print(f"   Contains {len(files)} files: {[f.name for f in files[:5]]}{'...' if len(files) > 5 else ''}")
    elif exists:
        print(f"   Size: {path.stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 50)
print("MODEL LOADING CHECK")
print("=" * 50)

model_manager.load_all()

for model_name, status in model_manager.status["models"].items():
    loaded = status == "loaded"
    symbol = "✅" if loaded else "❌"
    print(f"{symbol} {model_name}: {status}")

print("\n" + "=" * 50)
print(f"Device: {model_manager.status['device']}")
print(f"GPU Available: {model_manager.status['gpu_available']}")
print("=" * 50)