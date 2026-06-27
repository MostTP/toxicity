# train_svm_from_embeddings.py
import pickle
import numpy as np
from sklearn.svm import SVC
from pathlib import Path

# Load your embeddings
with open("app/models/svm/svm_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# Handle different formats
if isinstance(data, dict):
    X = data.get("X") or data.get("embeddings") or data.get("features")
    y = data.get("y") or data.get("labels") or data.get("targets")
elif isinstance(data, tuple) and len(data) == 2:
    X, y = data
else:
    raise ValueError(f"Unknown data format: {type(data)}")

print(f"Loaded {len(X)} samples, {len(X[0])} features each")
print(f"Labels: {np.unique(y)}")

# Train SVM
print("Training SVM...")
svm = SVC(kernel="rbf", probability=True, C=1.0)
svm.fit(X, y)

# Save trained model
Path("app/models/svm").mkdir(parents=True, exist_ok=True)
with open("app/models/svm/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("✅ SVM saved to app/models/svm/svm_model.pkl")