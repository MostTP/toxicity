
# Diagnostic script to check SVM label orientation
# Save this as check_svm.py and run it

code = '''
import pickle
import numpy as np
import sys

# Load your SVM
with open("app/models/svm/svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

print("=== SVM DIAGNOSTIC ===")
print(f"Type: {type(svm)}")
print(f"Classes: {svm.classes_}")
print(f"n_features_in_: {svm.n_features_in_}")

# Check if it's SGDClassifier
if hasattr(svm, 'loss'):
    print(f"Loss: {svm.loss}")
    print(f"Penalty: {svm.penalty}")

# Test with random embeddings of correct dimension
dim = svm.n_features_in_
print(f"\\nExpected embedding dimension: {dim}")

# Create test embeddings
np.random.seed(42)
neutral_emb = np.random.randn(1, dim) * 0.1
toxic_emb = np.random.randn(1, dim) * 2.0  # More extreme = "more toxic"

neutral_dec = svm.decision_function(neutral_emb)
toxic_dec = svm.decision_function(toxic_emb)

print(f"\\nNeutral embedding decision: {neutral_dec[0]:.4f}")
print(f"Toxic-like embedding decision: {toxic_dec[0]:.4f}")

# The key question: which class does positive decision map to?
print(f"\\nClass mapping:")
print(f"  decision > 0 → class {svm.classes_[1]}")
print(f"  decision < 0 → class {svm.classes_[0]}")

# Check if classes are [0,1] or [1,0] or strings
if list(svm.classes_) == [0, 1] or list(svm.classes_) == ['non-toxic', 'toxic']:
    print("\\n✓ Standard orientation: positive = class 1 (toxic)")
    print("  If toxic text gives negative decisions, labels are INVERTED")
elif list(svm.classes_) == [1, 0]:
    print("\\n⚠ INVERTED orientation: positive = class 0 (non-toxic)")
    print("  You need to flip the decision function or probabilities")
else:
    print(f"\\n? Unusual class order: {svm.classes_}")
'''
print(code)
