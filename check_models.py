import sys
sys.path.insert(0, '.')

from app.services.model_loader import model_manager

print("=== MODEL LOADER DIAGNOSTIC ===")
print(f"svm_tokenizer exists: {model_manager.svm_tokenizer is not None}")
print(f"svm_mbert_base exists: {model_manager.svm_mbert_base is not None}")
print(f"svm exists: {model_manager.svm is not None}")
print(f"tokenizer exists: {model_manager.tokenizer is not None}")
print(f"mbert_base exists: {model_manager.mbert_base is not None}")

if model_manager.svm_tokenizer:
    print(f"\nSVM tokenizer vocab size: {len(model_manager.svm_tokenizer)}")
if model_manager.tokenizer:
    print(f"Main tokenizer vocab size: {len(model_manager.tokenizer)}")

# Test encoding same text with both
text = "You are an idiot"
if model_manager.svm_tokenizer:
    svm_tokens = model_manager.svm_tokenizer.encode(text)
    print(f"\nSVM tokenizer tokens: {svm_tokens[:10]}...")
if model_manager.tokenizer:
    main_tokens = model_manager.tokenizer.encode(text)
    print(f"Main tokenizer tokens: {main_tokens[:10]}...")

print("\nStatus:", model_manager.status)