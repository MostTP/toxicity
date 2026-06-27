# Toxicity Detection API

Multilingual toxic comment classification using **mBERT**, **SVM**, and **CNN** with subword tokenization.

## Architecture

```
┌─────────┐     ┌─────────────┐     ┌─────────────────┐
│  Client │────▶│  FastAPI    │────▶│  SVM (CPU)      │──┐
│ (React) │◄────│   Service   │◄────│  mBERT (GPU)    │◄─┘
│         │     │             │     │  CNN (GPU)      │
└─────────┘     └──────┬──────┘     └─────────────────┘
                       │
              ┌────────┴────────┐
              │   SQLite        │  (predictions + feedback)
              │   Redis (opt)   │  (cache)
              └─────────────────┘
```

## Quick Start

```bash
# 1. Build and run
docker-compose up --build

# 2. Check health
curl http://localhost:8000/health

# 3. Classify a comment
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are stupid", "model": "auto"}'

# 4. Submit feedback
curl -X POST http://localhost:8000/admin/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "...", "correct_label": 1}'
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single comment classification |
| `/predict/batch` | POST | Bulk classification (max 100) |
| `/health` | GET | System + model status |
| `/admin/feedback` | POST | Submit prediction correction |
| `/admin/stats` | GET | Usage stats + drift detection |
| `/admin/models` | GET | Available models |

## Model Routing (Auto Mode)

1. **SVM** runs first (~5ms on CPU)
2. If confidence is clear (>0.92 or <0.08), return immediately
3. If uncertain, escalate to **mBERT** (~40ms on GPU)
4. If mBERT fails, fallback to **CNN**

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `./models` | Path to model artifacts |
| `DB_PATH` | `./data/toxicity.db` | SQLite database path |
| `REDIS_URL` | `None` | Redis cache (optional) |
| `MAX_BATCH` | `100` | Max texts per batch |
| `THRESHOLDS` | `{"default": 0.5, ...}` | Per-language thresholds |

## Model Artifacts

Place trained models in:

```
models/
├── svm/svm_model.pkl          # pickle'd SGDClassifier
├── mbert/                     # HuggingFace model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
└── cnn/cnn_best.pt            # PyTorch state_dict
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run locally
uvicorn app.main:app --reload
```

## License

MIT
