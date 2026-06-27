FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create data directory
RUN mkdir -p /app/data

# Environment defaults
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models
ENV DB_PATH=/app/data/toxicity.db
ENV REDIS_URL=""
ENV MAX_BATCH=100
ENV LOG_LEVEL=info
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
