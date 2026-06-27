"""API endpoint tests."""

import pytest
from httpx import AsyncClient

from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data


@pytest.mark.asyncio
async def test_predict_no_models(client):
    """Test prediction when models aren't loaded."""
    response = await client.post("/predict", json={
        "text": "this is a test comment",
        "model": "auto"
    })
    # Should return 503 if no models available, or success if mocked
    assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_predict_validation(client):
    """Test input validation."""
    response = await client.post("/predict", json={
        "text": "",  # Empty text - should fail
        "model": "auto"
    })
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_predict(client):
    response = await client.post("/predict/batch", json={
        "texts": ["comment one", "comment two"],
        "model": "auto"
    })
    assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_feedback_not_found(client):
    response = await client.post("/admin/feedback", json={
        "request_id": "non-existent-uuid",
        "correct_label": 1
    })
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_stats(client):
    response = await client.get("/admin/stats")
    assert response.status_code == 200
    data = response.json()
    assert "requests" in data
    assert "drift" in data


@pytest.mark.asyncio
async def test_models_list(client):
    response = await client.get("/admin/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 3
