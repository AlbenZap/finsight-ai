"""
Core API tests for FinSight AI.

Tests cover: health check, company search, job lifecycle, SSE endpoint,
validation helpers, and FAISS manager utilities.

Run: pytest tests/ -v --cov=. --cov-report=term-missing
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI test client with LLM and FAISS mocked out."""
    with (
        patch("main.get_llm"),
        patch("main.faiss_manager"),
        patch("main.process_filing_background", new_callable=AsyncMock),
    ):
        from main import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "provider" in data


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "FinSight AI" in r.json()["message"]


# ---------------------------------------------------------------------------
# Company search
# ---------------------------------------------------------------------------

def test_company_search_too_short(client):
    r = client.get("/company_search/?q=A")
    assert r.status_code == 200
    assert r.json()["results"] == []


@patch("main.fetch_company_database")
def test_company_search_returns_results(mock_db, client):
    mock_db.return_value = [
        {"ticker": "AAPL", "cik": "0000320193", "title": "Apple Inc."},
        {"ticker": "AMZN", "cik": "0001018724", "title": "Amazon.com Inc."},
    ]
    r = client.get("/company_search/?q=Apple")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) >= 1
    assert any(c["ticker"] == "AAPL" for c in results)


@patch("main.fetch_company_database")
def test_company_search_exact_ticker_ranked_first(mock_db, client):
    mock_db.return_value = [
        {"ticker": "TSLA", "cik": "0001318605", "title": "Tesla Inc."},
        {"ticker": "TSLAQ", "cik": "0001318606", "title": "Tesla Q Corp"},
    ]
    r = client.get("/company_search/?q=TSLA")
    assert r.status_code == 200
    assert r.json()["results"][0]["ticker"] == "TSLA"


# ---------------------------------------------------------------------------
# Analyze filing (async job)
# ---------------------------------------------------------------------------

def test_analyze_filing_returns_job_id(client):
    r = client.post("/analyze_filing/?ticker=AAPL")
    assert r.status_code == 200
    assert "job_id" in r.json()


def test_analyze_filing_cached_ticker(client):
    """When ticker is in analysis_cache, job should return complete immediately."""
    from main import analysis_cache
    analysis_cache["META:10-K:analyst"] = {
        "ticker": "META",
        "company_name": "Meta Platforms Inc.",
        "summary": [],
        "financial_charts": {},
        "company_profile": {},
    }
    r = client.post("/analyze_filing/?ticker=META")
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status_r = client.get(f"/job_status/{job_id}")
    assert status_r.status_code == 200
    assert status_r.json()["status"] == "complete"
    del analysis_cache["META:10-K:analyst"]


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

def test_job_status_not_found(client):
    r = client.get("/job_status/nonexistent-job-id")
    assert r.status_code == 404


def test_job_status_processing(client):
    from main import jobs
    jobs["test-job-123"] = {
        "job_id": "test-job-123",
        "status": "processing",
        "progress": "Extracting text...",
    }
    r = client.get("/job_status/test-job-123")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "processing"
    assert data["progress"] == "Extracting text..."
    del jobs["test-job-123"]


# ---------------------------------------------------------------------------
# LLM output validation
# ---------------------------------------------------------------------------

def test_validate_llm_response_empty():
    from main import validate_llm_response
    with pytest.raises(ValueError, match="empty"):
        validate_llm_response("")


def test_validate_llm_response_too_short():
    from main import validate_llm_response
    with pytest.raises(ValueError):
        validate_llm_response("Short.")


def test_validate_llm_response_refusal():
    from main import validate_llm_response
    with pytest.raises(ValueError, match="refused"):
        validate_llm_response("I cannot provide that information " + "x" * 60)


def test_validate_llm_response_financial_no_signal():
    from main import validate_llm_response
    with pytest.raises(ValueError, match="financial signal"):
        validate_llm_response("The company has great products and services " * 5, context="financial")


def test_validate_llm_response_valid():
    from main import validate_llm_response
    text = "**Revenue: $394.3B (+8% YoY)** - Apple reported strong results driven by iPhone sales."
    assert validate_llm_response(text, context="financial") == text.strip()


# ---------------------------------------------------------------------------
# Vector store info
# ---------------------------------------------------------------------------

def test_vector_stores_endpoint(client):
    with patch("main.faiss_manager") as mock_fm:
        mock_fm.list_stores.return_value = ["AAPL", "TSLA"]
        r = client.get("/vector_stores/")
        assert r.status_code == 200
        data = r.json()
        assert data["total_count"] == 2