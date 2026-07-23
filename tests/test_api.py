from datetime import date
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import dsview.api as api
from dsview.db.ingest import ContentAlreadyExists


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "check_db_connection", lambda session: None)
    monkeypatch.setattr(api, "pull_changes", lambda: None)
    monkeypatch.setattr(api, "upload_changes", lambda commit_message: None)
    monkeypatch.setattr(
        api.ingest_pipeline, "async_ingest_content", AsyncMock(return_value=None)
    )

    api.app.dependency_overrides[api.get_session] = lambda: iter([None])

    with TestClient(api.app) as test_client:
        yield test_client

    api.app.dependency_overrides.clear()


def _input_content(link: str) -> dict:
    return {
        "link": link,
        "upload_date": date.today().isoformat(),
    }


def test_ingest_non_pdf_link_runs_synchronously(client):
    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 200
    api.ingest_pipeline.async_ingest_content.assert_awaited_once()


def test_ingest_pdf_link_runs_in_background(client):
    response = client.post(
        "/ingest", json=_input_content("https://example.com/paper.pdf")
    )

    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    api.ingest_pipeline.async_ingest_content.assert_awaited_once()


def test_ingest_non_pdf_failure_returns_500_with_error_detail(client):
    api.ingest_pipeline.async_ingest_content.return_value = ValueError("boom")

    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error_type": "ValueError",
        "error_message": "boom",
    }


def test_ingest_duplicate_link_returns_409(client):
    api.ingest_pipeline.async_ingest_content.side_effect = ContentAlreadyExists(
        "https://example.com/post"
    )

    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 409


def test_ingest_status_pending_when_content_unknown(client, monkeypatch):
    monkeypatch.setattr(api, "get_content", lambda link, session: None)

    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 200
    assert response.json() == {"status": "pending"}


def test_ingest_status_success_when_extraction_exists(client, monkeypatch):
    fake_content = type("FakeContent", (), {"id": 1})()

    monkeypatch.setattr(api, "get_content", lambda link, session: fake_content)
    monkeypatch.setattr(api, "get_failed_ingestion", lambda content_id, session: None)
    monkeypatch.setattr(
        api, "get_content_extraction", lambda content_id, session: [["result"], [], []]
    )

    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 200
    assert response.json() == {"status": "success"}


def test_ingest_status_failed_when_failed_ingestion_exists(client, monkeypatch):
    fake_content = type("FakeContent", (), {"id": 1})()
    fake_failure = type(
        "FakeFailure",
        (),
        {"error_type": "WebRequestFailure", "error_message": "404"},
    )()

    monkeypatch.setattr(api, "get_content", lambda link, session: fake_content)
    monkeypatch.setattr(
        api, "get_failed_ingestion", lambda content_id, session: fake_failure
    )

    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "error_type": "WebRequestFailure",
        "error_message": "404",
    }
