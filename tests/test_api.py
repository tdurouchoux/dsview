import asyncio
import threading
from datetime import date
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import dsview.api as api
from dsview.db.ingest import ContentAlreadyExists


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "check_db_connection", lambda session: None)
    monkeypatch.setattr(api, "async_pull_changes", AsyncMock())
    monkeypatch.setattr(api, "async_upload_changes", AsyncMock())
    monkeypatch.setattr(api, "INGEST_SYNC_TIMEOUT_SECONDS", 0.05)
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


def test_ingest_fast_pipeline_returns_200_synchronously(client):
    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 200
    api.ingest_pipeline.async_ingest_content.assert_awaited_once()


def test_ingest_slow_pipeline_returns_202_then_completes_in_background(client):
    # Same link/content as the fast-path test above - dispatch must depend only
    # on how long the pipeline actually takes, not on the URL shape.
    done = threading.Event()

    async def slow_ingest(content, session):
        await asyncio.sleep(0.2)

    async def upload_and_signal(commit_message):
        done.set()

    api.ingest_pipeline.async_ingest_content.side_effect = slow_ingest
    api.async_upload_changes.side_effect = upload_and_signal

    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    assert len(api._background_ingest_tasks) >= 1

    assert done.wait(timeout=2.0), "background task never completed"
    api.async_upload_changes.assert_awaited_once()


def test_ingest_fast_failure_returns_500_with_error_detail(client):
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


def test_ingest_status_success_even_with_stale_failed_ingestion(client, monkeypatch):
    # Regression test: content that failed once and later succeeded (e.g. via a
    # CLI rebuild-mode retry) must report success, not a permanently stale failure.
    fake_content = type("FakeContent", (), {"id": 1})()
    stale_failure = type(
        "FakeFailure", (), {"error_type": "X", "error_message": "old"}
    )()

    monkeypatch.setattr(api, "get_content", lambda link, session: fake_content)
    monkeypatch.setattr(
        api, "get_failed_ingestion", lambda content_id, session: stale_failure
    )
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
    monkeypatch.setattr(
        api, "get_content_extraction", lambda content_id, session: [[], [], []]
    )

    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "failed",
        "error_type": "WebRequestFailure",
        "error_message": "404",
    }
