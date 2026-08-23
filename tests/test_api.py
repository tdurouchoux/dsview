import asyncio
import logging
import threading
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from dsview import api
from dsview.db.ingest import ContentAlreadyExists


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "check_db_connection", lambda session: None)
    monkeypatch.setattr(api, "async_pull_changes", AsyncMock())
    monkeypatch.setattr(api, "async_upload_changes", AsyncMock())
    # /ingest always checks for an existing link before doing anything else - default
    # to "no duplicate" so tests that aren't about the duplicate check don't hit a
    # real DB query through the faked (None) session.
    monkeypatch.setattr(
        api.ingest_pipeline, "get_existing_content", lambda link, session: None
    )
    # /ingest now commits the content row before returning 202 and the background
    # task re-reads it by id; stub both so no real DB access happens.
    monkeypatch.setattr(
        api.ingest_pipeline, "register_content", lambda content, session: content
    )
    monkeypatch.setattr(
        api,
        "get_content_by_id",
        lambda content_id, session: type(
            "FakeContent", (), {"id": content_id, "link": "https://example.com/post"}
        )(),
    )
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
        "upload_date": datetime.now(UTC).date().isoformat(),
    }


def test_ingest_returns_202_immediately(client):
    # /ingest is fully fire-and-forget now - it never waits on the pipeline, so it
    # always returns 202 regardless of how long ingestion takes.
    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 202
    assert response.json()["status"] == "processing"


def test_ingest_completes_in_background_and_uploads_vault_changes(client, monkeypatch):
    # The vault push only happens when a github vault is configured, which is a
    # local-config detail; pin it so the test doesn't depend on config/storage.yaml.
    monkeypatch.setattr(
        api.vault_config.github_vault, "repository", "owner/dsview_vault"
    )

    done = threading.Event()

    async def slow_ingest(content, session, already_saved=False):
        await asyncio.sleep(0.05)

    async def upload_and_signal(commit_message):
        done.set()

    api.ingest_pipeline.async_ingest_content.side_effect = slow_ingest
    api.async_upload_changes.side_effect = upload_and_signal

    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 202
    assert len(api._background_ingest_tasks) >= 1

    assert done.wait(timeout=2.0), "background task never completed"
    api.async_upload_changes.assert_awaited_once()


def test_ingest_duplicate_link_returns_409_without_starting_background_task(client):
    fake_content = type("FakeContent", (), {"id": 1})()
    api.ingest_pipeline.get_existing_content = lambda link, session: fake_content

    response = client.post("/ingest", json=_input_content("https://example.com/post"))

    assert response.status_code == 409
    api.ingest_pipeline.async_ingest_content.assert_not_awaited()


def test_log_background_ingest_result_logs_any_exception(caplog):
    # The duplicate-content check now happens synchronously before the background
    # task is even created, so ContentAlreadyExists reaching the background task is
    # just a rare race condition - no longer special-cased/suppressed, it's logged
    # as an error like any other background failure.
    async def failing():
        raise ContentAlreadyExists("https://example.com/post")

    async def run_to_completion():
        task = asyncio.create_task(failing())
        with pytest.raises(ContentAlreadyExists):
            await task
        return task

    task = asyncio.run(run_to_completion())

    with caplog.at_level(logging.ERROR):
        api._log_background_ingest_result(task)

    assert "Background ingest task failed" in caplog.text


def test_ingest_status_unknown_when_link_was_never_submitted(client):
    # /ingest commits the content row before returning, so a missing row can only
    # mean this link was never submitted.
    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 404
    assert response.json()["status"] == "unknown"


def test_ingest_status_pending_while_ingestion_is_running(client, monkeypatch):
    fake_content = type("FakeContent", (), {"id": 1})()

    monkeypatch.setattr(
        api.ingest_pipeline, "get_existing_content", lambda link, session: fake_content
    )
    monkeypatch.setattr(api, "get_failed_ingestion", lambda content_id, session: None)
    monkeypatch.setattr(
        api, "get_content_extraction", lambda content_id, session: [[], [], []]
    )

    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 202
    assert response.json() == {"status": "pending"}


def test_ingest_status_success_when_extraction_exists(client, monkeypatch):
    fake_content = type("FakeContent", (), {"id": 1})()

    monkeypatch.setattr(
        api.ingest_pipeline, "get_existing_content", lambda link, session: fake_content
    )
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
    stale_failure = type("FakeFailure", (), {"error_message": "old"})()

    monkeypatch.setattr(
        api.ingest_pipeline, "get_existing_content", lambda link, session: fake_content
    )
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
    fake_failure = type("FakeFailure", (), {"error_message": "404"})()

    monkeypatch.setattr(
        api.ingest_pipeline, "get_existing_content", lambda link, session: fake_content
    )
    monkeypatch.setattr(
        api, "get_failed_ingestion", lambda content_id, session: fake_failure
    )
    monkeypatch.setattr(
        api, "get_content_extraction", lambda content_id, session: [[], [], []]
    )

    response = client.get("/ingest/status", params={"link": "https://example.com/post"})

    assert response.status_code == 200
    assert response.json() == {"status": "failed", "detail": "404"}
