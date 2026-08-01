import asyncio
import time

from dsview.obsidian import sync_vault


def test_async_pull_and_upload_never_interleave(monkeypatch):
    order = []

    def fake_pull():
        order.append("pull-start")
        time.sleep(0.05)
        order.append("pull-end")

    def fake_upload(commit_message):
        order.append("upload-start")
        time.sleep(0.05)
        order.append("upload-end")

    monkeypatch.setattr(sync_vault, "pull_changes", fake_pull)
    monkeypatch.setattr(sync_vault, "upload_changes", fake_upload)

    async def run_both():
        await asyncio.gather(
            sync_vault.async_pull_changes(),
            sync_vault.async_upload_changes("msg"),
        )

    asyncio.run(run_both())

    assert order in (
        ["pull-start", "pull-end", "upload-start", "upload-end"],
        ["upload-start", "upload-end", "pull-start", "pull-end"],
    )
