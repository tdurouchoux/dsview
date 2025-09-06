from functools import wraps
from subprocess import run

from dsview.config import load_obsidian_config

config = load_obsidian_config()


def pull_changes():
    run(["git", "pull"], cwd=config.vault_path)


def upload_changes(commit_message: str):
    run(["git", "add", "."], cwd=config.vault_path)
    run(["git", "commit", "-am", commit_message], cwd=config.vault_path)

    run(["git", "push", config.github_vault.url, "main"], cwd=config.vault_path)


def api_sync_vault(function: callable) -> callable:
    @wraps(function)
    async def function_with_sync(*args, **kwargs):
        if config.github_vault.repository is None:
            return await function(*args, **kwargs)

        pull_changes()

        await function(*args, **kwargs)

        upload_changes("Adding content")

    return function_with_sync
