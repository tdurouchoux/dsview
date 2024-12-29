from functools import wraps
from subprocess import run

from dsview.config import load_obsidian_config

config = load_obsidian_config()

def pull_changes():
    run(["git", "pull"], cwd=config.vault_path)

def upload_changes(commit_message: str):

    run(["git", "add", "."], cwd=config.vault_path)
    run(["git", "commit", "-am", commit_message], cwd=config.vault_path)

    remote = (
        f"https://{config.github_vault.username}:{config.github_vault.token}"
        f"@{config.github_vault.repository.replace('https://', '')}"
    )
    run(["git", "push", remote, "main"], cwd=config.vault_path)

def label_sync_vault(label_type: str):
    def sync_vault(function: callable) -> callable:
        @wraps(function)
        def function_with_sync(*args, **kwargs):
            if config.github_vault.repository is None:
                function(*args, **kwargs)

            pull_changes()

            result = function(*args, **kwargs)

            upload_changes(f"Adding one {label_type} label")

            return result

        return function_with_sync

    return sync_vault

def api_sync_vault(function: callable) -> callable:
    @wraps(function)
    async def function_with_sync(*args, **kwargs):
        if config.github_vault.repository is None:
            return await function(*args, **kwargs)

        pull_changes()

        await function(*args, **kwargs)

        upload_changes("Adding content")

    return function_with_sync
