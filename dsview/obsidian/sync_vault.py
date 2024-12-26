from functools import wraps
from subprocess import run

from dsview.config import load_obsidian_config

config = load_obsidian_config()


def api_sync_vault(function: callable) -> callable:
    @wraps(function)
    async def function_with_sync(*args, **kwargs):
        if config.github_vault.repository is None:
            return await function(*args, **kwargs)

        run(["git", "pull"], cwd=config.vault_path)

        result = await function(*args, **kwargs)

        run(["git", "add", "."], cwd=config.vault_path)
        run(["git", "commit", "-am", "test"], cwd=config.vault_path)

        remote = (
            f"https://{config.github_vault.username}:{config.github_vault.token}"
            f"@{config.github_vault.repository.replace('https://', '')}"
        )
        run(["git", "push", remote, "main"], cwd=config.vault_path)

        return result

    return function_with_sync
