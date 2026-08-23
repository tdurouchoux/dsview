import asyncio
import logging
import subprocess
from functools import wraps
from pathlib import Path

from dsview.config import lazy, load_obsidian_config

config = lazy(load_obsidian_config)
logger = logging.getLogger(__name__)

# Serializes git pull/add/commit/push against config.vault_path so a background
# ingest completion and an overlapping /ingest or /relevance call can't interleave
# git operations on the same working-copy checkout.
vault_lock = asyncio.Lock()


class CommandFailed(Exception):
    def __init__(self, cmd_label: str, stdout: str, stderr: str):
        super().__init__(
            f"Command {cmd_label} failed with stdout : '{stdout}' and stderr '{stderr}'"
        )


def run_cmd(cmd: list[str], cmd_label: str, ignore_error: bool = False, **cmd_kwargs):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,  # Captures both stdout and stderr
            text=True,  # Decodes to str instead of bytes
            check=True,  # Raises CalledProcessError if non-zero exit code
            **cmd_kwargs,
        )
        logger.info("Command '%s' ran successfully", cmd_label)
        logger.debug("with stdout : %s", result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error("Failed to run command '%s' with stderr : %s", cmd_label, e.stderr)

        if not ignore_error:
            raise CommandFailed(cmd_label, e.stdout, e.stderr)


def clone_vault():
    logger.info("Github vault enabled, cloning repository ...")

    run_cmd(["git", "clone", config.github_vault.url, config.vault_path], "clone vault")

    run_cmd(
        ["git", "config", "user.name", config.github_vault.username],
        "git username configuration",
        cwd=Path(config.vault_path).resolve(),
    )
    run_cmd(
        ["git", "config", "user.email", config.github_vault.email],
        "git user email configuration",
        cwd=Path(config.vault_path).resolve(),
    )


def init_vault():

    if config.vault_path.exists():
        logger.info("Vault already exists")
        return

    if config.github_vault.repository is not None:
        clone_vault()
    else:
        config.vault_path.mkdir()

    (config.vault_path / config.content_directory).mkdir(exist_ok=True)
    (config.vault_path / config.topic_directory).mkdir(exist_ok=True)


def pull_changes():
    run_cmd(["git", "pull"], "Pull vault", cwd=config.vault_path)


def upload_changes(commit_message: str):
    run_cmd(["git", "add", "."], "Adding changes", cwd=config.vault_path)
    run_cmd(
        ["git", "commit", "-am", commit_message],
        "Commit changes",
        cwd=config.vault_path,
    )

    run_cmd(
        ["git", "push", config.github_vault.url, "main"],
        "Push changes",
        cwd=config.vault_path,
    )


async def async_pull_changes():
    async with vault_lock:
        await asyncio.to_thread(pull_changes)


async def async_upload_changes(commit_message: str):
    async with vault_lock:
        await asyncio.to_thread(upload_changes, commit_message)


def api_sync_vault(function: callable) -> callable:
    @wraps(function)
    async def function_with_sync(*args, **kwargs):
        if config.github_vault.repository is None:
            return await function(*args, **kwargs)

        await async_pull_changes()

        await function(*args, **kwargs)

        await async_upload_changes("Adding content")

    return function_with_sync
