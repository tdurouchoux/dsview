import logging
from pathlib import Path
import subprocess

from dsview import config

config.setup_logger()
logger = logging.getLogger("setup_vault")

obsidian_config = config.load_obsidian_config()

class CommandFailed(Exception):

    def __init__(self, cmd_label: str, stdout: str, stderr: str):
        super().__init__(
            f"Command {cmd_label} failed with stdout : '{stdout}'"
            f" and stderr '{stderr}'"
        )


def run_cmd(cmd: list[str], cmd_label: str, ignore_error: bool = False, cwd: Path | None = None):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,  # Captures both stdout and stderr
            text=True,            # Decodes to str instead of bytes
            check=True,            # Raises CalledProcessError if non-zero exit code
            cwd=cwd,
        )
        logger.info("Command '%s' ran successfully", cmd_label)
        logger.debug("with stdout : %s", result.stdout)

    except subprocess.CalledProcessError as e:

        logger.error("Failed to run command '%s' with stderr : %s", cmd_label, e.stderr)

        if not ignore_error:
            raise CommandFailed(
                cmd_label,
                e.stdout,
                e.stderr
            )


def init_vault():
    logger.info("Initializing vault...")
    if obsidian_config.github_vault.repository is not None:
        logger.info("Github vault enabled, cloning repository ...")

        run_cmd(["git", "clone", obsidian_config.github_vault.url, obsidian_config.vault_path], "clone vault")

        run_cmd(
            ["git", "config", "user.name", obsidian_config.github_vault.username],
            "git username configuration",
            cwd=Path(obsidian_config.vault_path).resolve(),
        )
        run_cmd(
            ["git", "config", "user.email", obsidian_config.github_vault.email],
            "git user email configuration",
            cwd=Path(obsidian_config.vault_path).resolve(),
        )

        return

    obsidian_config.vault_path.mkdir()


def create_vault_directory(directory: str):
    (obsidian_config.vault_path / directory).mkdir(exist_ok=True)


def main():
    if not obsidian_config.vault_path.exists():
        init_vault()

    create_vault_directory(obsidian_config.content_directory)
    create_vault_directory(obsidian_config.topic_directory)


if __name__ == "__main__":
    main()
