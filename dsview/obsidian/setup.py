import logging
from pathlib import Path
from subprocess import run

from dsview import config

config.setup_logger()
logger = logging.getLogger("setup_vault")

obsidian_config = config.load_obsidian_config()


def init_vault():
    logger.info("Initializing vault...")
    if obsidian_config.github_vault.repository is not None:
        logger.info("Github vault enabled, cloning repository ...")
        run(["git", "clone", obsidian_config.github_vault.repository])

        directory_name = obsidian_config.github_vault.repository.split("/")[-1].replace(
            ".git", ""
        )

        run(
            ["git", "config", "user.name", obsidian_config.github_vault.username],
            cwd=Path(directory_name).resolve(),
        )
        run(
            ["git", "config", "user.email", obsidian_config.github_vault.email],
            cwd=Path(directory_name).resolve(),
        )

        Path(directory_name).rename(obsidian_config.vault_path)

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
