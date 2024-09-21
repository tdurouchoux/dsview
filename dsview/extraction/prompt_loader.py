import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class MissingPromptFile(Exception):
    def __init__(self, prompt_filename: str):
        super().__init__(
            self,
            (
                f"Prompt file '{prompt_filename}' is missing "
                f"from prompt directory : {os.getenv('PROMPT_DIR')}."
            ),
        )


def get_prompt(prompt_filename: str) -> str:
    prompt_file = Path(os.getenv("PROMPT_DIR")) / prompt_filename

    if not prompt_file.exists():
        raise MissingPromptFile(prompt_filename)

    with open(prompt_file, "r") as txt_file:
        prompt = txt_file.read()

    return prompt
