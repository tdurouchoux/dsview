from abc import ABC, abstractmethod
import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


class ContentLoader(ABC):
    def __init__(self) -> None:
        self.content: str = None

    @abstractmethod
    def _load_content(self):
        pass

    def load(self):
        logger.info("Loading content with %s", self.__class__.__name__)

        if self.content is None:
            self._load_content()


class TextLoader(ContentLoader):
    def __init__(self, content_path: Path) -> None:
        self.content_path = content_path
        super().__init__()

    def _load_content(self):
        with open(self.content_path, "r") as content_file:
            self.content = content_file.read()


class UrlLoader(ContentLoader):
    ...

    # Check if is a pdf
    # Extract soup
    # get page content
    # Get urls > Custom LLM queries


class PdfUrlLoader(ContentLoader): ...


def get_content_loader(content_path: str) -> ContentLoader:
    if Path(content_path).exists():
        return TextLoader(Path(content_path))

    if content_path.endswith(".pdf"):
        return PdfUrlLoader(content_path)
    return UrlLoader(content_path)
