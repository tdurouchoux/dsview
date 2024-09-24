from abc import ABC, abstractmethod
import logging
from pathlib import Path
import requests
from typing import Union

from bs4 import BeautifulSoup
from pydantic import HttpUrl
from pypdf import PdfReader

from dsview.obsidian.obsidian_utils import get_pdf_filepath


logger = logging.getLogger(__name__)


class WebRequestFailure(Exception):
    def __init__(self, url: str, status_code: int) -> None:
        super().__init__(f"Request for url {url} failed with status code {status_code}")


class ContentLoader(ABC):
    def __init__(self, link: Union[Path, HttpUrl], token_limit: int) -> None:
        self.link = link
        self.token_limit = token_limit
        self.content: str = None

    @abstractmethod
    def _load_content(self):
        pass

    @abstractmethod
    def get_hyperlink(self) -> str:
        pass

    def load(self):
        logger.info("Loading content with %s", self.__class__.__name__)

        if self.content is None:
            self._load_content()


class TextLoader(ContentLoader):
    def __init__(self, link: Path, token_limit: int) -> None:
        super().__init__(link, token_limit)

    def get_hyperlink(self) -> str:
        return str(self.link.absolute())

    def _load_content(self):
        with open(self.link, "r") as content_file:
            self.content = content_file.read()


class WebContentLoader(ContentLoader):
    def __init__(self, link: HttpUrl, token_limit: int) -> None:
        super().__init__(link, token_limit)

    def _request_url(self) -> requests.Response:
        response = requests.get(self.link)

        if response.status_code != 200:
            raise WebRequestFailure(self.link, response.status_code)

        return response


class UrlLoader(WebContentLoader):
    def __init__(self, link: HttpUrl, token_limit: int) -> None:
        super().__init__(link, token_limit)

        self.content_soup = None
        self.content_links = None

    def _load_content(self):
        response = self._request_url()

        self.content_soup = BeautifulSoup(response.content, "html.parser")

        if self.link == "readmediu.comm":
            logger.info("Received a link from readmedium, ignoring included summary.")

            for line in self.content_soup.find_all(class_="!my-2"):
                line.decompose()

        self.content = self.content_soup.get_text()

        # Extracting links
        self.content_links = []
        all_content_links = self.content_soup.find_all("a")
        for content_link in all_content_links:
            content_link_url = content_link.get("href")
            if content_link_url is not None and content_link_url.startswith("http"):
                self.content_links.append(str(content_link))

    def get_hyperlink(self) -> str:
        return str(self.link)


class PdfUrlLoader(WebContentLoader):
    def __init__(self, link: HttpUrl, token_limit: int) -> None:
        super().__init__(link, token_limit)
        self.pdf_filepath = get_pdf_filepath(self.link.path.split("/")[-1])

    def _load_content(self):
        logger.info("Saving pdf file at path : %s", self.pdf_filepath)
        response = self._request_url()

        with open(self.pdf_filepath, "wb") as pdf_file:
            pdf_file.write(response.content)

        reader = PdfReader(self.pdf_filepath)

        self.content = ""
        word_count = 0
        word_limit = self.token_limit / 2

        for i, page in enumerate(reader.pages):
            page_content = page.extract_text()
            word_count += len(page_content.split(" "))

            if word_count > word_limit:
                logger.warning(
                    (
                        "Pdf document is too large, word limit is set at %s. "
                        "Stopped at page %s out of %s."
                    ),
                    word_limit,
                    i + 1,
                    len(reader.pages),
                )
                break

            self.content += page_content

    def get_hyperlink(self) -> str:
        return f"![]({'/'.join(self.pdf_filepath.parts[-2:])})"


def get_content_loader(link: Union[HttpUrl, Path], token_limit) -> ContentLoader:
    if isinstance(link, Path):
        return TextLoader(link, token_limit)

    # ! Improve pdf detection
    if link.path.endswith(".pdf") or (
        link.host == "arxiv.org" and link.path.startswith("/pdf/")
    ):
        return PdfUrlLoader(link, token_limit)
    return UrlLoader(link, token_limit)
