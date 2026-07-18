import logging
import tempfile
from abc import ABC, abstractmethod

import requests
from bs4 import BeautifulSoup
from pydantic import HttpUrl
from pypdf import PdfReader

from dsview.config import load_model_config

logger = logging.getLogger(__name__)


class WebRequestFailure(Exception):
    def __init__(self, url: str, status_code: int) -> None:
        super().__init__(f"Request for url {url} failed with status code {status_code}")


class ContentLoader(ABC):
    def __init__(self, link: HttpUrl) -> None:
        self.link = link
        self.content: str = None

    @abstractmethod
    def _load_content(self):
        pass

    def load(self):
        logger.info("Loading content with %s", self.__class__.__name__)

        if self.content is None:
            self._load_content()


class WebContentLoader(ContentLoader):
    def __init__(self, link: HttpUrl) -> None:
        super().__init__(link)

    def _request_url(self) -> requests.Response:
        response = requests.get(self.link)

        if response.status_code != 200:
            raise WebRequestFailure(self.link, response.status_code)

        return response


class UrlLoader(WebContentLoader):
    def __init__(self, link: HttpUrl) -> None:
        super().__init__(link)

        self.content_soup = None
        self.content_links = None

    def _load_content(self):
        response = self._request_url()

        self.content_soup = BeautifulSoup(response.content, "html.parser")

        if self.link.host == "readmedium.com":
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
                # Check for difference in performance
                # self.content_links.append(str(content_link))
                self.content_links.append(content_link_url)

        self.content_links = list(set(self.content_links))


class PdfUrlLoader(WebContentLoader):
    def __init__(self, link: HttpUrl) -> None:
        super().__init__(link)

    def _load_content(self):
        response = self._request_url()

        with tempfile.NamedTemporaryFile() as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf.flush()

            reader = PdfReader(temp_pdf.name)
            self._extract_pdf_content(reader)

    def _extract_pdf_content(self, reader: PdfReader):
        self.content = ""
        word_count = 0
        word_limit = load_model_config().token_limit / 2

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


# ? How to deal with token_limit


def get_content_loader(link: HttpUrl) -> ContentLoader:
    # ! Improve pdf detection
    if link.path.endswith(".pdf") or (
        link.host == "arxiv.org" and link.path.startswith("/pdf/")
    ):
        return PdfUrlLoader(link)
    return UrlLoader(link)
