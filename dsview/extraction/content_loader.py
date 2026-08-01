import json
import logging
import math
import re
import tempfile
from abc import ABC, abstractmethod
from functools import cache
from typing import TYPE_CHECKING, Callable
from urllib.parse import parse_qs

import requests
import tiktoken
from bs4 import BeautifulSoup
from pydantic import HttpUrl
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from dsview.config import load_model_config

if TYPE_CHECKING:
    # docling pulls in torch/transformers - only imported for real inside the
    # lazily-cached factories below, so processes that never load a PDF (MCP,
    # dashboards, labelling UI unless a PDF link is actually submitted) don't
    # pay that cost just for importing this module.
    from docling.chunking import HybridChunker
    from docling.datamodel.document import ConversionResult
    from docling.document_converter import DocumentConverter

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

    def _request_url(self, url: HttpUrl | None = None) -> requests.Response:
        url = url if url is not None else self.link
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            raise WebRequestFailure(url, response.status_code)

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
    # Chunk granularity for the over-budget sampling path only - independent from the
    # overall document token budget, just controls how finely a chapter can be trimmed.
    CHUNK_MAX_TOKENS = 512

    def __init__(self, link: HttpUrl) -> None:
        super().__init__(link)

    def _load_content(self):
        response = self._request_url()
        self._load_pdf_response(response)

    def _load_pdf_response(self, response: requests.Response):
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf.flush()

            result = self._get_pdf_converter().convert(temp_pdf.name)
            self._extract_pdf_content(result)

    def _extract_pdf_content(self, result: "ConversionResult"):
        token_limit = load_model_config().token_limit / 2
        full_content = result.document.export_to_markdown()

        chunker, count_tokens = self._get_pdf_chunker()

        if count_tokens(full_content) <= token_limit:
            self.content = full_content
            return

        logger.warning(
            "Pdf document is too large, token limit is set at %s. Sampling "
            "representative sections from across the whole document instead of "
            "only keeping the start.",
            token_limit,
        )

        chunks = [
            (chunk.meta.headings or [], chunk.text)
            for chunk in chunker.chunk(result.document)
        ]
        self.content = self._select_representative_content(
            chunks, token_limit, count_tokens
        )

    @staticmethod
    @cache
    def _get_pdf_converter() -> "DocumentConverter":
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    @staticmethod
    @cache
    def _get_pdf_chunker() -> tuple["HybridChunker", Callable[[str], int]]:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.get_encoding("cl100k_base"),
            max_tokens=PdfUrlLoader.CHUNK_MAX_TOKENS,
        )
        return HybridChunker(tokenizer=tokenizer), tokenizer.count_tokens

    @staticmethod
    def _select_representative_content(
        chunks: list[tuple[list[str], str]],
        token_limit: float,
        count_tokens: Callable[[str], int],
    ) -> str:
        """Evenly sample chunks across the whole document instead of only
        keeping the start, so a large document contributes content from every
        chapter. Stops as soon as the running total would exceed token_limit.
        """
        total_tokens = sum(count_tokens(text) for _, text in chunks)
        stride = max(1, math.ceil(total_tokens / token_limit))

        sections = []
        used_tokens = 0
        last_heading = None
        for index, (headings, text) in enumerate(chunks):
            if index % stride != 0:
                continue

            heading = headings[0] if headings else None
            if heading and heading != last_heading:
                text = f"## {heading}\n\n{text}"
                last_heading = heading

            text_tokens = count_tokens(text)
            if used_tokens + text_tokens > token_limit:
                break

            sections.append(text)
            used_tokens += text_tokens

        return "\n\n".join(sections)


class ArxivContentLoader(PdfUrlLoader):
    # Below this length, treat a 200 response as an HTML error page or an empty
    # overview rather than a real alphaXiv overview, and fall back to the PDF path.
    MIN_ALPHAXIV_OVERVIEW_LENGTH = 200

    def __init__(self, link: HttpUrl) -> None:
        super().__init__(link)

        # get_content_loader only dispatches here when this already resolves to a
        # real id, so paper_id is guaranteed non-None.
        self.paper_id: str | None = self.extract_paper_id(link.path)

    @staticmethod
    def extract_paper_id(path: str | None) -> str | None:
        # matches /abs/{id}, /pdf/{id}, /pdf/{id}.pdf, /pdf/{id}v2, an optional trailing
        # slash, and old-style /abs/hep-th/9901001 ids (which contain a slash). Anything
        # else (/html/{id}, /list/...) is not an id this class can handle - dispatch
        # (get_content_loader) uses this same method to fall back to UrlLoader instead.
        match = re.search(r"/(?:abs|pdf)/(.+?)/?(?:\.pdf)?$", path or "")
        if not match:
            return None

        return re.sub(r"v\d+$", "", match.group(1))

    def _load_content(self):
        if self._load_alphaxiv_overview():
            return

        response = self._request_url(HttpUrl(f"https://arxiv.org/pdf/{self.paper_id}"))
        self._load_pdf_response(response)

    def _load_alphaxiv_overview(self) -> bool:
        try:
            response = self._request_url(
                HttpUrl(f"https://www.alphaxiv.org/overview/{self.paper_id}.md")
            )
        except WebRequestFailure:
            return False

        content = response.text.strip()
        if len(content) < self.MIN_ALPHAXIV_OVERVIEW_LENGTH:
            return False

        self.content = content
        return True


class YoutubeTranscriptUnavailable(Exception):
    def __init__(self, video_id: str, reason: str) -> None:
        super().__init__(
            f"No transcript available for YouTube video {video_id}: {reason}"
        )


class JsonExtractionError(Exception):
    def __init__(self, marker: str, error: json.JSONDecodeError) -> None:
        super().__init__(f"Failed to parse JSON after marker '{marker}': {error}")


class YoutubeContentLoader(WebContentLoader):
    TRANSCRIPT_LANGUAGES = ["en", "fr"]
    WATCH_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
    SHORT_HOST = "youtu.be"

    def __init__(self, link: HttpUrl) -> None:
        super().__init__(link)

        self.video_id: str | None = self._extract_video_id()
        self.title: str | None = None
        self.description: str | None = None

    def _extract_video_id(self) -> str | None:
        if self.link.host in self.WATCH_HOSTS:
            if self.link.path == "/watch" and self.link.query:
                return parse_qs(self.link.query).get("v", [None])[0]
            if self.link.path and self.link.path.startswith("/shorts/"):
                return self.link.path.removeprefix("/shorts/").split("/")[0] or None
            return None

        if self.link.host == self.SHORT_HOST:
            video_id = (self.link.path or "").lstrip("/").split("/")[0]
            return video_id or None

        return None

    def _load_content(self):
        if self.video_id is None:
            raise ValueError(f"Could not extract a YouTube video id from {self.link}")

        self._load_metadata()
        transcript_text = self._load_transcript()

        sections = []
        if self.title:
            sections.append(f"Title: {self.title}")
        if self.description:
            sections.append(f"Description:\n{self.description}")
        sections.append(f"Transcript:\n{transcript_text}")

        self.content = "\n\n".join(sections)

    def _load_metadata(self):
        response = self._request_url()
        player_response = self._extract_json_after(
            response.text, "ytInitialPlayerResponse"
        )
        video_details = (player_response or {}).get("videoDetails", {})

        self.title = video_details.get("title")
        self.description = video_details.get("shortDescription")

    @staticmethod
    def _extract_json_after(html: str, marker: str) -> dict | None:
        # A regex like r"marker\s*=\s*(\{.*?\});" is unsound here: the description/keywords
        # text inside the JSON can itself contain the literal substring "};", which would
        # truncate a non-greedy match. Scan for the matching closing brace instead, skipping
        # over braces that appear inside quoted strings.
        start = html.find(marker)
        if start == -1:
            return None
        start = html.find("{", start)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(html[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(html[start : i + 1])
                    except json.JSONDecodeError as error:
                        raise JsonExtractionError(marker, error) from error
        return None

    def _load_transcript(self) -> str:
        api = YouTubeTranscriptApi()
        try:
            fetched = api.fetch(self.video_id, languages=self.TRANSCRIPT_LANGUAGES)
        except NoTranscriptFound:
            # Preferred languages not available - fall back to whatever transcript exists.
            transcript_list = api.list(self.video_id)
            try:
                transcript = next(iter(transcript_list))
            except StopIteration as error:
                raise YoutubeTranscriptUnavailable(
                    self.video_id, "no transcripts listed"
                ) from error
            fetched = transcript.fetch()
        except (TranscriptsDisabled, VideoUnavailable) as error:
            raise YoutubeTranscriptUnavailable(self.video_id, str(error)) from error

        return " ".join(snippet.text for snippet in fetched)


# ? How to deal with token_limit


YOUTUBE_HOSTS = YoutubeContentLoader.WATCH_HOSTS | {YoutubeContentLoader.SHORT_HOST}


def get_content_loader(link: HttpUrl) -> ContentLoader:
    if link.host in YOUTUBE_HOSTS:
        return YoutubeContentLoader(link)

    if (
        link.host == "arxiv.org"
        and ArxivContentLoader.extract_paper_id(link.path) is not None
    ):
        return ArxivContentLoader(link)

    # ! Improve pdf detection
    if link.path.endswith(".pdf"):
        return PdfUrlLoader(link)
    return UrlLoader(link)
