from pathlib import Path

import pytest
from pydantic import HttpUrl

from dsview.extraction.content_loader import (
    ArxivContentLoader,
    PdfUrlLoader,
    UrlLoader,
    WebRequestFailure,
    YoutubeContentLoader,
    get_content_loader,
)

TEST_FILE_CONTENT = "This is a test."


def _count_words(text: str) -> int:
    return len(text.split())


def test_select_representative_content_within_budget_keeps_everything():
    chunks = [
        (["Chapter 1"], "one two three"),
        (["Chapter 2"], "four five six"),
    ]

    content = PdfUrlLoader._select_representative_content(
        chunks, token_limit=100, count_tokens=_count_words
    )

    assert "## Chapter 1" in content
    assert "## Chapter 2" in content
    for word in ["one", "two", "three", "four", "five", "six"]:
        assert word in content


def test_select_representative_content_samples_across_whole_document():
    # 30 chunks, only a third of which fit in budget - the selected content
    # should still span from the start to well past the midpoint, instead of
    # only ever keeping the first few chunks.
    chunks = [([f"Chapter {i}"], f"word{i} " * 20) for i in range(30)]

    content = PdfUrlLoader._select_representative_content(
        chunks, token_limit=200, count_tokens=_count_words
    )

    selected = [i for i in range(30) if f"word{i}" in content]

    assert min(selected) <= 3
    assert max(selected) >= 15
    assert len(selected) >= 5


def test_select_representative_content_front_matter_has_no_heading_prefix():
    chunks = [
        ([], "front matter text with no heading"),
        (["Chapter 1"], "chapter one text"),
    ]

    content = PdfUrlLoader._select_representative_content(
        chunks, token_limit=100, count_tokens=_count_words
    )

    assert content.startswith("front matter text")
    assert "## Chapter 1" in content


def test_select_representative_content_never_exceeds_budget():
    # Regression test: a document with many small headings (e.g. a report with
    # hundreds of subsections) must not blow the token budget - the heading
    # markup added to each kept chunk counts against the budget too.
    chunks = [([f"Section {i}"], f"word{i} " * 20) for i in range(200)]

    token_limit = 500
    content = PdfUrlLoader._select_representative_content(
        chunks, token_limit=token_limit, count_tokens=_count_words
    )

    assert _count_words(content) <= token_limit


@pytest.mark.parametrize(
    "link,expected_loader",
    [
        (HttpUrl("https://datajunction.io/"), UrlLoader),
        (
            HttpUrl("https://sante.gouv.fr/IMG/pdf/etat_des_lieux_ia_en_sante.pdf"),
            PdfUrlLoader,
        ),
        (HttpUrl("https://arxiv.org/pdf/2402.02716"), ArxivContentLoader),
        (HttpUrl("https://arxiv.org/abs/2402.02716"), ArxivContentLoader),
        (
            HttpUrl("https://www.youtube.com/watch?v=jNQXAC9IVRw"),
            YoutubeContentLoader,
        ),
        (HttpUrl("https://youtu.be/jNQXAC9IVRw"), YoutubeContentLoader),
    ],
)
def test_get_content_loader(link, expected_loader):
    content_loader = get_content_loader(link)
    assert isinstance(content_loader, expected_loader)


def test_text_loader(tmp_path):
    test_file = tmp_path / "test_file.txt"
    test_file.write_text(TEST_FILE_CONTENT)

    # TextLoader is no longer supported, skipping this test
    pytest.skip("TextLoader is no longer supported")


def test_web_request_failure():
    content_loader = get_content_loader(
        HttpUrl("https://fastapi.tiangolo.com/tutorial/random")
    )

    with pytest.raises(WebRequestFailure):
        content_loader.load()


def test_url_loader():
    content_loader = get_content_loader(
        HttpUrl("https://fastapi.tiangolo.com/tutorial/testing")
    )

    content_loader.load()

    assert content_loader.content is not None
    assert len(content_loader.content_links) > 0


def test_pdf_url_loader(tmp_path):
    content_loader = get_content_loader(
        HttpUrl(
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        )
    )

    content_loader.load()

    # Current implementation doesn't save to pdf_filepath, just extracts content
    assert len(content_loader.content) > 0
    assert len(content_loader.content.split(" ")) < 30_000


def test_youtube_content_loader():
    # "Me at the zoo" - the first YouTube video, has auto-captions, essentially
    # never disappears.
    content_loader = get_content_loader(
        HttpUrl("https://www.youtube.com/watch?v=jNQXAC9IVRw")
    )

    content_loader.load()

    assert content_loader.content is not None
    assert "Title:" in content_loader.content
    assert "Transcript:" in content_loader.content


def test_arxiv_content_loader_alphaxiv_overview():
    content_loader = get_content_loader(HttpUrl("https://arxiv.org/abs/2402.02716"))

    content_loader.load()

    assert len(content_loader.content) > 0
    assert "##" in content_loader.content  # alphaXiv overview is markdown


def test_arxiv_content_loader_pdf_fallback(monkeypatch):
    monkeypatch.setattr(
        ArxivContentLoader, "_load_alphaxiv_overview", lambda self: False
    )

    content_loader = get_content_loader(HttpUrl("https://arxiv.org/abs/2402.02716"))
    content_loader.load()

    assert len(content_loader.content) > 0
    assert len(content_loader.content.split(" ")) < 30_000
