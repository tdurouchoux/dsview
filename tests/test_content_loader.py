import base64
from pathlib import Path

import pytest
from pydantic import HttpUrl

from dsview.extraction.content_loader import (
    ArxivContentLoader,
    GithubContentLoader,
    JsonExtractionError,
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


def test_select_representative_content_samples_past_midpoint_when_slightly_over_budget():
    # 28 chunks, total tokens 1.4x the budget - round(1.4) == 1, which used to
    # produce a no-op stride and only ever sample the document's start. With
    # a ceil()-based stride, sampling should still reach well past the
    # midpoint even for documents only moderately over budget.
    chunks = [([f"Chapter {i}"], f"word{i} " * 10) for i in range(28)]

    content = PdfUrlLoader._select_representative_content(
        chunks, token_limit=200, count_tokens=_count_words
    )

    selected = [i for i in range(28) if f"word{i}" in content]

    assert max(selected) >= 24


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
        # Not an /abs/ or /pdf/ shape - falls back to UrlLoader instead of hard-failing.
        (HttpUrl("https://arxiv.org/html/2402.02716"), UrlLoader),
        (HttpUrl("https://arxiv.org/list/cs.LG/2301"), UrlLoader),
        (
            HttpUrl("https://www.youtube.com/watch?v=jNQXAC9IVRw"),
            YoutubeContentLoader,
        ),
        (HttpUrl("https://youtu.be/jNQXAC9IVRw"), YoutubeContentLoader),
        (HttpUrl("https://github.com/pytorch/pytorch"), GithubContentLoader),
        (HttpUrl("https://github.com/pytorch/pytorch/"), GithubContentLoader),
        (HttpUrl("https://github.com/pytorch/pytorch/tree/main"), GithubContentLoader),
        (
            HttpUrl("https://github.com/pytorch/pytorch/tree/main/torch/nn"),
            GithubContentLoader,
        ),
        # Not a repo-root or tree shape - falls back to UrlLoader instead of hard-failing.
        (HttpUrl("https://github.com/pytorch/pytorch/blob/main/README.md"), UrlLoader),
        (HttpUrl("https://github.com/pytorch/pytorch/issues/123"), UrlLoader),
        (HttpUrl("https://github.com/pytorch/pytorch/pull/456"), UrlLoader),
        (HttpUrl("https://github.com/pytorch/pytorch/wiki"), UrlLoader),
        (HttpUrl("https://gist.github.com/octocat/6cad326836d38bd3a7ae"), UrlLoader),
    ],
)
def test_get_content_loader(link, expected_loader):
    content_loader = get_content_loader(link)
    assert isinstance(content_loader, expected_loader)


@pytest.mark.parametrize(
    "path,expected_paper_id",
    [
        ("/abs/2402.02716", "2402.02716"),
        ("/pdf/2402.02716", "2402.02716"),
        ("/pdf/2402.02716.pdf", "2402.02716"),
        ("/pdf/2402.02716v2", "2402.02716"),
        ("/abs/2402.02716/", "2402.02716"),
        ("/abs/hep-th/9901001", "hep-th/9901001"),
        ("/html/2402.02716", None),
        ("/list/cs.LG/2301", None),
        (None, None),
    ],
)
def test_extract_arxiv_paper_id(path, expected_paper_id):
    assert ArxivContentLoader.extract_paper_id(path) == expected_paper_id


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/pytorch/pytorch", ("pytorch", "pytorch", None)),
        ("/pytorch/pytorch/", ("pytorch", "pytorch", None)),
        ("/pytorch/pytorch/tree/main", ("pytorch", "pytorch", None)),
        ("/pytorch/pytorch/tree/main/", ("pytorch", "pytorch", None)),
        (
            "/pytorch/pytorch/tree/main/torch/nn",
            ("pytorch", "pytorch", "torch/nn"),
        ),
        (
            "/pytorch/pytorch/tree/main/torch/nn/",
            ("pytorch", "pytorch", "torch/nn"),
        ),
        ("/pytorch/pytorch/blob/main/README.md", None),
        ("/pytorch/pytorch/issues/123", None),
        ("/pytorch/pytorch/pull/456", None),
        ("/pytorch/pytorch/wiki", None),
        ("/pytorch", None),
        (None, None),
    ],
)
def test_parse_github_repo_path(path, expected):
    assert GithubContentLoader.parse_repo_path(path) == expected


@pytest.mark.parametrize(
    "link,expected_video_id",
    [
        (HttpUrl("https://www.youtube.com/watch?v=jNQXAC9IVRw"), "jNQXAC9IVRw"),
        (HttpUrl("https://www.youtube.com/shorts/abcDEF12345"), "abcDEF12345"),
        (HttpUrl("https://youtu.be/jNQXAC9IVRw"), "jNQXAC9IVRw"),
        (HttpUrl("https://www.youtube.com/watch"), None),
        (HttpUrl("https://www.youtube.com/"), None),
    ],
)
def test_extract_video_id(link, expected_video_id):
    content_loader = YoutubeContentLoader(link)
    assert content_loader.video_id == expected_video_id


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


def test_load_alphaxiv_overview_rejects_short_response(monkeypatch):
    fake_response = type("FakeResponse", (), {"text": "Not found"})()
    monkeypatch.setattr(
        ArxivContentLoader, "_request_url", lambda self, url=None: fake_response
    )

    content_loader = ArxivContentLoader(HttpUrl("https://arxiv.org/abs/2402.02716"))

    assert content_loader._load_alphaxiv_overview() is False
    assert content_loader.content is None


def test_load_alphaxiv_overview_accepts_long_response(monkeypatch):
    fake_response = type(
        "FakeResponse", (), {"text": "# Paper title\n\n" + "word " * 100}
    )()
    monkeypatch.setattr(
        ArxivContentLoader, "_request_url", lambda self, url=None: fake_response
    )

    content_loader = ArxivContentLoader(HttpUrl("https://arxiv.org/abs/2402.02716"))

    assert content_loader._load_alphaxiv_overview() is True
    assert content_loader.content.startswith("# Paper title")


def test_github_content_loader_readme():
    content_loader = get_content_loader(
        HttpUrl("https://github.com/huggingface/transformers")
    )
    content_loader.load()

    assert len(content_loader.content) > 0
    # README fetched via the API, not scraped from the rendered page.
    assert "<html" not in content_loader.content.lower()


def test_github_content_loader_directory_readme():
    content_loader = get_content_loader(
        HttpUrl("https://github.com/huggingface/transformers/tree/main/examples")
    )
    content_loader.load()

    assert len(content_loader.content) > 0
    assert "<html" not in content_loader.content.lower()


def test_github_content_loader_fallback(monkeypatch):
    monkeypatch.setattr(GithubContentLoader, "_load_readme", lambda self: False)

    content_loader = get_content_loader(
        HttpUrl("https://github.com/huggingface/transformers")
    )
    content_loader.load()

    assert len(content_loader.content) > 0
    assert len(content_loader.content_links) > 0


def test_load_readme_rejects_blank_content(monkeypatch):
    fake_content = base64.b64encode(b"   \n  ").decode()
    fake_response = type(
        "FakeResponse", (), {"json": lambda self: {"content": fake_content}}
    )()
    monkeypatch.setattr(
        GithubContentLoader, "_request_url", lambda self, url=None: fake_response
    )

    content_loader = GithubContentLoader(HttpUrl("https://github.com/pytorch/pytorch"))

    assert content_loader._load_readme() is False
    assert content_loader.content is None


def test_load_readme_accepts_real_content(monkeypatch):
    readme_text = (
        "# Title\n\n"
        "Some real readme content. See [the docs](https://example.com/docs) "
        "and [an issue](https://github.com/pytorch/pytorch/issues/1)."
    )
    fake_content = base64.b64encode(readme_text.encode()).decode()
    fake_response = type(
        "FakeResponse", (), {"json": lambda self: {"content": fake_content}}
    )()
    monkeypatch.setattr(
        GithubContentLoader, "_request_url", lambda self, url=None: fake_response
    )

    content_loader = GithubContentLoader(HttpUrl("https://github.com/pytorch/pytorch"))

    assert content_loader._load_readme() is True
    assert content_loader.content.startswith("# Title")
    assert sorted(content_loader.content_links) == [
        "https://example.com/docs",
        "https://github.com/pytorch/pytorch/issues/1",
    ]


@pytest.mark.parametrize(
    "content,expected_links",
    [
        # Plain external link is kept.
        (
            "See [docs](https://example.com/docs) for more.",
            ["https://example.com/docs"],
        ),
        # Relative link and anchor link are dropped (no scheme/host); further
        # filtering (e.g. repo-self-referencing links) happens downstream, not here.
        ("See [docs](./docs/guide.md) and [top](#top).", []),
        # Absolute links back to the same repo are kept here - not this
        # method's job to drop them.
        (
            (
                "See [issues](https://github.com/owner/repo/issues) and "
                "[wiki](https://github.com/owner/repo/wiki)."
            ),
            [
                "https://github.com/owner/repo/issues",
                "https://github.com/owner/repo/wiki",
            ],
        ),
        # Badge-wrapped link: the outer target, not the inner badge image, is kept.
        (
            "[![Build](https://img.shields.io/badge.svg)](https://example.com/actions)",
            ["https://example.com/actions"],
        ),
        # Reference-style link, resolved via its separate [ref]: url definition.
        (
            "See [the docs][ref] for more.\n\n[ref]: https://example.com/docs",
            ["https://example.com/docs"],
        ),
    ],
)
def test_extract_readme_links_drops_relative_links(content, expected_links):
    content_loader = GithubContentLoader(HttpUrl("https://github.com/owner/repo"))

    assert sorted(content_loader._extract_readme_links(content)) == sorted(
        expected_links
    )


def test_extract_json_after_returns_none_when_marker_missing():
    assert (
        YoutubeContentLoader._extract_json_after(
            "<html></html>", "ytInitialPlayerResponse"
        )
        is None
    )


def test_extract_json_after_returns_none_when_no_opening_brace():
    html = "var ytInitialPlayerResponse = ; more html"

    assert (
        YoutubeContentLoader._extract_json_after(html, "ytInitialPlayerResponse")
        is None
    )


def test_extract_json_after_handles_embedded_brace_semicolon_in_string():
    html = (
        'var ytInitialPlayerResponse = {"videoDetails": '
        '{"shortDescription": "some odd text }; still going"}};'
    )

    result = YoutubeContentLoader._extract_json_after(html, "ytInitialPlayerResponse")

    assert result["videoDetails"]["shortDescription"] == "some odd text }; still going"


def test_extract_json_after_raises_typed_error_on_malformed_json():
    html = 'var ytInitialPlayerResponse = {"a": };'

    with pytest.raises(JsonExtractionError):
        YoutubeContentLoader._extract_json_after(html, "ytInitialPlayerResponse")
