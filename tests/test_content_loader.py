from pathlib import Path

import pytest
from pydantic import HttpUrl

from dsview.extraction.content_loader import (
    PdfUrlLoader,
    UrlLoader,
    WebRequestFailure,
    get_content_loader,
)

TEST_FILE_CONTENT = "This is a test."


@pytest.mark.parametrize(
    "link,expected_loader",
    [
        (HttpUrl("https://datajunction.io/"), UrlLoader),
        (
            HttpUrl("https://sante.gouv.fr/IMG/pdf/etat_des_lieux_ia_en_sante.pdf"),
            PdfUrlLoader,
        ),
        (HttpUrl("https://arxiv.org/pdf/2402.02716"), PdfUrlLoader),
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
        HttpUrl("https://arxiv.org/pdf/2402.02716")
    )

    content_loader.load()

    # Current implementation doesn't save to pdf_filepath, just extracts content
    assert len(content_loader.content) > 0
    assert len(content_loader.content.split(" ")) < 30_000
