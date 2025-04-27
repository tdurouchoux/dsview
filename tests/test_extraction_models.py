import pytest
from pydantic import HttpUrl

from dsview.content.content_loader import get_content_loader
from dsview.extraction.content_extraction import ContentExtractor
from dsview.models.llm_models.description_generation import (
    ContentDescription,
    DataScienceTag,
    DescriptionGenerator,
)
from dsview.models.llm_models.links_extraction import (
    LinkList,
    LinksExtractor,
    RelevantLink,
)
from dsview.models.llm_models.summary_generation import SummaryGenerator
from dsview.models.llm_models.topics_extraction import (
    DataScienceTopic,
    TopicList,
    TopicsExtractor,
)

TEST_URL = HttpUrl("https://fastapi.tiangolo.com/tutorial/testing")


@pytest.fixture
def content_loader():
    content_loader = get_content_loader(TEST_URL, 128_000)

    content_loader.load()

    return content_loader


@pytest.fixture
def content(content_loader):
    return content_loader.content


def test_summary_generation(content):
    summary_generator = SummaryGenerator()
    summary = summary_generator.predict({"content": content})
    assert len(summary) > 0


def test_content_description(content):
    description_generator = DescriptionGenerator()
    content_description = description_generator.predict({"content": content})
    assert isinstance(content_description, ContentDescription)


def test_topics_extraction(content):
    topics_extractor = TopicsExtractor()
    topic_list = topics_extractor.predict({"content": content})
    assert isinstance(topic_list, TopicList)


def test_links_extraction(content_loader):
    links_extractor = LinksExtractor()
    link_list = links_extractor.predict(content_loader)
    assert isinstance(link_list, LinkList)


def test_select_valid_properties():
    valid_properties, invalid_properties = ContentExtractor.select_valid_properties(
        [
            DataScienceTag(name="Cloud Computing"),
            DataScienceTag(name="Mathematics"),
            DataScienceTag(name="MlOps"),
        ],
        ["Cloud Computing", "Mathematics"],
        "tag",
    )

    assert valid_properties == [
        DataScienceTag(name="Cloud Computing"),
        DataScienceTag(name="Mathematics"),
    ]
    assert invalid_properties == [DataScienceTag(name="MlOps")]


def test_content_extractor(content_loader):
    content_extractor = ContentExtractor()
    summary, content_description, topics, content_links = (
        content_extractor.extract_content(content_loader, None, 1)
    )

    assert len(summary) > 0 and isinstance(summary, str)
    assert isinstance(content_description, ContentDescription)
    assert len(topics) > 0 and isinstance(topics[0], DataScienceTopic)
    assert len(content_links) > 0 and isinstance(content_links[0], RelevantLink)
