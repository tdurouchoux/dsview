import asyncio

import pytest

from dsview.extraction.models.er_classification import ERClassifier, ERResult
from dsview.extraction.models.topics_extraction import DataScienceTopic, TopicType


@pytest.fixture
def sample_topics():
    """Create sample topics for testing ER classification."""
    topic1 = DataScienceTopic(
        name="Machine Learning",
        type=TopicType.CONCEPT,
        description="Machine Learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    )
    topic2 = DataScienceTopic(
        name="ML",
        type=TopicType.CONCEPT,
        description="ML stands for Machine Learning, a field of computer science that uses statistical techniques to give computer systems the ability to learn.",
    )
    topic3 = DataScienceTopic(
        name="Deep Learning",
        type=TopicType.CONCEPT,
        description="Deep Learning is a subset of machine learning that uses neural networks with many layers.",
    )

    return topic1, topic2, topic3


def test_er_classifier_initialization():
    """Test that ERClassifier can be initialized properly."""
    classifier = ERClassifier()

    assert classifier is not None
    assert hasattr(classifier, "DEFAULT_MODEL_CONFIG")
    assert hasattr(classifier, "DEFAULT_SYSTEM_PROMPT_FILE")
    assert hasattr(classifier, "DEFAULT_USER_PROMPT_FILE")
    assert hasattr(classifier, "DEFAULT_STRUCTURED_OUTPUT_CLASS")
    assert classifier.DEFAULT_STRUCTURED_OUTPUT_CLASS == ERResult


def test_format_topics(sample_topics):
    """Test the _format_topics method."""
    classifier = ERClassifier()
    topic1, topic2, _ = sample_topics

    formatted = classifier._format_topics(topic1, topic2)

    assert isinstance(formatted, dict)
    assert "name_1" in formatted
    assert "type_1" in formatted
    assert "description_1" in formatted
    assert "name_2" in formatted
    assert "type_2" in formatted
    assert "description_2" in formatted

    assert formatted["name_1"] == "Machine Learning"
    assert formatted["type_1"] == "Concept"
    assert formatted["name_2"] == "ML"
    assert formatted["type_2"] == "Concept"


def test_er_classifier_predict(sample_topics):
    """Test the predict method with similar topics (should suggest merge)."""
    classifier = ERClassifier()
    topic1, topic2, _ = sample_topics

    # This should return an ERResult indicating whether to merge
    result = classifier.predict(topic1, topic2)

    assert isinstance(result, ERResult)
    assert hasattr(result, "merge_topic")
    assert hasattr(result, "topic")
    assert isinstance(result.merge_topic, bool)

    # For similar topics like "Machine Learning" and "ML", we expect merge_topic to be True
    # Note: The actual value depends on the LLM's response, so we just check the structure


def test_er_classifier_predict_different_topics(sample_topics):
    """Test the predict method with different topics (should not suggest merge)."""
    classifier = ERClassifier()
    topic1, _, topic3 = sample_topics  # Machine Learning vs Deep Learning

    result = classifier.predict(topic1, topic3)

    assert isinstance(result, ERResult)
    assert isinstance(result.merge_topic, bool)

    # For different topics like "Machine Learning" and "Deep Learning",
    # we expect merge_topic to be False
    # Note: The actual value depends on the LLM's response


def test_er_classifier_async_predict(sample_topics):
    """Test the async_predict method."""
    classifier = ERClassifier()
    topic1, topic2, _ = sample_topics

    # Run the async function in an event loop
    result = asyncio.run(classifier.async_predict(topic1, topic2))

    assert isinstance(result, ERResult)
    assert hasattr(result, "merge_topic")
    assert hasattr(result, "topic")
    assert isinstance(result.merge_topic, bool)


def test_er_result_model():
    """Test the ERResult Pydantic model."""
    # Test with merge_topic=True
    result1 = ERResult(
        merge_topic=True,
        topic=DataScienceTopic(
            name="Merged Topic",
            type=TopicType.CONCEPT,
            description="This is a merged topic",
        ),
    )

    assert result1.merge_topic is True
    assert result1.topic.name == "Merged Topic"

    # Test with merge_topic=False
    result2 = ERResult(
        merge_topic=False,
        topic=DataScienceTopic(
            name="Original Topic",
            type=TopicType.TOOL,
            description="This is the original topic",
        ),
    )

    assert result2.merge_topic is False
    assert result2.topic.name == "Original Topic"


def test_er_classifier_with_real_world_examples():
    """Test with real-world topic examples that should definitely merge or not merge."""
    classifier = ERClassifier()

    # Test 1: Same topic with different names (should merge)
    topic_ml_full = DataScienceTopic(
        name="Machine Learning",
        type=TopicType.CONCEPT,
        description="Subset of AI that builds systems learning from data",
    )

    topic_ml_abbrev = DataScienceTopic(
        name="ML",
        type=TopicType.CONCEPT,
        description="Abbreviation for Machine Learning",
    )

    result = classifier.predict(topic_ml_full, topic_ml_abbrev)
    assert isinstance(result, ERResult)

    # Test 2: Completely different topics (should not merge)
    topic_pytorch = DataScienceTopic(
        name="PyTorch",
        type=TopicType.TOOL,
        description="Deep learning framework by Facebook",
    )

    topic_scikitlearn = DataScienceTopic(
        name="scikit-learn",
        type=TopicType.TOOL,
        description="Machine learning library for Python",
    )

    result2 = classifier.predict(topic_pytorch, topic_scikitlearn)
    assert isinstance(result2, ERResult)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
