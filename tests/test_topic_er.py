import asyncio

import pytest
from sqlalchemy import select
from sqlmodel import Session

from dsview.db.ingest import ingest_extraction
from dsview.db.schemas.extraction_schema import ERComparison, ExtractionTopic
from dsview.extraction import topic_er
from dsview.extraction.models.er_classification import ERResult
from dsview.extraction.models.topics_extraction import DataScienceTopic, TopicType
from dsview.extraction.topic_er import TopicERDecision, TopicMerge, TopicResolver


@pytest.fixture
def embedder(monkeypatch):
    """Replace the lazy INDEX model provider used by embed_and_{save,update}_topic."""

    class StubProvider:
        def __init__(self):
            self.embedded = []

        async def async_embed(self, text: str):
            # None, not a vector: SQLite cannot bind the Postgres ARRAY column
            self.embedded.append(text)

    stub = StubProvider()
    monkeypatch.setattr(ingest_extraction, "model_provider", stub)

    return stub


def make_topic(name: str, topic_type: TopicType = TopicType.CONCEPT):
    return DataScienceTopic(
        name=name, type=topic_type, description=f"description of {name}"
    )


def make_resolver(monkeypatch, results: list[ERResult]) -> TopicResolver:
    """A TopicResolver whose classifier replays `results`, one per comparison."""

    class StubClassifier:
        def __init__(self):
            self.calls = []

        async def async_predict(self, topic1, topic2):
            self.calls.append((topic1.name, topic2.name))
            return results[len(self.calls) - 1]

    monkeypatch.setattr(topic_er, "ERClassifier", StubClassifier)

    return TopicResolver()


class StubTopicsIndex:
    """Stands in for the DuckDB-backed TopicsIndex (candidates keyed by topic id)."""

    def __init__(self, candidates: dict[str, dict[int, DataScienceTopic]]):
        self.candidates = candidates
        self.deleted = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def query_topic(self, topic):
        return {
            candidate_id: {
                "topic": candidate,
                "fts_score": 1.0,
                "vss_distance": 0.1,
            }
            for candidate_id, candidate in self.candidates.get(topic.name, {}).items()
        }

    def delete_rows(self, where_clause):
        self.deleted.append(where_clause)


def merge_result(topic: DataScienceTopic) -> ERResult:
    return ERResult(analysis="same thing", merge_topic=True, topic=topic)


def no_merge_result() -> ERResult:
    return ERResult(analysis="different", merge_topic=False, topic=None)


def seed_topic(session: Session, name: str, topic_type: TopicType) -> int:
    """Insert a topic and detach it, so it is not in the session identity map."""
    topic = ExtractionTopic(
        name=name, type=topic_type.value, description=f"description of {name}"
    )
    session.add(topic)
    session.commit()
    topic_id = topic.id
    session.expunge_all()

    return topic_id


# --- _resolve_topic: async decision phase, no session ------------------------


def test_resolve_topic_without_candidates_is_an_insert(monkeypatch):
    resolver = make_resolver(monkeypatch, [])
    topic = make_topic("Diffusion Models")

    decision = asyncio.run(resolver._resolve_topic(topic, StubTopicsIndex({})))

    assert decision == TopicERDecision(topic, [], None)


def test_resolve_topic_stops_at_the_first_merging_candidate(monkeypatch):
    topic = make_topic("BigQuery")
    merged = make_topic("Google BigQuery", TopicType.TOOL)
    index = StubTopicsIndex(
        {
            "BigQuery": {
                11: make_topic("Big Query Studio", TopicType.TOOL),
                12: make_topic("Google BigQuery", TopicType.PLATFORM),
                13: make_topic("Never Examined"),
            }
        }
    )
    resolver = make_resolver(
        monkeypatch, [no_merge_result(), merge_result(merged), no_merge_result()]
    )

    decision = asyncio.run(resolver._resolve_topic(topic, index))

    # the pre-merge identity of candidate 12, not the merged one
    assert decision.merge == TopicMerge(12, "Google BigQuery", "Platform")
    assert decision.topic == merged
    # third candidate is never examined
    assert resolver.er_classifier.calls == [
        ("BigQuery", "Big Query Studio"),
        ("BigQuery", "Google BigQuery"),
    ]
    assert [comparison.name_2 for comparison in decision.comparisons] == [
        "Big Query Studio",
        "Google BigQuery",
    ]
    assert [comparison.merge_topic for comparison in decision.comparisons] == [
        False,
        True,
    ]
    # the merged candidate must not be offered to a later topic of the same batch
    assert index.deleted == ["id=12"]


# --- _apply_decision: sequential session phase -------------------------------


def test_apply_decision_inserts_new_topic_and_comparisons(
    session, embedder, monkeypatch
):
    resolver = make_resolver(monkeypatch, [])
    topic = make_topic("Diffusion Models")
    comparison = ERComparison(
        name_1="Diffusion Models",
        type_1="Concept",
        description_1="d1",
        name_2="Diffusion",
        type_2="Concept",
        description_2="d2",
        fts_score=1.0,
        vss_distance=0.1,
        decision_date="2026-08-10",
        merge_topic=False,
    )

    extraction_topic = asyncio.run(
        resolver._apply_decision(TopicERDecision(topic, [comparison], None), session)
    )
    session.commit()

    assert extraction_topic.id is not None
    assert extraction_topic.name == "Diffusion Models"
    assert extraction_topic.type == "Concept"
    assert embedder.embedded == ["description of Diffusion Models"]
    assert session.exec(select(ERComparison)).scalars().all() == [comparison]


def test_apply_decision_merges_in_place(session, embedder, monkeypatch):
    topic_id = seed_topic(session, "BigQuery", TopicType.PLATFORM)
    resolver = make_resolver(monkeypatch, [])
    merged = make_topic("Google BigQuery", TopicType.TOOL)

    extraction_topic = asyncio.run(
        resolver._apply_decision(
            TopicERDecision(merged, [], TopicMerge(topic_id, "BigQuery", "Platform")),
            session,
        )
    )

    assert extraction_topic.id == topic_id
    assert extraction_topic.name == "Google BigQuery"
    assert extraction_topic.type == "Tool"
    assert session.exec(select(ExtractionTopic)).scalars().all() == [extraction_topic]


# --- resolve_topics: end to end over a stub index ----------------------------


def test_resolve_topics_skips_topics_already_in_db(session, embedder, monkeypatch):
    seed_topic(session, "BigQuery", TopicType.PLATFORM)
    resolver = make_resolver(monkeypatch, [])
    monkeypatch.setattr(topic_er, "TopicsIndex", lambda: StubTopicsIndex({}))

    resolved, merges = asyncio.run(
        resolver.resolve_topics([make_topic("bigquery")], session)
    )

    # matched case-insensitively, so no ER call and no new row
    assert [topic.name for topic in resolved] == ["BigQuery"]
    assert merges == []
    assert resolver.er_classifier.calls == []
    assert embedder.embedded == []


def test_resolve_topics_drops_llm_duplicates(session, embedder, monkeypatch):
    resolver = make_resolver(monkeypatch, [])
    monkeypatch.setattr(topic_er, "TopicsIndex", lambda: StubTopicsIndex({}))

    resolved, _ = asyncio.run(
        resolver.resolve_topics(
            [make_topic("Diffusion Models"), make_topic("Diffusion Models")], session
        )
    )

    assert len(resolved) == 1


def test_resolve_topics_reports_every_merge(session, embedder, monkeypatch):
    """Regression guard for #55.

    The vault needs the pre-merge name/type of every merged topic to find the
    notes it has to rewrite. This used to be read back from SQLAlchemy attribute
    history, which any flush resets, so only one merge of a batch survived.
    """
    bigquery_id = seed_topic(session, "BigQuery", TopicType.PLATFORM)
    stakeholder_id = seed_topic(session, "Stakeholder Management", TopicType.CONCEPT)

    index = StubTopicsIndex(
        {
            "Google BigQuery": {
                bigquery_id: make_topic("BigQuery", TopicType.PLATFORM)
            },
            "Stakeholder Management in Data Projects": {
                stakeholder_id: make_topic("Stakeholder Management")
            },
        }
    )
    monkeypatch.setattr(topic_er, "TopicsIndex", lambda: index)

    new_topics = [
        make_topic("Google BigQuery", TopicType.TOOL),
        make_topic("Stakeholder Management in Data Projects"),
    ]
    resolver = make_resolver(
        monkeypatch, [merge_result(new_topics[0]), merge_result(new_topics[1])]
    )

    resolved, merges = asyncio.run(resolver.resolve_topics(new_topics, session))

    assert sorted(topic.id for topic in resolved) == sorted(
        [bigquery_id, stakeholder_id]
    )

    # every merge is reported, not just the last one to survive a flush
    assert sorted(merges, key=lambda merge: merge.topic_id) == sorted(
        [
            TopicMerge(bigquery_id, "BigQuery", "Platform"),
            TopicMerge(stakeholder_id, "Stakeholder Management", "Concept"),
        ],
        key=lambda merge: merge.topic_id,
    )
