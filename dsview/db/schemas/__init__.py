from sqlmodel import SQLModel

from dsview.config import get_sqlite_engine

from .extraction_schema import (
    ContentTopicRelation,
    ERComparison,
    ERDecision,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
)
from .input_content_schema import FailedIngestion, InputContent
from .labels_schema import (
    ContentTypeLabels,
    ERLabels,
    LabelledContent,
    LinksLabels,
    TagLabels,
    TitleLabels,
    TopicsLabels,
)

from .schema_utils import drop_tables

engine = get_sqlite_engine()
SQLModel.metadata.create_all(engine)
