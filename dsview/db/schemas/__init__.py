from .content_schema import FailedIngestion, InputContent
from .extraction_schema import (
    ContentTopicRelation,
    ERComparison,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
)
from .labels_schema import (
    LABELS_SCHEMA,
    ContentTypeLabels,
    ERLabels,
    LabelledContent,
    LinksLabels,
    TagLabels,
    TitleLabels,
    TopicsLabels,
)
from .schema_utils import drop_tables
