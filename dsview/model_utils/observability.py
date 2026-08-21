"""Make the OpenInference Mistral spans legible to Logfire's LLM features.

Logfire translates OpenInference attributes into the shape its LLM views expect —
OTel GenAI semantic conventions for tokens and cost, an 'LLM' tag, and
request/response payloads for the Generation panel — but only for the langchain,
langsmith and litellm scopes (see `logfire._internal.exporters.processor_wrapper`).
Mistral spans therefore arrive carrying `llm.*` names that nothing reads, and show
up as ordinary spans with neither tokens, price, nor a conversation view.

This does for the Mistral scope what logfire's own `_transform_litellm_span` does
for litellm, and mirrors its choices. Price is derived by the Logfire backend from
the semconv attributes — the translations logfire ships never set `operation.cost`
themselves — so there is deliberately no price computation here.

Embeddings need the opposite treatment: OpenInference wraps Mistral's chat methods
but not `Embeddings.create`, so there is no span to rewrite and `embedding_span`
emits one instead.
"""

import json
import logging
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import SpanKind

logger = logging.getLogger(__name__)

MISTRAL_SCOPE = "openinference.instrumentation.mistralai"

PROMPT_TOKENS = "llm.token_count.prompt"

# OpenInference attribute -> OTel GenAI semantic convention attribute.
SEMCONV_RENAMES = {
    "llm.model_name": "gen_ai.request.model",
    PROMPT_TOKENS: "gen_ai.usage.input_tokens",
    "llm.token_count.completion": "gen_ai.usage.output_tokens",
}

# The 'LLM' tag is what switches on logfire's LLM features; request_data and
# response_data are what its Generation panel renders, and the schema tells the
# backend to parse those two as JSON instead of displaying them as strings.
TAGS_KEY = "logfire.tags"
JSON_SCHEMA_KEY = "logfire.json_schema"
PAYLOAD_JSON_SCHEMA = (
    '{"type":"object","properties":'
    '{"request_data":{"type":"object"},"response_data":{"type":"object"}}}'
)
EMBEDDING_JSON_SCHEMA = (
    '{"type":"object","properties":'
    '{"request_data":{"type":"object"},"response_data":{"type":"object"},'
    '"gen_ai.usage.raw":{"type":"object"}}}'
)


def _payload_attributes(attributes) -> dict:
    """Recast the OpenInference request/response blobs for the Generation panel.

    `output.value` is the whole ChatCompletionResponse as JSON, so the assistant
    message and the responding model both come out of it.
    """
    try:
        response = json.loads(attributes["output.value"])
        payload = {
            "request_data": attributes["input.value"],
            "response_data": json.dumps({"message": response["choices"][0]["message"]}),
            JSON_SCHEMA_KEY: PAYLOAD_JSON_SCHEMA,
        }

        if "model" in response:
            payload["gen_ai.response.model"] = response["model"]

        return payload

    except Exception:
        logger.debug("Could not read Mistral span payload", exc_info=True)
        return {}


class MistralUsageSpanProcessor(SpanProcessor):
    """Rewrite OpenInference Mistral spans into the form logfire's LLM views read."""

    def on_end(self, span: ReadableSpan) -> None:
        scope = span.instrumentation_scope

        if scope is None or scope.name != MISTRAL_SCOPE:
            return

        attributes = span.attributes

        # No token count means this isn't a model request. Tagging it as one would
        # double count cost in the UI.
        if attributes is None or PROMPT_TOKENS not in attributes:
            return

        new_attributes = {
            target: attributes[source]
            for source, target in SEMCONV_RENAMES.items()
            if source in attributes
        }

        # `gen_ai.system` is the older spelling of `gen_ai.provider.name`; logfire
        # writes both, so match it.
        new_attributes["gen_ai.system"] = new_attributes["gen_ai.provider.name"] = (
            "mistral"
        )
        # What logfire's native integrations tag a completion with, and what its
        # LLM/providers dashboard filters on.
        new_attributes["gen_ai.operation.name"] = "chat"
        new_attributes[TAGS_KEY] = ["LLM"]
        new_attributes.update(_payload_attributes(attributes))

        # Replace the mapping rather than assigning into it. Logfire wraps the whole
        # pipeline in a MainSpanProcessorWrapper that rebuilds every span through
        # `span_to_dict`, which reads the immutable `ReadableSpan.attributes`
        # property, so the span we get here holds a mappingproxy.
        #
        # `_attributes` is a private opentelemetry-sdk attribute, not a published
        # API - assert the assumption instead of silently no-op'ing if it ever
        # stops holding. `on_end` runs inline in the real Mistral call path
        # (`SynchronousMultiSpanProcessor` has no try/except around processors), so
        # the assertion failure must not propagate into a live call - only observability
        # breaks, not ingestion.
        try:
            assert hasattr(span, "_attributes"), (
                "ReadableSpan has no _attributes to rewrite; "
                "opentelemetry-sdk's internals may have changed"
            )
            span._attributes = {**attributes, **new_attributes}
        except AssertionError:
            logger.error("Could not rewrite Mistral span attributes", exc_info=True)


# A plain OTel tracer rather than `logfire.span()`: the entrypoints that never call
# `logfire.configure()` (CLI, evals, notebooks) then get a silent no-op tracer instead
# of a LogfireNotConfiguredWarning. Resolves the global provider lazily, so importing
# this module before `logfire.configure()` is fine.
EMBEDDINGS_TRACER = trace.get_tracer("dsview.embeddings")


def _embedding_response_attributes(response) -> dict:
    """Model, usage and vector width — never the vectors themselves.

    Mirrors logfire's own embeddings handling, which sets `response_data` to just
    `{"usage": ...}`, plus the two things it can't derive from an OpenAI response:
    the dimension count, and an explicit 0 for output tokens. Logfire omits the
    latter only because OpenAI's embedding usage object has no such field.
    """
    try:
        usage = response.usage.model_dump(exclude_none=True)

        return {
            "gen_ai.response.model": response.model,
            "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
            "gen_ai.usage.output_tokens": 0,
            # `gen_ai.embeddings.dimension.count` is the semantic convention for
            # vector width; reading it off the response keeps the vector out of the
            # span while still reporting its size.
            "gen_ai.embeddings.dimension.count": len(response.data[0].embedding),
            # The provider-native usage blob, which is what genai-prices reads to
            # derive a price. Logfire sets it alongside the token counts.
            "gen_ai.usage.raw": json.dumps(usage),
            "response_data": json.dumps({"usage": usage}),
        }

    except Exception:
        logger.debug("Could not read Mistral embedding usage", exc_info=True)
        return {}


@contextmanager
def embedding_span(model: str, input: str):
    """Trace one Mistral embedding call, yielding a callable to record its response.

    Attribute vocabulary matches what logfire's OpenAI integration emits for
    `/embeddings`, so these spans get the same treatment as any other LLM span.
    """
    with EMBEDDINGS_TRACER.start_as_current_span(
        f"Embedding Creation with {model!r}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "embeddings",
            "gen_ai.system": "mistral",
            "gen_ai.provider.name": "mistral",
            "gen_ai.request.model": model,
            "request_data": json.dumps({"model": model, "input": [input]}),
            TAGS_KEY: ["LLM"],
            JSON_SCHEMA_KEY: EMBEDDING_JSON_SCHEMA,
        },
    ) as span:

        def record(response) -> None:
            span.set_attributes(_embedding_response_attributes(response))

        yield record
