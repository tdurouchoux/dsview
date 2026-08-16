import json

import logfire
from mistralai.client.models import EmbeddingResponse, EmbeddingResponseData, UsageInfo
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dsview.model_utils import observability
from dsview.model_utils.observability import (
    MISTRAL_SCOPE,
    MistralUsageSpanProcessor,
    embedding_span,
)

MISTRAL_CHAT_ATTRIBUTES = {
    "llm.model_name": "mistral-small-2603",
    "llm.token_count.prompt": 1000,
    "llm.token_count.completion": 500,
}


def emit_span(scope: str, attributes: dict):
    """Run one span through a real logfire pipeline and return what got exported.

    Going through logfire rather than a bare TracerProvider matters: logfire wraps
    everything in a MainSpanProcessorWrapper that rebuilds each span, leaving the
    span's attributes an immutable mappingproxy by the time our processor sees it.
    `local=True` keeps this off the global tracer provider.
    """
    exporter = InMemorySpanExporter()

    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        additional_span_processors=[
            MistralUsageSpanProcessor(),
            SimpleSpanProcessor(exporter),
        ],
    )

    tracer = instance.config.get_tracer_provider().get_tracer(scope)
    with tracer.start_as_current_span("chat") as span:
        span.set_attributes(attributes)

    return exporter.get_finished_spans()[0].attributes


def test_mistral_usage_is_translated_to_semconv():
    attributes = emit_span(MISTRAL_SCOPE, MISTRAL_CHAT_ATTRIBUTES)

    assert attributes["gen_ai.request.model"] == "mistral-small-2603"
    assert attributes["gen_ai.usage.input_tokens"] == 1000
    assert attributes["gen_ai.usage.output_tokens"] == 500
    assert attributes["gen_ai.system"] == "mistral"
    assert attributes["gen_ai.provider.name"] == "mistral"

    # The originals stay put, so the OpenInference view of the span is unchanged.
    assert attributes["llm.token_count.prompt"] == 1000


def test_other_scopes_are_left_alone():
    attributes = emit_span(
        "openinference.instrumentation.langchain", MISTRAL_CHAT_ATTRIBUTES
    )

    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.system" not in attributes


def test_spans_without_token_counts_are_left_alone():
    # Not a model request: tagging it would double count cost in the UI.
    attributes = emit_span(MISTRAL_SCOPE, {"llm.model_name": "mistral-small-2603"})

    assert "gen_ai.request.model" not in attributes
    assert "gen_ai.system" not in attributes


def embedding_response(prompt_tokens: int = 12) -> EmbeddingResponse:
    return EmbeddingResponse(
        id="emb-1",
        object="list",
        model="mistral-embed",
        usage=UsageInfo(
            prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens
        ),
        data=[EmbeddingResponseData(object="embedding", embedding=[0.1, 0.2], index=0)],
    )


def emit_embedding_span(response, monkeypatch) -> dict:
    exporter = InMemorySpanExporter()

    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
    )

    # `embedding_span` emits through the global tracer provider, which logfire only
    # installs when it isn't local; point it at this instance rather than letting a
    # test reconfigure the whole process.
    monkeypatch.setattr(
        observability,
        "EMBEDDINGS_TRACER",
        instance.config.get_tracer_provider().get_tracer("dsview.embeddings"),
    )

    with embedding_span("mistral-embed", "some text to embed") as record:
        record(response)

    return exporter.get_finished_spans()[0].attributes


def test_embedding_span_carries_usage_and_llm_tag(monkeypatch):
    attributes = emit_embedding_span(embedding_response(), monkeypatch)

    assert attributes["gen_ai.operation.name"] == "embeddings"
    assert attributes["gen_ai.system"] == "mistral"
    assert attributes["gen_ai.provider.name"] == "mistral"
    assert attributes["gen_ai.request.model"] == "mistral-embed"
    assert attributes["gen_ai.response.model"] == "mistral-embed"
    assert attributes["gen_ai.usage.input_tokens"] == 12
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert attributes["gen_ai.embeddings.dimension.count"] == 2
    assert attributes["logfire.tags"] == ("LLM",)


def test_embedding_span_records_the_input_but_not_the_vector(monkeypatch):
    attributes = emit_embedding_span(embedding_response(), monkeypatch)

    assert json.loads(attributes["request_data"])["input"] == ["some text to embed"]
    assert json.loads(attributes["response_data"]) == {
        "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12}
    }


def test_unreadable_response_does_not_break_the_embedding_call(monkeypatch):
    # Observability must never take down a live call - a response we can't read just
    # loses its usage attributes.
    attributes = emit_embedding_span(object(), monkeypatch)

    assert "gen_ai.usage.input_tokens" not in attributes
    assert attributes["gen_ai.request.model"] == "mistral-embed"
