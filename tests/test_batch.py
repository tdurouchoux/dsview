import json
from types import SimpleNamespace

import pandas as pd
from pydantic import BaseModel

from dsview.config import ModelConfig
from dsview.model_utils.model_provider import ModelProvider


class FakeResult(BaseModel):
    answer: str


def make_fake_topic(name: str, type_value: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, type=SimpleNamespace(value=type_value))


class EchoProvider(ModelProvider):
    def _retrieve_model(self, model_name: str):
        pass

    def _embed(self, input):
        raise NotImplementedError

    async def _async_embed(self, input):
        raise NotImplementedError

    def _complete(self, messages, structured_output_class=None):
        raise NotImplementedError

    async def _async_complete(self, messages, structured_output_class=None):
        return messages[-1]["content"]


def make_model_config() -> ModelConfig:
    return ModelConfig(chat_model="fake-model", token_limit=1000)


def test_batch_send_messages_fallback_preserves_order():
    provider = EchoProvider(make_model_config())

    batch_messages = [
        [{"role": "system", "content": "s"}, {"role": "user", "content": str(i)}]
        for i in range(25)
    ]

    results = provider.batch_send_messages(batch_messages)

    assert results == [str(i) for i in range(25)]


def test_mistral_batch_file_and_output_roundtrip(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")

    from dsview.model_utils.providers.mistral import MistralProvider

    provider = MistralProvider(make_model_config())

    batch_messages = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]

    batch_file = provider._build_batch_file(batch_messages, FakeResult)
    lines = [json.loads(line) for line in batch_file.decode().splitlines()]

    assert [line["custom_id"] for line in lines] == ["0", "1"]
    assert lines[0]["body"]["messages"] == batch_messages[0]
    assert lines[0]["body"]["response_format"]["type"] == "json_schema"

    output_lines = [
        json.dumps(
            {
                "custom_id": "1",
                "response": {
                    "status_code": 200,
                    "body": {"choices": [{"message": {"content": '{"answer": "ok"}'}}]},
                },
            }
        ),
        # Failed request: must stay None so the sync fallback picks it up
        json.dumps({"custom_id": "0", "response": {"status_code": 500}}),
    ]

    results = [None, None]
    provider._parse_batch_output("\n".join(output_lines).encode(), results, FakeResult)

    assert results[0] is None
    assert results[1] == FakeResult(answer="ok")


def test_score_row_mixed_matches():
    from dsview.evaluation.topics_extraction import score_row

    row = pd.Series(
        {
            "pred_topics": [
                make_fake_topic("pandas", "Library"),
                make_fake_topic("panda dataframe", "Library"),
                make_fake_topic("unrelated", "Concept"),
            ],
            "name": ["pandas", "numpy"],
            "type": ["Library", "Library"],
        }
    )
    # Second predicted topic matched to "pandas" by the ER judge, with a
    # wrong predicted type; third topic unmatched.
    er_matches = {("row", 1): "pandas"}

    scores = score_row("row", row, er_matches)
    count_correct, precision, recall, average_precision, count_type_correct = scores

    assert count_correct == 2
    assert precision == 2 / 3
    # Both matches point to the same labelled topic
    assert recall == 1 / 2
    assert average_precision == (1 + 1) / 3
    assert count_type_correct == 2


def test_score_row_no_predictions():
    from dsview.evaluation.topics_extraction import score_row

    row = pd.Series({"pred_topics": [], "name": ["pandas"], "type": ["Library"]})

    scores = score_row("row", row, {})
    count_correct, precision, recall, average_precision, count_type_correct = scores

    assert count_correct == 0
    assert precision == 1
    assert recall == 0
    assert average_precision == 1
    assert count_type_correct == 0
