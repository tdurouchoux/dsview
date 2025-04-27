from typing import List

from pydantic import BaseModel

from dsview.config import ModelType, load_model_config
from dsview.content.content_loader import UrlLoader

from ..model_utils import LLMModel

links_extraction_model_config = load_model_config(ModelType.LINKS_EXTRACTION)


class RelevantLink(BaseModel):
    name: str
    url: str
    description: str


class LinkList(BaseModel):
    links: List[RelevantLink]


class LinksExtractor(LLMModel):
    DEFAULT_MODEL_CONFIG = links_extraction_model_config
    DEFAULT_SYSTEM_PROMPT_FILE = "system_links_extraction.txt"
    DEFAULT_USER_PROMPT_FILE = "user_links_extraction.txt"
    DEFAULT_STRUCTURED_OUTPUT_CLASS = LinkList

    def _preprocess_content(self, content_loader: UrlLoader) -> dict:
        if not isinstance(content_loader, UrlLoader):
            raise ValueError(
                "LinksExtractor can only extract links from an URL source."
            )

        input = {
            "url": content_loader.link,
            "content": content_loader.content,
            "content_links": "\n".join(content_loader.content_links),
        }
        return input

    def predict(self, content_loader: UrlLoader) -> BaseModel:
        return super().predict(self._preprocess_content(content_loader))

    async def async_predict(self, content_loader: UrlLoader) -> BaseModel:
        return await super().async_predict(self._preprocess_content(content_loader))
