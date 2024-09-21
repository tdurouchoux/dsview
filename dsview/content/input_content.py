from datetime import date
import logging
from pathlib import Path
from typing import Dict, Union

from pydantic import BaseModel, Field, HttpUrl

MEDIUM_HOSTS = ["medium.com", "towardsdatascience.com"]

logger = logging.getLogger(__name__)


class InputContent(BaseModel):
    link: Union[HttpUrl, Path]
    upload_date: date
    already_read: bool = False
    read_priority: int = Field(default=1, ge=0, le=5)
    source: str = None

    def model_post_init(self, __context):
        try:
            if self.link.host in MEDIUM_HOSTS:
                logger.info("Received a medium link, redirecting to readmedium")
                self.link = HttpUrl("https://readmedium.com/" + str(self.link))

        except AttributeError:
            return

    def get_str_dict(self) -> Dict:
        instance_dict = dict(self)
        instance_dict["link"] = str(self.link)
        instance_dict["upload_date"] = self.upload_date.isoformat()

        if self.source is None:
            del instance_dict["source"]

        return instance_dict
