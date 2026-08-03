import os

import yaml

from finminutes.core.exceptions import ConfigError
from finminutes.core.models import Background


class BackgroundLoader:
    def load(self, path: str) -> Background:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Background file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"Invalid background format in {path}")
        return self._parse(data)

    @staticmethod
    def _parse(data: dict) -> Background:
        consensus = data.get("known_consensus", [])
        if consensus is None:
            consensus = []
        if not isinstance(consensus, list):
            raise ConfigError("'known_consensus' must be a list")
        return Background(
            company=data.get("company", ""),
            industry=data.get("industry", ""),
            participants=data.get("participants", ""),
            known_consensus=consensus,
            meeting_purpose=data.get("meeting_purpose", ""),
        )
