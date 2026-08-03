import glob
import os

import yaml

from finminutes.core.exceptions import ConfigError
from finminutes.core.models import Section, Template


class TemplateLoader:
    def __init__(self, template_dir: str | None = None, subdir: str = ""):
        self._dir = os.path.join(template_dir or self._default_dir(), subdir)

    def load(self, name: str) -> Template:
        path = os.path.join(self._dir, f"{name}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"Invalid template format in {path}")
        return self._parse(data)

    def list_templates(self) -> list[str]:
        pattern = os.path.join(self._dir, "*.yaml")
        files = glob.glob(pattern)
        return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)

    def _parse(self, data: dict) -> Template:
        name = data.get("name", "")
        description = data.get("description", "")
        raw_sections = data.get("sections", [])
        if not isinstance(raw_sections, list):
            raise ConfigError("'sections' must be a list")
        sections = []
        for item in raw_sections:
            if not isinstance(item, dict) or "title" not in item:
                raise ConfigError(f"Invalid section entry: {item}")
            sections.append(
                Section(title=item["title"], prompt=item.get("prompt", ""))
            )
        return Template(name=name, description=description, sections=sections)

    @staticmethod
    def _default_dir() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "templates")
