import glob
import os

import yaml

from finminutes.core.exceptions import ConfigError
from finminutes.core.models import Glossary, Term


class GlossaryLoader:
    def __init__(self, glossary_dir: str | None = None):
        self._dir = glossary_dir or self._default_dir()

    def load(self, tag_or_path: str) -> Glossary:
        """加载术语表：参数可以是文件路径（存在则直接加载），否则按内置标签名查找。"""
        # 路径优先：参数本身就是一个可访问的文件路径
        if os.path.isfile(tag_or_path):
            path = tag_or_path
        else:
            # tag 兜底：到自带目录查找 <tag>.yaml
            path = os.path.join(self._dir, f"{tag_or_path}.yaml")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Glossary file not found: neither path '{tag_or_path}' "
                    f"nor bundled tag '{tag_or_path}' under '{self._dir}'"
                )
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ConfigError(f"Invalid glossary format in {path}")
        return self._parse(data)

    def list_tags(self) -> list[str]:
        pattern = os.path.join(self._dir, "*.yaml")
        files = glob.glob(pattern)
        return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)

    @staticmethod
    def filter_terms(glossary: Glossary, text: str, top_n: int = 30) -> list[Term]:
        if not glossary.terms:
            return []
        text_lower = text.lower()
        scored: list[tuple[Term, int]] = []
        for term in glossary.terms:
            count = text_lower.count(term.term.lower())
            for corr in term.corrections:
                count += text_lower.count(corr.lower())
            if count > 0:
                scored.append((term, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_n]]

    def _parse(self, data: dict) -> Glossary:
        industry = data.get("industry", "")
        raw_terms = data.get("terms", [])
        if not isinstance(raw_terms, list):
            raise ConfigError("'terms' must be a list")
        terms = []
        for item in raw_terms:
            if not isinstance(item, dict) or "term" not in item:
                raise ConfigError(f"Invalid term entry: {item}")
            terms.append(
                Term(
                    term=item["term"],
                    context=item.get("context", ""),
                    corrections=item.get("corrections", []),
                )
            )
        return Glossary(industry=industry, terms=terms)

    @staticmethod
    def _default_dir() -> str:
        return os.path.join(os.path.dirname(__file__), "..", "data", "glossary")
