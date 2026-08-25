import os

import pytest

from finminutes.core.qa_parser import (
    _split_frontmatter,
    parse_qa_pairs_from_body,
    parse_sections_from_body,
    sync_frontmatter_with_body,
)


def _review_body():
    return """# 校验稿

## 主题要点

### 新一代架构
技术路线已定型，重点提升良率，目标良率约 60%。

### 高端产品进度
XX 已量产，今年可出新一代产品。

**Q**：新一代架构的良率目标是多少？
**A**：目标良率约 60%。

**Q**：高端产品进度如何？
**A**：XX 已量产。
"""


class TestParseSectionsFromBody:
    def test_parses_h3_sections(self):
        sections = parse_sections_from_body(_review_body())
        assert len(sections) == 2
        assert sections[0]["title"] == "新一代架构"
        assert "良率" in sections[0]["content"]
        assert sections[1]["title"] == "高端产品进度"

    def test_empty_when_no_marker(self):
        assert parse_sections_from_body("**Q**：x\n**A**：y") == []

    def test_content_before_first_h3_is_captured(self):
        body = "## 主题要点\n\n无标题内容\n\n### 标题\n内容"
        sections = parse_sections_from_body(body)
        # 无标题内容被兜底收纳为一个空标题小节
        assert sections[0]["title"] == ""
        assert "无标题内容" in sections[0]["content"]
        assert sections[1]["title"] == "标题"

    def test_stops_at_next_h2(self):
        body = "## 主题要点\n\n### 话题\n内容\n\n## 其他章节\n\n### 无关\n内容"
        sections = parse_sections_from_body(body)
        assert len(sections) == 1
        assert sections[0]["title"] == "话题"


class TestFrontmatterSync:
    def _review_file(self, tmp_path, body, qa_pairs=None, sections=None):
        data = {}
        if qa_pairs is not None:
            data["qa_pairs"] = qa_pairs
        if sections is not None:
            data["sections"] = sections
        fm = "---\n" if data else "---\n"
        if data:
            import yaml

            fm += yaml.safe_dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False).strip() + "\n"
        path = tmp_path / "校验稿.md"
        path.write_text(fm + "---\n\n" + body.strip() + "\n", encoding="utf-8")
        return path

    def test_sync_updates_qa_pairs_and_sections(self, tmp_path):
        path = self._review_file(
            tmp_path,
            _review_body(),
            qa_pairs=[{"question": "旧问题", "answer": "旧答案"}],
            sections=[{"title": "旧话题", "content": "旧内容"}],
        )
        # 模拟用户改了正文：Q&A 与主题要点都是新内容，frontmatter 仍是旧内容
        new_qa = [{"question": "新问题", "answer": "新答案", "asker": ""}]
        from finminutes.core.summarizer import SectionContent

        new_sections = [SectionContent(title="新话题", content="新内容")]
        sync_frontmatter_with_body(str(path), new_qa, new_sections)

        data, body = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert data["qa_pairs"][0]["question"] == "新问题"
        assert data["sections"][0]["title"] == "新话题"
        assert data["sections"][0]["content"] == "新内容"
        # 正文未被破坏（sync 只改 frontmatter，不动正文）
        assert "## 主题要点" in body
        assert "**Q**：新一代架构的良率目标是多少？" in body

    def test_sync_sections_none_preserves_frontmatter_sections(self, tmp_path):
        path = self._review_file(
            tmp_path,
            _review_body(),
            qa_pairs=[{"question": "q", "answer": "a"}],
            sections=[{"title": "保留", "content": "保留内容"}],
        )
        new_qa = [{"question": "新q", "answer": "新a", "asker": ""}]
        sync_frontmatter_with_body(str(path), new_qa, None)

        data, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
        assert data["qa_pairs"][0]["question"] == "新q"
        # sections 未传 → 保留原 frontmatter sections
        assert data["sections"][0]["title"] == "保留"


class TestParseQAPairsFromBody:
    def test_parses_standard_q_a(self):
        pairs = parse_qa_pairs_from_body(_review_body())
        assert len(pairs) == 2
        assert pairs[0]["question"] == "新一代架构的良率目标是多少？"
        assert "60%" in pairs[0]["answer"]
