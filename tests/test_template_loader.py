import os
import tempfile

import pytest
import yaml

from finminutes.core.exceptions import ConfigError
from finminutes.core.models import Section, Template
from finminutes.core.template_loader import TemplateLoader


@pytest.fixture
def loader():
    return TemplateLoader()


def test_load_expert_interview(loader):
    template = loader.load("expert_interview")
    assert isinstance(template, Template)
    assert template.name == "expert_interview"
    assert template.description == "一级市场专家访谈纪要"
    assert len(template.sections) == 8


def test_sections_have_title_and_prompt(loader):
    template = loader.load("expert_interview")
    for section in template.sections:
        assert isinstance(section, Section)
        assert section.title
        assert section.prompt


def test_list_templates(loader):
    names = loader.list_templates()
    assert "expert_interview" in names


def test_load_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_template")


class TestParseErrors:
    def test_invalid_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "dict"], f)
            path = f.name
        tag = os.path.splitext(os.path.basename(path))[0]
        loader = TemplateLoader(os.path.dirname(path))
        with pytest.raises(ConfigError):
            loader.load(tag)
        os.unlink(path)

    def test_sections_not_a_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "test", "sections": "not a list"}, f)
            path = f.name
        tag = os.path.splitext(os.path.basename(path))[0]
        loader = TemplateLoader(os.path.dirname(path))
        with pytest.raises(ConfigError, match="'sections' must be a list"):
            loader.load(tag)
        os.unlink(path)

    def test_section_missing_title(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "test", "sections": [{"prompt": "no title"}]}, f)
            path = f.name
        tag = os.path.splitext(os.path.basename(path))[0]
        loader = TemplateLoader(os.path.dirname(path))
        with pytest.raises(ConfigError, match="Invalid section entry"):
            loader.load(tag)
        os.unlink(path)
