import os
import tempfile

import pytest
import yaml

from finminutes.core.exceptions import ConfigError
from finminutes.core.glossary_loader import GlossaryLoader
from finminutes.core.models import Glossary, Term


@pytest.fixture
def loader():
    return GlossaryLoader()


def test_load_semiconductor(loader):
    glossary = loader.load("semiconductor")
    assert glossary.industry == "半导体与存储"
    assert len(glossary.terms) >= 7


def test_load_returns_glossary(loader):
    glossary = loader.load("semiconductor")
    assert isinstance(glossary, Glossary)
    assert all(isinstance(t, Term) for t in glossary.terms)


def test_term_has_all_fields(loader):
    glossary = loader.load("semiconductor")
    term = glossary.terms[0]
    assert term.term
    assert isinstance(term.context, str)
    assert isinstance(term.corrections, list)


def test_list_tags(loader):
    tags = loader.list_tags()
    assert "semiconductor" in tags


def test_load_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_glossary")


class TestFilterTerms:
    def test_filter_returns_matching_terms(self, loader):
        glossary = loader.load("semiconductor")
        text = "HBM是先进封装的关键技术，HBM直接影响性能"
        result = GlossaryLoader.filter_terms(glossary, text, top_n=10)
        assert any(t.term == "HBM" for t in result)

    def test_filter_ordered_by_frequency(self, loader):
        glossary = loader.load("semiconductor")
        text = "HBM HBM HBM 热压键合 热压键合 EUV"
        result = GlossaryLoader.filter_terms(glossary, text, top_n=10)
        # HBM appears 3x -> top term
        assert result[0].term == "HBM"

    def test_filter_respects_top_n(self, loader):
        glossary = loader.load("semiconductor")
        text = "HBM 4F2 EUV 混合键合 存算一体 乙硼烷 六氟化钨"
        result = GlossaryLoader.filter_terms(glossary, text, top_n=3)
        assert len(result) == 3

    def test_filter_all_when_smaller_than_top_n(self, loader):
        glossary = loader.load("semiconductor")
        text = "HBM 4F2"
        result = GlossaryLoader.filter_terms(glossary, text, top_n=100)
        # Only 2 terms match, but top_n is 100, so all matching returned
        assert len(result) == 2

    def test_filter_empty_text(self, loader):
        glossary = loader.load("semiconductor")
        result = GlossaryLoader.filter_terms(glossary, "", top_n=10)
        assert result == []

    def test_filter_match_by_correction(self, loader):
        glossary = loader.load("semiconductor")
        text = "晶圆对晶圆是先进的键合工艺"  # "晶圆对晶圆" is a correction for "Wafer to wafer"
        result = GlossaryLoader.filter_terms(glossary, text, top_n=10)
        assert any(t.term == "Wafer to wafer" for t in result)


class TestParseErrors:
    def test_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not: valid: yaml: [[[")
            path = f.name
        loader = GlossaryLoader(os.path.dirname(path))
        tag = os.path.splitext(os.path.basename(path))[0]
        with pytest.raises(yaml.YAMLError):
            loader.load(tag)
        os.unlink(path)

    def test_invalid_format_not_a_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["list", "not", "dict"], f)
            path = f.name
        loader = GlossaryLoader(os.path.dirname(path))
        tag = os.path.splitext(os.path.basename(path))[0]
        with pytest.raises(ConfigError):
            loader.load(tag)
        os.unlink(path)

    def test_terms_not_a_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"industry": "test", "terms": "not a list"}, f)
            path = f.name
        loader = GlossaryLoader(os.path.dirname(path))
        tag = os.path.splitext(os.path.basename(path))[0]
        with pytest.raises(ConfigError, match="'terms' must be a list"):
            loader.load(tag)
        os.unlink(path)

    def test_term_missing_term_field(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"industry": "test", "terms": [{"context": "no term field"}]}, f)
            path = f.name
        loader = GlossaryLoader(os.path.dirname(path))
        tag = os.path.splitext(os.path.basename(path))[0]
        with pytest.raises(ConfigError, match="Invalid term entry"):
            loader.load(tag)
        os.unlink(path)
