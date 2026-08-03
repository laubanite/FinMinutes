import os
import tempfile

import pytest
import yaml

from finminutes.core.background_loader import BackgroundLoader
from finminutes.core.exceptions import ConfigError
from finminutes.core.models import Background


@pytest.fixture
def loader():
    return BackgroundLoader()


def test_load_example(loader):
    bg_path = os.path.join(
        os.path.dirname(__file__), "..", "finminutes", "data", "backgrounds", "example.yaml"
    )
    bg = loader.load(bg_path)
    assert isinstance(bg, Background)
    assert bg.company == "某半导体设计公司"
    assert bg.industry == "半导体"
    assert len(bg.known_consensus) == 2
    assert bg.meeting_purpose == "Q3 业绩展望及新产品路线图"


def test_load_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path.yaml")


def test_load_invalid_format():
    loader = BackgroundLoader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(["not", "a", "dict"], f)
        path = f.name
    with pytest.raises(ConfigError):
        loader.load(path)
    os.unlink(path)


def test_load_empty_file():
    loader = BackgroundLoader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({}, f)
        path = f.name
    bg = loader.load(path)
    assert isinstance(bg, Background)
    assert bg.company == ""
    os.unlink(path)


def test_known_consensus_none():
    loader = BackgroundLoader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"known_consensus": None}, f)
        path = f.name
    bg = loader.load(path)
    assert bg.known_consensus == []
    os.unlink(path)


def test_known_consensus_not_a_list():
    loader = BackgroundLoader()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"known_consensus": "not a list"}, f)
        path = f.name
    with pytest.raises(ConfigError, match="'known_consensus' must be a list"):
        loader.load(path)
    os.unlink(path)
