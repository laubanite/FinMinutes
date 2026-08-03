import time

import pytest

from finminutes.core.preprocessor import Preprocessor


@pytest.fixture
def pp():
    return Preprocessor()


class TestMergeSpeakers:
    def test_merge_consecutive_same_speaker(self, pp):
        text = "张三:\n今天天气不错\n张三:\n我们讨论一下"
        result = pp.merge_speakers(text)
        assert "张三:\n今天天气不错我们讨论一下" in result

    def test_keep_different_speakers_separate(self, pp):
        text = "张三:\n今天天气不错\n李四:\n我同意"
        result = pp.merge_speakers(text)
        assert "张三:" in result
        assert "李四:" in result

    def test_merge_multiple_same_speaker_blocks(self, pp):
        text = "A:\n1\nA:\n2\nB:\n3\nA:\n4"
        result = pp.merge_speakers(text)
        assert "A:\n12" in result
        assert "B:\n3" in result
        assert "A:\n4" in result

    def test_no_speaker_labels(self, pp):
        text = "just some text\nwithout any labels"
        assert pp.merge_speakers(text) == text

    def test_empty_text(self, pp):
        assert pp.merge_speakers("") == ""
        assert pp.merge_speakers("  ") == "  "

    def test_single_speaker(self, pp):
        text = "张三:\ncontent here"
        result = pp.merge_speakers(text)
        assert "张三:\ncontent here" in result


class TestRemoveFillers:
    def test_remove_single_filler(self, pp):
        assert pp.remove_fillers("嗯今天开会") == "今天开会"

    def test_remove_multiple_fillers(self, pp):
        assert pp.remove_fillers("嗯啊那个这个") == ""

    def test_no_fillers(self, pp):
        text = "今天天气不错"
        assert pp.remove_fillers(text) == text

    def test_empty_text(self, pp):
        assert pp.remove_fillers("") == ""
        assert pp.remove_fillers("  ") == ""

    def test_fillers_at_boundaries(self, pp):
        assert pp.remove_fillers("嗯内容") == "内容"
        assert pp.remove_fillers("内容啊") == "内容"

    def test_collapse_extra_whitespace(self, pp):
        result = pp.remove_fillers("嗯  然后  就是")
        assert result == ""

    def test_custom_fillers(self, pp):
        custom = Preprocessor(fillers=["custom", "xyz"])
        assert custom.remove_fillers("custom test xyz done") == "test done"

    def test_consecutive_fillers_cleanup(self, pp):
        result = pp.remove_fillers("嗯\n\n啊\n\n那个")
        assert result == ""


class TestNormalizeNumbers:
    def test_hundreds(self, pp):
        assert "300" in pp.normalize_numbers("三百")

    def test_thousands_with_annotation(self, pp):
        result = pp.normalize_numbers("两千五百")
        assert "2500" in result
        assert "原:两千五百" in result

    def test_ten_thousands(self, pp):
        assert "13000" in pp.normalize_numbers("一万三千")

    def test_hundred_millions(self, pp):
        result = pp.normalize_numbers("五百万")
        assert "5000000" in result
        assert "原:五百万" in result

    def test_simple_ten(self, pp):
        result = pp.normalize_numbers("十五")
        assert "15" in result

    def test_no_chinese_numbers(self, pp):
        text = "今天天气不错"
        assert pp.normalize_numbers(text) == text

    def test_empty_text(self, pp):
        assert pp.normalize_numbers("") == ""
        assert pp.normalize_numbers("  ") == "  "

    def test_mixed_text(self, pp):
        result = pp.normalize_numbers("收入三百万元")
        assert "3000000" in result

    def test_multiple_numbers(self, pp):
        result = pp.normalize_numbers("成本一百二十，收入三百")
        assert "120" in result
        assert "300" in result

    def test_big_number_with_yi(self, pp):
        result = pp.normalize_numbers("十二亿三千四百五十六万七千八百九十")
        assert "1234567890" in result


class TestClean:
    def test_full_pipeline(self, pp):
        text = "张三:\n嗯今天三百元\n张三:\n然后收入不错\n李四:\n啊是的"
        result = pp.clean(text)
        assert "张三:" in result
        assert "李四:" in result
        assert "300" in result
        assert "嗯" not in result
        assert "然后" not in result
        assert "啊" not in result

    def test_empty_text(self, pp):
        assert pp.clean("") == ""
        assert pp.clean("  ") == ""


class TestPerformance:
    def test_15k_chars_under_1_second(self, pp):
        line = "张三:\n嗯今天收入三百万元\n李四:\n啊那个成本大概五百\n"
        text = line * 500  # ~15000 chars
        start = time.time()
        pp.clean(text)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Took {elapsed:.3f}s, expected < 1s"


class TestChineseToArabic:
    def test_edge_case_zero(self, pp):
        assert pp._chinese_to_arabic("零") is None

    def test_edge_case_ten(self, pp):
        assert pp._chinese_to_arabic("十") == 10

    def test_edge_case_eleven(self, pp):
        assert pp._chinese_to_arabic("十一") == 11

    def test_edge_case_twenty(self, pp):
        assert pp._chinese_to_arabic("二十") == 20

    def test_invalid_chars(self, pp):
        assert pp._chinese_to_arabic("abc") is None

    def test_empty_string(self, pp):
        assert pp._chinese_to_arabic("") is None
