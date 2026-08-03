from unittest.mock import MagicMock

import pytest

from finminutes.core.models import Background, Glossary, Term
from finminutes.core.rewriter import (
    Rewriter,
    chunk_text,
    find_corrections,
    levenshtein_distance,
)


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_one_substitution(self):
        assert levenshtein_distance("cat", "car") == 1

    def test_one_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") >= 3


class TestFindCorrections:
    def test_exact_match(self):
        term = Term(term="GMV", context="商品交易总额", corrections=["GMB", "GMW"])
        assert find_corrections("GMB is growing", term) == "GMV"

    def test_no_match(self):
        term = Term(term="FinFET", context="", corrections=["fin fat"])
        assert find_corrections("normal text here", term) is None

    def test_fuzzy_match(self):
        term = Term(term="流片", context="", corrections=["留片"])
        assert find_corrections("liu pian is key", term) is None  # too different

    def test_fuzzy_match_levenshtein(self):
        term = Term(term="FinFET", context="", corrections=["finfet"])
        assert find_corrections("finft is used", term) == "FinFET"


class TestChunkText:
    def test_no_chunking_needed(self):
        text = "短文本"
        assert chunk_text(text, max_chars=2000) == [text]

    def test_chunks_at_sentence_boundary(self):
        text = "第一句。" + "B" * 500 + "。第二句。" + "C" * 500
        chunks = chunk_text(text, max_chars=300)
        assert len(chunks) >= 2
        # Each chunk should end with sentence boundary or be truncated
        for c in chunks[:-1]:
            assert len(c) <= 330

    def test_overlap_between_chunks(self):
        text = "A" * 500 + "OVERLAP_CONTENT" + "B" * 500
        chunks = chunk_text(text, max_chars=400, overlap=100)
        if len(chunks) > 1:
            # Check overlap exists
            assert chunks[0][-100:] in chunks[1]

    def test_empty_text(self):
        assert chunk_text("") == [""]

    def test_short_text_returned_as_one(self):
        assert len(chunk_text("hello", max_chars=2000)) == 1


class TestRewriter:
    @pytest.fixture
    def llm_mock(self):
        m = MagicMock()
        m.generate.return_value = "rewritten content"
        return m

    @pytest.fixture
    def glossary(self):
        return Glossary(
            industry="半导体",
            terms=[
                Term(term="GMV", context="商品交易总额", corrections=["GMB", "GMW"]),
                Term(term="FinFET", context="鳍式场效应晶体管", corrections=["fin fat"]),
            ],
        )

    @pytest.fixture
    def background(self):
        return Background(
            company="某科技公司",
            industry="半导体",
            participants="CEO 王总",
            known_consensus=["公司计划明年上市"],
            meeting_purpose="Q4 业绩讨论",
        )

    def test_rewrite_calls_llm(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        result = rw.rewrite("some transcript text")
        assert result == "rewritten content"
        llm_mock.generate.assert_called_once()

    def test_prompt_contains_background(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        rw.rewrite("hello")
        prompt = llm_mock.generate.call_args[0][0]
        assert "某科技公司" in prompt
        assert "半导体" in prompt
        assert "CEO 王总" in prompt
        assert "明年上市" in prompt

    def test_prompt_contains_glossary_rules(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        rw.rewrite("hello")
        prompt = llm_mock.generate.call_args[0][0]
        assert "GMB" in prompt
        assert "GMV" in prompt
        assert "FinFET" in prompt

    def test_prompt_contains_chunk(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        rw.rewrite("the transcript text")
        prompt = llm_mock.generate.call_args[0][0]
        assert "the transcript text" in prompt

    def test_empty_transcript(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        assert rw.rewrite("") == ""
        assert rw.rewrite("   ") == ""
        llm_mock.generate.assert_not_called()

    def test_chunks_long_text(self, llm_mock, glossary, background):
        long_text = "Sentence one. " * 500  # ~7500 chars
        rw = Rewriter(llm_mock, glossary, background)
        rw.rewrite(long_text)
        assert llm_mock.generate.call_count > 1

    def test_merge_chunks_single(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        merged = rw._merge_chunks(["only chunk"])
        assert merged == "only chunk"

    def test_merge_chunks_consecutive(self, llm_mock, glossary, background):
        rw = Rewriter(llm_mock, glossary, background)
        merged = rw._merge_chunks(["chunk1", "chunk2"])
        assert "chunk1" in merged
        assert "chunk2" in merged

    def test_merge_deduplicates_overlap(self):
        rw = Rewriter(MagicMock(), Glossary(terms=[]), Background())
        merged = rw._merge_chunks(["part1 commonPart", "commonPart part3"])
        assert merged == "part1 commonPart\n part3"

    def test_background_with_empty_fields(self, llm_mock, glossary):
        bg = Background()  # all empty
        rw = Rewriter(llm_mock, glossary, bg)
        rw.rewrite("test")
        prompt = llm_mock.generate.call_args[0][0]
        assert "（无）" in prompt or "背景信息" in prompt

    def test_glossary_without_corrections(self, llm_mock):
        glossary = Glossary(
            industry="test",
            terms=[Term(term="GDP", context="国内生产总值", corrections=[])],
        )
        rw = Rewriter(llm_mock, glossary, Background())
        rw.rewrite("test")
        prompt = llm_mock.generate.call_args[0][0]
        assert "（无）" in prompt  # glossary rules section shows 无


class TestIntegration:
    @pytest.fixture
    def glossary(self):
        return Glossary(
            industry="半导体",
            terms=[
                Term(term="GMV", context="商品交易总额", corrections=["GMB", "GMW"]),
            ],
        )

    @pytest.fixture
    def background(self):
        return Background(
            company="某科技公司",
            known_consensus=["公司计划推进IPO"],
            meeting_purpose="Q4 业绩讨论",
        )

    def test_mock_integration_gmb_to_gmv(self, glossary, background):
        llm = MagicMock()
        llm.generate.return_value = "本季度 GMV 保持增长"

        rw = Rewriter(llm, glossary, background)
        result = rw.rewrite("本季度 GMB 保持增长")
        assert result == "本季度 GMV 保持增长"
