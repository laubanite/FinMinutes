"""ReviewRenderer 渲染行为测试（重点：timerange 只渲染真实时间戳，过滤 LLM 脑补的垃圾值）。"""

from finminutes.core.review_renderer import ReviewRenderer
from finminutes.core.summarizer import QAPair, SectionContent, StructuredMinutes


class TestTimerangeFilter:
    def _render_qa(self, timerange: str) -> str:
        minutes = StructuredMinutes(
            sections=[],
            qa_pairs=[QAPair(question="营收多少？", answer="1.5亿元", asker="分析师", timerange=timerange)],
        )
        return ReviewRenderer(minutes).render()

    def test_timestamp_rendered(self):
        out = self._render_qa("09:32-09:35")
        assert "[09:32-09:35]" in out

    def test_timestamp_with_seconds_rendered(self):
        out = self._render_qa("09:32:15-09:35:20")
        assert "[09:32:15-09:35:20]" in out

    def test_junk_timerange_filtered(self):
        # LLM 在转录无时间戳时脑补的垃圾值，一律不渲染
        for junk in ("当前", "2023-2028", "上午", "会议中", "第1段"):
            out = self._render_qa(junk)
            assert f"[{junk}]" not in out
            assert "营收多少" in out  # QA 本身仍渲染

    def test_empty_timerange_not_rendered(self):
        out = self._render_qa("")
        assert "[" not in out  # 无时间戳时正文不含任何 [ts] 标记

    def test_normalized_junk_still_filtered(self):
        out = self._render_qa(" 当前 ")
        assert "[当前]" not in out


class TestTermReport:
    """术语纠错报告应追加在校验稿正文之后；无报告时不改变原有结构。"""

    def _minutes(self) -> StructuredMinutes:
        return StructuredMinutes(
            sections=[],
            qa_pairs=[QAPair(question="良率多少？", answer="约九成", asker="", timerange="")],
        )

    def test_term_report_appended(self):
        out = ReviewRenderer(self._minutes(), term_report="## 术语纠错\n\n- `热压结合` → **热压键合**").render()
        assert "## 术语纠错" in out
        assert "热压键合" in out
        assert out.index("**Q**") < out.index("## 术语纠错")

    def test_no_term_report_keeps_body(self):
        out = ReviewRenderer(self._minutes()).render()
        assert "## 术语纠错" not in out
        assert "**Q**：良率多少？" in out
