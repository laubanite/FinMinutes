import time

from finminutes.core.background_loader import BackgroundLoader
from finminutes.core.config_manager import ConfigManager
from finminutes.core.fact_checker import FactChecker, FactCheckReport
from finminutes.core.glossary_loader import GlossaryLoader
from finminutes.core.models import Background, Glossary, default_background
from finminutes.core.preprocessor import Preprocessor
from finminutes.core.renderer import MarkdownRenderer
from finminutes.core.rewriter import Rewriter
from finminutes.core.summarizer import StructuredGenerator, StructuredMinutes


class PipelineResult:
    def __init__(self):
        self.mode: str = ""
        self.transcript_raw: str = ""
        self.transcript_cleaned: str = ""
        self.transcript_enhanced: str = ""
        self.structured_minutes: StructuredMinutes | None = None
        self.fact_check_report: FactCheckReport | None = None
        self.markdown_output: str = ""

        self.elapsed_total: float = 0.0
        self.elapsed_preprocess: float = 0.0
        self.elapsed_rewrite: float = 0.0
        self.elapsed_summarize: float = 0.0
        self.elapsed_factcheck: float = 0.0
        self.elapsed_render: float = 0.0

        self.errors: list[str] = []
        self.success: bool = True
        self.aborted: bool = False  # True = 关键 LLM 阶段失败，流水线中止，不应产出文件
        self.performance_tier: str = ""

    @property
    def summary(self) -> str:
        parts = [
            f"模式: {self.mode}",
            f"性能档位: {self.performance_tier}",
            f"状态: {'成功' if self.success else '有错误'}",
            f"总耗时: {self.elapsed_total:.2f}s",
        ]
        if self.errors:
            parts.append(f"错误数: {len(self.errors)}")
        parts.append(f"原始文本: {len(self.transcript_raw)} 字符")
        parts.append(f"清洗后: {len(self.transcript_cleaned)} 字符")
        if self.transcript_enhanced:
            parts.append(f"增强后: {len(self.transcript_enhanced)} 字符")
        if self.structured_minutes:
            parts.append(f"章节数: {len(self.structured_minutes.sections)}")
            parts.append(f"问答数: {len(self.structured_minutes.qa_pairs)}")
        if self.fact_check_report:
            cov = self.fact_check_report.coverage
            if cov is not None:
                parts.append(f"内容覆盖率: {cov.ratio * 100:.0f}%")
            else:
                score = f"{self.fact_check_report.confidence_score * 100:.0f}%"
                parts.append(f"事实校验置信度: {score}")
        return "\n".join(parts)


class Pipeline:
    """Three-tier pipeline for generating financial meeting minutes.

    Modes:
        fast     — preprocess only, no LLM calls（输出清洗后转录文本）
        full     — preprocess + rewrite + summarize + fact-check + render
    """

    def __init__(self, config: ConfigManager | None = None):
        self._config = config or ConfigManager()
        self._preprocessor = Preprocessor()

    def run(
        self,
        transcript: str,
        background_path: str = "",
        glossary_tag: str = "",
        mode: str = "full",
        progress_callback: callable = None,
        on_error: callable = None,
        skip_rewrite: bool = False,
        format: str = "qa",
    ) -> PipelineResult:
        """运行流水线。

        progress_callback(stage, message="", index=None, total=None, elapsed=None) —
            阶段进度；message 用于重试等待等补充信息；index/total/elapsed 用于
            分片级进度与预计时间。
        on_error(stage, message)              — 某阶段失败即刻回调（不等待流水线跑完）。
        skip_rewrite                         — 跳过改写润色阶段（减少 LLM 调用）。
        format                               — 转录类型：
                                               "qa"    纯对话：只生成 Q&A（跳过 rewrite，省调用）
                                               "speech" 纯独白：只生成主题要点（不提取 Q&A）
                                               "both"  独白+对话：主题要点（只独白部分）+ Q&A
        """
        result = PipelineResult()
        result.mode = mode
        result.transcript_raw = transcript
        result.performance_tier = self._classify(transcript)

        start_total = time.time()

        def _fail(stage: str, label: str, exc: Exception):
            msg = f"{label}: {exc}"
            result.errors.append(msg)
            if on_error:
                on_error(stage, msg)

        fmt = (format or "qa").lower()
        if fmt not in ("qa", "speech", "both"):
            _fail("loading", "参数错误", ValueError(f"未知的 format: {format}（可选 qa/speech/both）"))
            result.success = False
            result.elapsed_total = time.time() - start_total
            return result

        # format 决定各阶段开关（仅 full 模式生效；standard 模式本质就是改写，不受 format 影响）。
        # 信息兜底：rewrite 覆盖全篇陈述 + QA 覆盖全篇问答 = 并集，分类错误不造成信息空白。
        if mode == "full":
            need_rewrite = fmt in ("speech", "both") and not skip_rewrite
            need_qa = fmt in ("qa", "both")
            build_sections = fmt in ("speech", "both")  # qa 模式不生成主题要点
        else:  # fast：仅预处理，零 LLM 调用
            need_rewrite = False
            need_qa = False
            build_sections = False

        def _retry_handler(stage: str):
            def handler(attempt: int, delay: float):
                if progress_callback:
                    progress_callback(stage, f"限流/超时，等待 {delay:.0f}s 后重试（第 {attempt} 次）")
            return handler

        def _stage_progress(stage: str):
            def handler(index: int, total: int, elapsed: float):
                if progress_callback:
                    progress_callback(stage, "", index, total, elapsed)
            return handler

        if progress_callback:
            progress_callback("loading")

        try:
            background, glossary = self._load_context(background_path, glossary_tag)
        except Exception as e:
            _fail("loading", "加载数据失败", e)
            background = Background()
            glossary = Glossary()

        if progress_callback:
            progress_callback("preprocess")
        try:
            t0 = time.time()
            result.transcript_cleaned = self._preprocessor.clean(transcript)
            citation_style = Preprocessor.detect_citation_style(transcript)
            result.elapsed_preprocess = time.time() - t0
        except Exception as e:
            _fail("preprocess", "预处理失败", e)
            result.transcript_cleaned = transcript
            citation_style = "paragraph"

        section_blocks = None  # rewrite 阶段产出的分片 topic 块（sections 保真来源）
        if need_rewrite:
            if progress_callback:
                progress_callback("rewrite")
            try:
                t0 = time.time()
                llm = self._config.get_llm_client()
                rewriter = Rewriter(
                    llm, glossary, background,
                    chunk_size=self._safe_chunk_size(),
                    on_retry=_retry_handler("rewrite"),
                    on_progress=_stage_progress("rewrite"),
                )
                result.transcript_enhanced = rewriter.rewrite(result.transcript_cleaned)
                section_blocks = rewriter.topic_blocks
                result.elapsed_rewrite = time.time() - t0
            except Exception as e:
                # 关键阶段：改写失败 → 立即中止，不继续后续阶段
                result.elapsed_rewrite = time.time() - t0
                _fail("rewrite", "改写失败", e)
                result.transcript_enhanced = result.transcript_cleaned
                result.aborted = True
                result.success = False
                result.elapsed_total = time.time() - start_total
                return result
        else:
            result.transcript_enhanced = result.transcript_cleaned

        if mode == "full":
            if progress_callback:
                progress_callback("summarize")
            try:
                t0 = time.time()
                llm = self._config.get_llm_client()
                gen = StructuredGenerator(
                    llm,
                    on_retry=_retry_handler("summarize"),
                    qa_chunk_size=self._safe_qa_chunk_size(),
                    glossary=glossary,
                    on_progress=_stage_progress("summarize"),
                )
                # 结构化输入一律用清洗后的原文（rewrite 书面化结果不参与，
                # 主题要点来自 section_blocks 的 type 标注，QA 需要原文问法）
                result.structured_minutes = gen.generate(
                    result.transcript_cleaned,
                    section_blocks=section_blocks,
                    transcript_type=self._qa_type_for(fmt),
                    build_sections=build_sections,
                    extract_qa=need_qa,
                )
                result.elapsed_summarize = time.time() - t0
            except Exception as e:
                # 关键阶段：摘要失败 → 立即中止，不继续后续阶段
                result.elapsed_summarize = time.time() - t0
                _fail("summarize", "结构化摘要失败", e)
                result.aborted = True
                result.success = False
                result.elapsed_total = time.time() - start_total
                return result

            if progress_callback:
                progress_callback("factcheck")
            try:
                t0 = time.time()
                if result.structured_minutes:
                    checker = FactChecker(result.transcript_cleaned, result.structured_minutes)
                    result.fact_check_report = checker.check(citation_style, ok_ratio=self._safe_coverage_threshold())
                result.elapsed_factcheck = time.time() - t0
            except Exception as e:
                _fail("factcheck", "事实校验失败", e)

        # fast 模式不产出校验稿（CLI 直接写清洗稿），不渲染进度事件，避免 stage_labels 缺键打印裸标签
        if progress_callback and mode != "fast":
            progress_callback("render")
        try:
            t0 = time.time()
            minutes = result.structured_minutes or StructuredMinutes()
            renderer = MarkdownRenderer(minutes, result.fact_check_report)
            result.markdown_output = renderer.render()
            result.elapsed_render = time.time() - t0
        except Exception as e:
            _fail("render", "渲染失败", e)
            result.markdown_output = f"# 会议纪要\n\n渲染过程中出错: {e}"

        result.elapsed_total = time.time() - start_total
        result.success = len(result.errors) == 0
        return result

    def _safe_coverage_threshold(self) -> float:
        try:
            v = self._config.get_coverage_threshold()
        except Exception:
            return 0.90
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.90

    def _safe_chunk_size(self) -> int:
        return self._safe_int_config("get_rewrite_chunk_size", 4000)

    def _safe_qa_chunk_size(self) -> int:
        return self._safe_int_config("get_qa_chunk_size", 8000)

    def _safe_int_config(self, method: str, default: int) -> int:
        # 注意：不能对非 int 直接 int()（MagicMock().__int__ 会静默返回 1）
        try:
            val = getattr(self._config, method)()
        except Exception:
            return default
        if isinstance(val, bool):
            return default
        if isinstance(val, int) and val > 0:
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return default

    def _load_context(self, background_path: str, glossary_tag: str) -> tuple[Background, Glossary]:
        if glossary_tag:
            loader = GlossaryLoader()
            glossary = loader.load(glossary_tag)
        else:
            glossary = Glossary()

        if background_path:
            loader = BackgroundLoader()
            background = loader.load(background_path)
        else:
            background = default_background()

        return background, glossary

    @staticmethod
    def _classify(transcript: str) -> str:
        length = len(transcript)
        if length < 1000:
            return "small"
        if length < 5000:
            return "medium"
        return "large"

    @staticmethod
    def _qa_type_for(fmt: str) -> str:
        """QA 提取的转录类型指令：qa 模式完整提取所有问答；both 模式仅提取明显问答
        （独白区不硬凑成问答对）；speech 模式不提取 QA。"""
        return "qa" if fmt == "qa" else "narration"
