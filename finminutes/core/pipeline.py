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
            parts.append(f"要点数: {len(self.structured_minutes.takeaways)}")
        if self.fact_check_report:
            score = f"{self.fact_check_report.confidence_score * 100:.0f}%"
            parts.append(f"事实校验置信度: {score}")
        return "\n".join(parts)


class Pipeline:
    """Three-tier pipeline for generating financial meeting minutes.

    Modes:
        fast     — preprocess only, no LLM calls
        standard — preprocess + rewrite (LLM-based correction)
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
    ) -> PipelineResult:
        result = PipelineResult()
        result.mode = mode
        result.transcript_raw = transcript
        result.performance_tier = self._classify(transcript)

        start_total = time.time()

        if progress_callback:
            progress_callback("loading")

        try:
            background, glossary = self._load_context(background_path, glossary_tag)
        except Exception as e:
            result.errors.append(f"加载数据失败: {e}")
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
            result.errors.append(f"预处理失败: {e}")
            result.transcript_cleaned = transcript
            citation_style = "paragraph"

        if mode in ("standard", "full"):
            if progress_callback:
                progress_callback("rewrite")
            try:
                t0 = time.time()
                llm = self._config.get_llm_client()
                rewriter = Rewriter(llm, glossary, background)
                result.transcript_enhanced = rewriter.rewrite(result.transcript_cleaned)
                result.elapsed_rewrite = time.time() - t0
            except Exception as e:
                result.errors.append(f"改写失败: {e}")
                result.transcript_enhanced = result.transcript_cleaned
        else:
            result.transcript_enhanced = result.transcript_cleaned

        if mode == "full":
            if progress_callback:
                progress_callback("summarize")
            try:
                t0 = time.time()
                llm = self._config.get_llm_client()
                gen = StructuredGenerator(llm)
                result.structured_minutes = gen.generate(result.transcript_enhanced)
                result.elapsed_summarize = time.time() - t0
            except Exception as e:
                result.errors.append(f"结构化摘要失败: {e}")

            if progress_callback:
                progress_callback("factcheck")
            try:
                t0 = time.time()
                if result.structured_minutes:
                    checker = FactChecker(transcript, result.structured_minutes)
                    result.fact_check_report = checker.check(citation_style)
                result.elapsed_factcheck = time.time() - t0
            except Exception as e:
                result.errors.append(f"事实校验失败: {e}")

        if progress_callback:
            progress_callback("render")
        try:
            t0 = time.time()
            minutes = result.structured_minutes or StructuredMinutes()
            renderer = MarkdownRenderer(minutes, result.fact_check_report)
            result.markdown_output = renderer.render()
            result.elapsed_render = time.time() - t0
        except Exception as e:
            result.errors.append(f"渲染失败: {e}")
            result.markdown_output = f"# 会议纪要\n\n渲染过程中出错: {e}"

        result.elapsed_total = time.time() - start_total
        result.success = len(result.errors) == 0
        return result

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
