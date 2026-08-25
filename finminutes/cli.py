import os
import sys

import click

from finminutes import __version__
from finminutes.core.asr_client import FFmpegMissingError
from finminutes.core.config_manager import ConfigManager
from finminutes.core.exceptions import FinMinutesError
from finminutes.core.glossary_loader import GlossaryLoader
from finminutes.core.llm_client import LLMFactory
from finminutes.core.pipeline import Pipeline
from finminutes.core.review_renderer import ReviewRenderer
from finminutes.core.formal_renderer import FormalGenerator
from finminutes.core.glossary_generator import generate_glossary, save_glossary, _read_text
from finminutes.core.qa_parser import parse_qa_pairs_from_body, parse_sections_from_body, sync_frontmatter_with_body, _split_frontmatter
from finminutes.core.template_loader import TemplateLoader
from finminutes.core.background_loader import BackgroundLoader
from finminutes.core.rewriter import format_glossary_rules
from finminutes.core.fact_checker import FactChecker

_CONFIG_PATH = os.path.expanduser("~/.finminutes/config.yaml")

# 视频扩展名（转写前自动提取音轨并保留为 {stem}_音频.mp3）
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".m4v", ".flv"}

# init / config add 中「手动配置」选项的哨兵值
_MANUAL = "__manual__"


def _get_config() -> ConfigManager:
    try:
        return ConfigManager(_CONFIG_PATH)
    except FinMinutesError:
        try:
            return ConfigManager()
        except FinMinutesError:
            click.echo("错误：无法加载配置文件。请先运行 `finminutes init`。", err=True)
            sys.exit(1)


def _normalize_path(p: str, label: str = "") -> str:
    """路径归一化：把反斜杠统一转换为 /，并提示用户（Git Bash 中反斜杠易被转义吞掉）。"""
    if p and "\\" in p:
        click.echo(f"[提示]{label}路径含反斜杠，已转换为 /（Git Bash 中反斜杠易被转义吞掉）。", err=True)
        return p.replace("\\", "/")
    return p


def _resolve_output_path(source_file: str, output: str, suffix: str, stem: str | None = None) -> str:
    """统一解析 -o 输出路径规则。

    - 不指定 output: 源文件所在目录 + 源文件名 + suffix
    - output 以 / 或 反斜杠 结尾: 视为目录 -> output/源文件名+suffix
    - output 不含 / 或 反斜杠: 视为前缀 -> 源文件目录/output+suffix
    - output 包含 / 或 反斜杠: 视为目录+前缀 -> output+suffix
    """
    output = output.replace("\\", "/") if output else output
    source_dir = os.path.dirname(source_file)
    source_name = stem if stem is not None else os.path.splitext(os.path.basename(source_file))[0]

    if not output:
        return os.path.join(source_dir, source_name + suffix)
    if output.endswith("/") or output.endswith("\\"):
        return os.path.join(output, source_name + suffix)
    if "/" in output or "\\" in output:
        return output + suffix
    return os.path.join(source_dir, output + suffix)


_BACKGROUND_HINT = """[提示]未指定背景信息，将使用默认背景。如需更精准的纪要，建议使用 -b 指定背景文件。

背景文件格式示例：
  company: "某半导体设计公司"
  industry: "半导体"
  participants: "CEO 张总，CFO 李总"
  known_consensus:
    - "公司计划明年申报科创板"
  meeting_purpose: "Q3 业绩展望"

更多示例请参考: data/backgrounds/example.yaml"""


def _maybe_show_background_hint(background: str):
    if not background:
        click.echo(_BACKGROUND_HINT)


def _estimate_llm_calls(text: str, chunk_size: int = 4000) -> int:
    """估算本次 LLM 调用次数：改写分片数 + 摘要 1 次。"""
    from finminutes.core.rewriter import chunk_text
    chunks = chunk_text(text or "", max_chars=chunk_size)
    return max(len(chunks), 1) + 1


def _estimate_qa_calls(text: str, chunk_size: int = 8000) -> int:
    """估算 QA 分块提取的 LLM 调用次数。"""
    from finminutes.core.rewriter import chunk_text
    chunks = chunk_text(text or "", max_chars=chunk_size, overlap=400)
    return max(len(chunks), 1)


# ---------------------------------------------------------------------------
# main group
# ---------------------------------------------------------------------------


class _MainGroup(click.Group):
    """自定义 group：按工作流逻辑排序 --help 子命令显示顺序。"""

    _CMD_ORDER = ("config", "init", "glossary", "transcribe", "process", "render")
    _HIDDEN = {"serve"}

    def list_commands(self, ctx):
        # 固定顺序在前，其余命令按注册顺序追加；serve 保持隐藏
        others = [name for name in super().list_commands(ctx) if name not in self._CMD_ORDER and name not in self._HIDDEN]
        return list(self._CMD_ORDER) + others


@click.group(cls=_MainGroup)
@click.version_option(version=__version__, prog_name="finminutes")
def main():
    pass


# ---------------------------------------------------------------------------
# init  —  interactive cloud-first onboarding wizard
# ---------------------------------------------------------------------------


def _has_real_key(provider_cfg) -> bool:
    """provider 是否已配置可用的 API Key（非空且非环境变量占位符）。"""
    if not provider_cfg:
        return False
    api_key = provider_cfg.get("api_key", "")
    return bool(api_key) and "${" not in api_key


def _prompt_provider(kind_label: str, providers: dict, active_name: str) -> str:
    """列出可用提供商（内置 + 用户已有）供选择，返回名称或 _MANUAL。

    providers 来自合并配置（出厂默认 + 用户自定义），因此 config.yaml 里新增的
    提供商（如 siliconflow）会自动出现，用户 config add 的自定义项也会出现。
    """
    names = list(providers.keys())
    if not names:
        click.echo(f"[提示]当前没有可用的{kind_label}提供商（可在下面手动配置）。")
    for i, n in enumerate(names, 1):
        marker = " [当前]" if n == active_name else ""
        label = _provider_label(providers[n])
        click.echo(f"  [{i}] {n}{marker}  {label}" if label else f"  [{i}] {n}{marker}")
    manual_idx = len(names) + 1
    click.echo(f"  [{manual_idx}] 手动配置（添加自定义{kind_label}提供商）")
    default = names.index(active_name) + 1 if active_name in names else 1
    while True:
        choice = click.prompt("请输入编号", type=int, default=default)
        if 1 <= choice <= len(names):
            return names[choice - 1]
        if choice == manual_idx:
            return _MANUAL
        click.echo(f"[错误]编号超出范围（1-{manual_idx}），请重试。", err=True)


def _provider_label(entry: dict) -> str:
    """provider 菜单标签：免费标记 + 当前配置的模型名 + 限速（均读配置，非硬编码）。"""
    parts = []
    if entry.get("free"):
        parts.append("免费")
    model = entry.get("model", "")
    if model:
        parts.append(model)
    rate_limit = entry.get("rate_limit", "")
    if rate_limit:
        parts.append(rate_limit)
    return " · ".join(parts)


def _prompt_key_if_missing(kind_label: str, name: str, entry: dict, placeholder: str = "") -> None:
    """provider 已有可用 key 则提示保留；否则交互录入。

    placeholder 为出厂配置里该 provider 的原始环境变量占位符（如 ${OPENROUTER_API_KEY}），
    用于空 key 时展示默认值；provider 名→env 名无法可靠推导（openrouter_free 对应 OPENROUTER_API_KEY）。
    """
    api_key = entry.get("api_key", "")
    if api_key and "${" not in api_key:
        click.echo(f"  {name} 已配置 API Key（{_mask_key(api_key)}），按回车保留。")
        return
    default = placeholder or (api_key if "${" in api_key else "${%s_API_KEY}" % name.upper())
    new_key = click.prompt(
        f"请输入 {name} 的 API Key（留空则用环境变量占位）",
        default=default,
        hide_input=True,
    )
    if new_key:
        entry["api_key"] = new_key


def _test_llm_connection(provider_config) -> tuple:
    try:
        client = LLMFactory.create(provider_config)
        return client.test_connection()
    except Exception as e:
        return (False, str(e), None)


def _build_asr_client(provider_config):
    from finminutes.core.asr_client import GroqASRClient, SiliconFlowASRClient

    provider = provider_config.get("provider", "")
    if provider == "groq":
        return GroqASRClient(provider_config)
    if provider == "siliconflow":
        return SiliconFlowASRClient(provider_config)
    return None


def _test_asr_connection(provider_config) -> tuple:
    """ASR 轻量校验：GET /models 验证 key + base_url 可达性，零音频、无配额消耗。"""
    client = _build_asr_client(provider_config)
    if client is None:
        return (True, "跳过测试（未知 ASR 类型）", None)
    try:
        return client.test_connection()
    except Exception as e:
        return (False, str(e), None)


def _custom_provider_wizard(kind: str, name: str) -> dict:
    """自定义提供商全参数向导（config add 的新名称档 / init 的手动配置入口）。"""
    env_placeholder = "${%s_API_KEY}" % name.upper()
    if kind == "llm":
        fmt = click.prompt(
            "API 格式（OpenAI 兼容 / Anthropic）",
            type=click.Choice(["openai", "anthropic"]),
            default="openai",
        )
        if fmt == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": click.prompt("API Key（留空则用环境变量占位）", default=env_placeholder, hide_input=True),
                "base_url": click.prompt("Base URL", default=_DEFAULT_BASE_URLS.get("anthropic", "")),
                "model": click.prompt("模型名", default="claude-sonnet-5"),
            }
        return {
            "provider": "openai",
            "api_key": click.prompt("API Key（留空则用环境变量占位）", default=env_placeholder, hide_input=True),
            "base_url": click.prompt("Base URL", default=_DEFAULT_BASE_URLS.get("openai", "")),
            "model": click.prompt("模型名"),
        }
    # ASR：groq / siliconflow（可改 base_url / model 指向 OpenAI 兼容端点）
    ptype = click.prompt(
        "提供商类型",
        type=click.Choice(["groq", "siliconflow"]),
        default="siliconflow",
    )
    default_model = "whisper-large-v3-turbo" if ptype == "groq" else "FunAudioLLM/SenseVoiceSmall"
    return {
        "provider": ptype,
        "api_key": click.prompt("API Key（留空则用环境变量占位）", default=env_placeholder, hide_input=True),
        "model": click.prompt("模型名", default=default_model),
        "base_url": click.prompt("Base URL", default=_DEFAULT_BASE_URLS.get(ptype, "")),
        "max_file_size_mb": click.prompt("文件大小上限(MB)", type=int, default=25),
        "chunk_duration_minutes": click.prompt("切分时长(分钟)", type=int, default=10),
        "overlap_seconds": click.prompt("重叠秒数", type=int, default=5),
        "max_retries": click.prompt("重试次数", type=int, default=3),
    }


def _build_provider_entry(kind: str, name: str, merged: dict, existing_raw: dict) -> dict:
    """构建/录入一个 provider entry。

    - 名称已存在于用户配置 → 报错（用 config set 修改）；
    - 名称是内置预设（存在于合并配置） → 预设档：只填 key；
    - 否则 → 自定义档：全参数向导。
    """
    ns = f"{kind}_providers"
    existing = existing_raw.get(ns, {})
    if name in existing:
        click.echo(f"[错误]提供商 {name} 已存在，可用 `config set` 修改，或换个名字。", err=True)
        sys.exit(1)
    builtin = (merged.get(ns, {}) or {}).get(name)
    if builtin:
        entry = dict(builtin)
        _prompt_key_if_missing(kind.upper(), name, entry)
        return entry
    return _custom_provider_wizard(kind, name)


@main.command()
@click.option("--force", is_flag=True, help="覆盖已有配置，重新引导 LLM 与 ASR")
def init(force):
    """交互式初始化向导：引导 LLM（纪要后处理）与 ASR（语音转写）提供商

    提供商列表来自合并配置（出厂默认 + 用户自定义），预设只填 Key，另有手动配置入口。
    仅写入用户实际选择的 provider 与 Key（最小写），高级参数走出厂默认。
    """
    import yaml

    cfg_path = _CONFIG_PATH

    # 现有用户配置（原始 YAML）：最小写时保留用户已有项，不落盘全量默认
    existing_raw = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                existing_raw = yaml.safe_load(f) or {}
        except Exception:
            existing_raw = {}

    try:
        cm = ConfigManager(cfg_path)
        merged = cm.config
        pkg_defaults = cm._package_default()
    except FinMinutesError:
        merged = {}
        pkg_defaults = {}

    llm_providers = merged.get("llm_providers", {})
    asr_providers = merged.get("asr_providers", {})
    active_llm = merged.get("active_llm", "")
    active_asr = merged.get("active_asr", "")

    # ---- 快速路径：LLM + ASR 均已配置 Key 且 LLM 连接正常 → 已就绪直接返回 ----
    if not force:
        llm_key_ok = _has_real_key(llm_providers.get(active_llm))
        asr_key_ok = _has_real_key(asr_providers.get(active_asr))
        if llm_key_ok and asr_key_ok:
            click.echo("正在检测现有配置...")
            ok, msg, latency = _test_llm_connection(llm_providers[active_llm])
            if ok:
                click.echo(f"配置已就绪！（LLM: {active_llm} · ASR: {active_asr}，延迟 {latency}ms）")
                click.echo("如需更换提供商，请运行 `finminutes init --force`。")
                return
            click.echo(f"检测到连接失败: {msg}")
            if not click.confirm("是否重新配置？", default=True):
                return
        else:
            missing = []
            if not llm_key_ok:
                missing.append(f"LLM（{active_llm or '未设置'}）")
            if not asr_key_ok:
                missing.append(f"ASR（{active_asr or '未设置'}）")
            click.echo("检测到配置不完整：" + "、".join(missing) + " 缺少 API Key，进入配置引导...")

    # ---- 步骤 1：LLM 提供商（纪要后处理）----
    click.echo("")
    click.echo("=== LLM 提供商（用于纪要的后处理/生成）===")
    llm_name = _prompt_provider("LLM", llm_providers, active_llm)
    if llm_name == _MANUAL:
        llm_name = click.prompt("新 LLM 提供商名称", default="")
        if not llm_name:
            click.echo("未输入提供商名称，已取消。", err=True)
            sys.exit(1)
        llm_cfg = _build_provider_entry("llm", llm_name, merged, existing_raw)
        llm_providers[llm_name] = llm_cfg
    else:
        llm_cfg = dict(llm_providers[llm_name])
        _prompt_key_if_missing(
            "LLM", llm_name, llm_cfg,
            placeholder=(pkg_defaults.get("llm_providers", {}).get(llm_name, {}) or {}).get("api_key", ""),
        )
    click.echo("")
    click.echo("正在测试连接...")
    ok, msg, latency = _test_llm_connection(llm_cfg)
    if not ok:
        click.echo(f"连接测试返回: {msg}")
        if not click.confirm("连接测试未通过，仍然保存配置？", default=False):
            return
    else:
        click.echo(f"连接成功！延迟: {latency}ms")

    # ---- 步骤 2：ASR 提供商（语音转写）----
    click.echo("")
    click.echo("=== ASR 提供商（用于语音转写）===")
    asr_name = _prompt_provider("ASR", asr_providers, active_asr)
    if asr_name == _MANUAL:
        asr_name = click.prompt("新 ASR 提供商名称", default="")
        if not asr_name:
            click.echo("未输入提供商名称，已取消。", err=True)
            sys.exit(1)
        asr_cfg = _build_provider_entry("asr", asr_name, merged, existing_raw)
        asr_providers[asr_name] = asr_cfg
    else:
        asr_cfg = dict(asr_providers[asr_name])
        _prompt_key_if_missing(
            "ASR", asr_name, asr_cfg,
            placeholder=(pkg_defaults.get("asr_providers", {}).get(asr_name, {}) or {}).get("api_key", ""),
        )
    click.echo("")
    click.echo("正在测试连接...")
    ok, msg, latency = _test_asr_connection(asr_cfg)
    if not ok:
        click.echo(f"连接测试返回: {msg}")
        if not click.confirm("连接测试未通过，仍然保存配置？", default=False):
            return
    else:
        click.echo(f"连接成功！延迟: {latency}ms")

    # ---- 最小写：只更新选中的 provider + active + logging ----
    existing_raw.setdefault("llm_providers", {})[llm_name] = llm_cfg
    existing_raw["active_llm"] = llm_name
    existing_raw.setdefault("asr_providers", {})[asr_name] = asr_cfg
    existing_raw["active_asr"] = asr_name
    existing_raw.setdefault("logging", {
        "level": "INFO",
        "file": "finminutes.log",
        "redact_transcript": True,
        "redact_api_key": True,
    })

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(existing_raw, f, allow_unicode=True, default_flow_style=False)

    click.echo("")
    click.echo(f"配置文件已保存至: {cfg_path}")
    click.echo("")
    click.echo("后续使用指南：")
    click.echo("  finminutes process -t 转录文件.txt    # 文本 → 校验稿")
    click.echo("  finminutes transcribe -a 音频.m4a     # 音频 → 转录 → 校验稿")
    click.echo("")
    click.echo("高级参数（默认已适配多数场景，一般无需改动）：")
    click.echo("  查看全部配置：  finminutes config show --all")
    click.echo("  修改已有项：    finminutes config set <点路径> <值>")
    click.echo("     例：finminutes config set asr_providers.groq.max_file_size_mb 50")
    click.echo("  新增提供商：    finminutes config add asr/llm <名称>")
    click.echo("")
    click.echo("⚠️ 配置分层提醒：本工具优先读取 ~/.finminutes/config.yaml（系统盘用户配置）；")
    click.echo("  项目目录里的 finminutes/config.yaml 只是出厂默认值，会被用户配置覆盖。")
    click.echo("  改高级参数请用上面的命令，或直接编辑 ~/.finminutes/config.yaml，不要改项目文件。")


# ---------------------------------------------------------------------------
# process  —  run the full pipeline
# ---------------------------------------------------------------------------


@main.command()
@click.option("--transcript", "-t", required=True, help="转录文本文件路径")
@click.option("--background", "-b", default="", help="背景信息 YAML 文件路径")
@click.option("--glossary", "-g", default="", help="术语表 YAML 文件路径，或内置标签名（如 semiconductor）")
@click.option(
    "--mode", "-m",
    default="full",
    type=click.Choice(["fast", "full"], case_sensitive=False),
    help="流水线模式：fast 仅清洗转录输出清洗稿（零 LLM），full 完整流水线（默认）",
)
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 转录稿所在目录）")
@click.option("--skip-rewrite", is_flag=True, help="跳过改写润色阶段（减少 LLM 调用，适合长文本/限流场景）")
@click.option(
    "--format", "-f", "fmt",
    default="qa",
    type=click.Choice(["qa", "speech", "both"], case_sensitive=False),
    help="转录类型：qa 纯对话（只 Q&A，默认）/ speech 纯独白（只主题要点）/ both 独白+对话（主题要点+Q&A）",
)
def process(transcript, background, glossary, mode, output, skip_rewrite, fmt):
    """从转录稿生成校验稿"""
    transcript = _normalize_path(transcript, "-t ")
    background = _normalize_path(background, "-b ")
    glossary = _normalize_path(glossary, "-g ")
    output = _normalize_path(output, "-o ")
    if not os.path.exists(transcript):
        click.echo(f"错误：转录文件不存在: {transcript}", err=True)
        click.echo("提示：若你使用了 \\ 反斜杠路径，请改用 / 或相对路径（Git Bash 中反斜杠会被转义吞掉）。", err=True)
        sys.exit(1)

    with open(transcript, "r", encoding="utf-8") as f:
        text = f.read()

    cfg = _get_config()
    pipeline = Pipeline(cfg)

    if mode == "full" and not _ensure_llm_config(cfg):
        sys.exit(1)

    if mode == "full":
        if fmt == "qa":
            est = _estimate_qa_calls(text, chunk_size=cfg.get_qa_chunk_size())
            if est > 20:
                click.echo(f"[提示]预计本次 LLM 调用约 {est} 次（QA 分块提取）。OpenRouter 免费档日配额 50 次。", err=True)
        elif not skip_rewrite:
            est = _estimate_llm_calls(text, chunk_size=cfg.get_rewrite_chunk_size())
            if est > 20:
                click.echo(f"[提示]预计本次 LLM 调用约 {est} 次（改写分片 + 摘要）。OpenRouter 免费档日配额 50 次，长文本可加 --skip-rewrite 减半。", err=True)

    _maybe_show_background_hint(background)

    stage_labels = {
        "loading": "[1/6] 加载数据...",
        "preprocess": "[2/6] 预处理文本...",
        "rewrite": "[3/6] 改写润色...",
        "summarize": "[4/6] 生成摘要...",
        "factcheck": "[5/6] 事实校验...",
        "render": "[6/6] 渲染输出...",
        "done": "处理完成。",
    }
    if mode == "fast":
        stage_labels = {
            "loading": "[1/2] 加载数据...",
            "preprocess": "[2/2] 预处理文本...",
            "done": "清洗完成。",
        }
    elif mode == "full" and (fmt == "qa" or skip_rewrite):
        stage_labels = {
            "loading": "[1/5] 加载数据...",
            "preprocess": "[2/5] 预处理文本...",
            "summarize": "[3/5] 生成摘要...",
            "factcheck": "[4/5] 事实校验...",
            "render": "[5/5] 渲染输出...",
            "done": "处理完成。",
        }

    if not background and not glossary and mode == "full" and len(text) > 20000:
        click.echo("[提示]提示：长文本且未提供术语表/背景，可用 --skip-rewrite 跳过改写阶段，LLM 调用约减半。", err=True)

    def _on_progress(stage: str, message: str = "", index: int = None, total: int = None, elapsed: float = None):
        label = stage_labels.get(stage, stage)
        if index is not None and total:
            # 分片级进度 + 滚动预计时间（LLM 生成速度同篇内较稳定，2-3 片后估算收敛）
            done = max(index, 1)
            elapsed_total = float(elapsed or 0)
            per = elapsed_total / done
            remain = per * (total - index)
            if remain < 60:
                remain_text = f"~{remain:.0f}s"
            else:
                remain_text = f"~{remain / 60:.1f}分钟"
            click.echo(f"\r  {label} {index}/{total} 片 · 已用 {elapsed_total:.0f}s · 预计还需 {remain_text}", err=True, nl=False)
            if index >= total:
                click.echo("", err=True)  # 该阶段完成，换行
        elif message:
            click.echo(f"  {label}（{message}）", err=True)
        else:
            click.echo(f"  {label}", err=True)

    try:
        result = pipeline.run(
            transcript=text,
            background_path=background,
            glossary_tag=glossary,
            mode=mode,
            skip_rewrite=skip_rewrite,
            format=fmt,
            progress_callback=_on_progress,
            on_error=lambda stage, msg: click.echo(f"  [失败] {msg}", err=True),
        )
    except Exception as e:
        click.echo(f"处理失败: {e}", err=True)
        sys.exit(1)

    if result.aborted:
        click.echo("", err=True)
        click.echo("处理中止：关键阶段失败，未生成校验稿。", err=True)
        sys.exit(1)

    _on_progress("done")

    if mode == "fast":
        # fast：仅清洗，输出清洗后纯文本（零 LLM 调用）
        cleaned = result.transcript_cleaned or text
        clean_path = _resolve_output_path(transcript, output, "_清洗稿.txt")
        os.makedirs(os.path.dirname(clean_path) or ".", exist_ok=True)
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        click.echo(f"清洗稿已输出至: {clean_path}")
        if not result.success:
            click.echo("", err=True)
            click.echo("处理过程中出现以下问题：", err=True)
            for err in result.errors:
                click.echo(f"  - {err}", err=True)
        return

    minutes = result.structured_minutes
    if minutes is None:
        from finminutes.core.summarizer import StructuredMinutes
        minutes = StructuredMinutes()

    if mode == "full" and fmt != "speech" and not minutes.qa_pairs:
        click.echo("", err=True)
        click.echo("[警告]未提取到问答对。若原文确有问答，可能是 QA 提取失败；", err=True)
        click.echo("       可重新运行，或在渲染成品稿前人工补充。", err=True)

    report = result.fact_check_report
    if report is not None and report.coverage is not None:
        cov = report.coverage
        covered = cov.total_chunks - cov.uncovered_chunks
        click.echo(f"[完整度] {cov.ratio * 100:.0f}%（{covered}/{cov.total_chunks} 段转录被提取内容覆盖）")
        if not cov.ok:
            click.echo(
                f"[警告]完整度低于阈值 {cov.ok_ratio * 100:.0f}%，可能存在内容遗漏，请对照录音核对未覆盖片段。",
                err=True,
            )

    review_content = ReviewRenderer(minutes, report, result.transcript_raw).render()

    review_path = _resolve_output_path(transcript, output, "_校验稿.md")
    os.makedirs(os.path.dirname(review_path) or ".", exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(review_content)
    click.echo(f"校验稿已输出至: {review_path}")

    if not result.success:
        click.echo("", err=True)
        click.echo("处理过程中出现以下问题：", err=True)
        for err in result.errors:
            click.echo(f"  - {err}", err=True)


# ---------------------------------------------------------------------------
# render  —  render formal draft from review draft
# ---------------------------------------------------------------------------


@main.command()
@click.option("--review", "-r", default="", help="校验稿文件路径（与 -j 二选一）")
@click.option("--json", "-j", "json_path", default="", help="网页版 AI 返回的原始 JSON（sections/qa_pairs），直接渲染成品稿（与 -r 二选一）")
@click.option("--template", "-T", default="default", help="成品稿模板名称（默认使用 default）")
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 输入文件所在目录）")
def render(review, json_path, template, output):
    """从校验稿渲染成品稿"""
    import yaml
    import os

    review = _normalize_path(review, "-r ")
    json_path = _normalize_path(json_path, "-j ")
    output = _normalize_path(output, "-o ")
    if bool(review) == bool(json_path):
        click.echo("错误：-r（校验稿）与 -j（网页版 AI 原始 JSON）二选一，必须且只能提供一个。", err=True)
        sys.exit(1)

    input_path = json_path or review
    if not os.path.exists(input_path):
        click.echo(f"错误：输入文件不存在: {input_path}", err=True)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    if json_path:
        # 网页版 AI 原始 JSON 直接渲染，不经校验稿中间格式
        data = _parse_web_result(content)
        minutes = _build_minutes_from_result(data)
        if not minutes.sections and not minutes.qa_pairs:
            click.echo("错误：无法从结果中解析出 Q&A 或主题要点。请确认网页版 AI 按输出要求返回了 JSON。", err=True)
            sys.exit(1)
        # FormalGenerator 的 QA 输入是 dict（内部 .get），sections 是 SectionContent 对象
        qa_pairs = [
            {"question": q.question, "answer": q.answer, "asker": q.asker, "timerange": q.timerange}
            for q in minutes.qa_pairs
        ]
        sections = minutes.sections
    else:
        data, body = _split_frontmatter(content)
        if data is None:
            click.echo("错误：校验稿格式不正确（缺少 YAML 头）", err=True)
            sys.exit(1)

        from finminutes.core.summarizer import SectionContent
        raw_qa = data.get("qa_pairs", [])
        raw_sections = data.get("sections", [])

        def _to_sections(raw):
            return [
                SectionContent(
                    title=s.get("title", ""),
                    content=s.get("content", ""),
                    citations=s.get("citations", []),
                )
                for s in raw if isinstance(s, dict)
            ]

        # 校验稿正文是用户手动编辑的权威来源：上方 Front Matter（properties）可能落后于正文，
        # 因此正文能解析出 Q&A 或主题要点时，一律以正文为准，并把 Front Matter 回写为正文最新内容。
        body_pairs = parse_qa_pairs_from_body(body)
        body_sections = parse_sections_from_body(body)
        has_section_marker = "## 主题要点" in body

        # —— Q&A：正文优先 ——
        if body_pairs:
            qa_pairs = body_pairs
        elif not raw_qa and not raw_sections:
            click.echo(
                "错误：校验稿中未找到任何 Q&A 数据或主题要点（正文解析失败，Front Matter 也为空）。"
                "请检查校验稿格式是否正确。",
                err=True,
            )
            sys.exit(1)
        elif not raw_qa:
            click.echo("[警告]正文无 Q&A 数据，使用 Front Matter 中的主题要点（纯独白模式）。", err=True)
            qa_pairs = raw_qa
        else:
            click.echo("[警告]正文解析失败，使用 Front Matter 中的数据。请检查校验稿正文格式。", err=True)
            qa_pairs = raw_qa

        # —— 主题要点（sections）：正文优先；正文含标记但解析失败则回退 Front Matter ——
        if has_section_marker and body_sections:
            sections = [
                SectionContent(title=s["title"], content=s["content"], citations=s.get("citations", []))
                for s in body_sections
            ]
        elif has_section_marker and raw_sections:
            click.echo("[警告]正文主题要点解析失败，使用 Front Matter 中的主题要点。", err=True)
            sections = _to_sections(raw_sections)
        else:
            sections = _to_sections(raw_sections)

        # 正文权威（能解析出 Q&A 或主题要点）时，把 Front Matter 同步为正文最新内容
        if body_pairs or (has_section_marker and body_sections):
            sync_frontmatter_with_body(
                review,
                qa_pairs,
                sections if (has_section_marker and body_sections) else None,
            )

    try:
        loader = TemplateLoader(subdir="formal")
        tmpl = loader.load(template)
    except FileNotFoundError:
        click.echo(f"错误：成品稿模板不存在: {template}", err=True)
        click.echo(f"可用模板: {', '.join(loader.list_templates())}", err=True)
        sys.exit(1)

    cfg = _get_config()
    if not _ensure_llm_config(cfg):
        sys.exit(1)
    try:
        llm = cfg.get_llm_client()
    except Exception as e:
        click.echo(f"错误：获取 LLM 客户端失败: {e}", err=True)
        sys.exit(1)

    click.echo(f"正在调用 LLM 生成成品稿（{len(qa_pairs)} 组 Q&A，{len(sections)} 条主题要点，模板: {template}）...")
    generator = FormalGenerator(llm)
    try:
        formal_output = generator.generate(qa_pairs, tmpl, sections=sections)
    except Exception as e:
        click.echo(f"错误：LLM 生成失败: {e}", err=True)
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(input_path))[0]
    stem = stem.replace("_校验稿", "")
    formal_path = _resolve_output_path(input_path, output, "_成品稿.md", stem=stem)
    os.makedirs(os.path.dirname(formal_path) or ".", exist_ok=True)
    with open(formal_path, "w", encoding="utf-8") as f:
        f.write(formal_output)
    click.echo(f"成品稿已输出至: {formal_path}")


# ---------------------------------------------------------------------------
# export-prompt / import-result — 网页版 AI 协同（豆包/DeepSeek 等，0 成本、可反复调试）
# ---------------------------------------------------------------------------


def _load_background_text(path: str) -> str:
    if not path:
        return "（无，使用默认背景）"
    try:
        bg = BackgroundLoader().load(path)
    except Exception:
        return f"（背景文件加载失败: {path}）"
    parts = []
    for key, val in [("公司", bg.company), ("行业", bg.industry), ("参与者", bg.participants), ("会议目的", bg.meeting_purpose)]:
        if val:
            parts.append(f"{key}：{val}")
    for item in bg.known_consensus:
        parts.append(f"已知共识：{item}")
    return "\n".join(parts) or "（无）"


def _build_export_prompt(transcript: str, background_text: str, glossary_text: str, fmt: str) -> str:
    parts = [
        "你是金融会议纪要整理专家。请把下面的会议转录整理成【校验稿】，用于用户对照录音逐条核对。",
        "",
        "【输出格式要求（严格遵守，只输出 JSON，不要任何其他文字或 markdown 标记）】",
    ]
    if fmt == "qa":
        parts.append('{"qa_pairs": [{"question": "问题", "answer": "书面化回答（保留全部数字、金额、公司名与事实）", "asker": "提问者", "timerange": ""}]}')
        parts.append("- 按时间顺序提取每一个提问及其回答（含重复、跟进提问），数量与转录中的提问数基本一致")
        parts.append("- answer 用书面语改写，去除口语化（嗯/啊/就是说/蛮/其实等），但数字事实一条不能丢")
    elif fmt == "speech":
        parts.append('{"sections": [{"title": "话题标题", "content": "该话题的书面化内容（保留全部数字与事实）"}]}')
        parts.append("- 按出现顺序把内容聚合为 2-4 个较大的话题块，不要切得过细")
        parts.append("- 书面化改写，信息零丢失")
    else:  # both
        parts.append('{"sections": [{"title": "话题标题", "content": "独白部分的书面化内容（保留全部数字与事实）"}], "qa_pairs": [{"question": "问题", "answer": "书面化回答", "asker": "提问者", "timerange": ""}]}')
        parts.append("- sections 只针对大段单向陈述/介绍（独白）部分；问答互动放入 qa_pairs")
        parts.append("- 独白部分书面化改写保留全部数字事实；问答按时间顺序提取，answer 书面化且保留全部数字事实")
    parts += ["", "【背景信息】", background_text, "", "【术语纠错规则】", glossary_text, "", "【转录全文】", transcript]
    return "\n".join(parts)


@main.command()
@click.option("--transcript", "-t", required=True, help="转录稿文件路径")
@click.option("--background", "-b", default="", help="背景信息 YAML 文件路径")
@click.option("--glossary", "-g", default="", help="术语表 YAML 文件路径，或内置标签名（如 semiconductor）")
@click.option(
    "--format", "-f", "fmt",
    default="qa",
    type=click.Choice(["qa", "speech", "both"], case_sensitive=False),
    help="转录类型：qa 纯对话 / speech 纯独白 / both 独白+对话",
)
@click.option("--output", "-o", default="", help="输出路径（默认: 转录稿同目录 _prompt.txt）")
def export_prompt(transcript, background, glossary, fmt, output):
    """导出可直接粘贴/上传给网页版 AI 的 prompt """
    transcript = _normalize_path(transcript, "-t ")
    background = _normalize_path(background, "-b ")
    glossary = _normalize_path(glossary, "-g ")
    output = _normalize_path(output, "-o ")
    if not os.path.exists(transcript):
        click.echo(f"错误：转录文件不存在: {transcript}", err=True)
        sys.exit(1)
    with open(transcript, "r", encoding="utf-8") as f:
        text = f.read()

    glossary_text = format_glossary_rules(GlossaryLoader().load(glossary)) if glossary else "（无）"
    bg_text = _load_background_text(background)

    prompt_text = _build_export_prompt(text, bg_text, glossary_text, fmt)

    prompt_path = _resolve_output_path(transcript, output, "_prompt.txt")
    os.makedirs(os.path.dirname(prompt_path) or ".", exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    click.echo(f"prompt 包已输出至: {prompt_path}")

    click.echo("")
    click.echo("【网页版协同使用指引】", err=True)
    click.echo(f"  1. 打开豆包 / DeepSeek 网页版等 AI，把上面的 prompt 包文件拖进去（支持上传 txt），或复制全文粘贴", err=True)
    click.echo("  2. 等待网页版输出结果，在结果页『复制全部』", err=True)
    click.echo("  3. 本地新建文本文件粘贴保存（如 result.txt）", err=True)
    click.echo("  4. 运行：finminutes import-result -r result.txt [--source 原转录稿]", err=True)
    click.echo("  （网页版 0 成本，可反复调试直到满意）", err=True)


def _parse_web_result(text: str) -> dict:
    """解析网页版 AI 返回：优先 JSON（sections/qa_pairs），兜底整段作为内容。"""
    import json
    import re

    t = (text or "").strip()
    candidates = [t]
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                if "qa_pairs" in data or "sections" in data:
                    return data
            elif isinstance(data, list):
                if all(isinstance(x, dict) and ("question" in x or "answer" in x) for x in data):
                    return {"qa_pairs": data}
                if all(isinstance(x, dict) and ("title" in x or "content" in x) for x in data):
                    return {"sections": data}
        except Exception:
            continue
    return {"sections": [{"title": "内容", "content": t}]}


def _build_minutes_from_result(data: dict):
    from finminutes.core.summarizer import StructuredMinutes, QAPair, SectionContent
    sections = [
        SectionContent(title=s.get("title", ""), content=s.get("content", ""))
        for s in data.get("sections", []) if isinstance(s, dict)
    ]
    qa_pairs = [
        QAPair(question=q.get("question", ""), answer=q.get("answer", ""),
               asker=q.get("asker", ""), timerange=q.get("timerange", ""))
        for q in data.get("qa_pairs", []) if isinstance(q, dict)
    ]
    return StructuredMinutes(sections=sections, qa_pairs=qa_pairs)


def _coverage_threshold() -> float:
    """读取内容完整度阈值；配置不可用时回落 0.90。"""
    try:
        from finminutes.core.config_manager import ConfigManager
        return ConfigManager().get_coverage_threshold()
    except Exception:
        return 0.90


def _minutes_to_dict(minutes) -> dict:
    return {
        "sections": [
            {"title": s.title, "content": s.content, "citations": s.citations}
            for s in minutes.sections
        ],
        "qa_pairs": [
            {"question": q.question, "answer": q.answer, "asker": q.asker, "timerange": q.timerange}
            for q in minutes.qa_pairs
        ],
    }


def _build_debug_prompt(minutes, uncovered_snippets: list[str]) -> str:
    """构造网页版 AI 调试 prompt：携带当前已提取内容 + 未覆盖转录片段，要求合并补全输出完整 JSON。"""
    import json

    has_sections = bool(minutes.sections)
    has_qa = bool(minutes.qa_pairs)
    if has_sections and has_qa:
        schema = '{"sections": [{"title": "...", "content": "书面化内容（保留全部数字与事实）"}], "qa_pairs": [{"question": "...", "answer": "书面化回答（保留全部数字、金额、公司名与事实）", "asker": "提问者", "timerange": ""}]}'
    elif has_sections:
        schema = '{"sections": [{"title": "...", "content": "书面化内容（保留全部数字与事实）"}]}'
    else:
        schema = '{"qa_pairs": [{"question": "...", "answer": "书面化回答（保留全部数字、金额、公司名与事实）", "asker": "提问者", "timerange": ""}]}'

    current = json.dumps(_minutes_to_dict(minutes), ensure_ascii=False, indent=1)
    parts = [
        "你是金融会议纪要整理专家。上一次网页版 AI 提取的纪要内容完整度不足，以下转录片段未被提取内容覆盖（尤其数字、金额、公司名、事件）。",
        "",
        "【任务】把未覆盖片段的信息补入现有内容，并输出一份【包含全部已有内容 + 新增内容】的完整 JSON，直接覆盖式替换上一次的结果。",
        "",
        "【输出格式要求（只输出 JSON，不要任何其他文字或 markdown 标记）】",
        schema,
        "",
        "【当前已提取内容（请保留全部，在此基础上补充）】",
        current,
        "",
        f"【未覆盖转录片段（共 {len(uncovered_snippets)} 段，请把其中信息补入，不得遗漏）】",
    ]
    for i, snip in enumerate(uncovered_snippets, 1):
        parts.append(f"\n片段{i}：\n{snip}")
    parts.append("\n请直接输出合并补全后的完整 JSON：")
    return "\n".join(parts)


@main.command()
@click.option("--result", "-r", required=True, help="网页版 AI 返回结果文件路径（txt/md）")
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 结果文件所在目录）")
@click.option("--source", "-s", "source_transcript", default="", help="原始转录稿路径（用于内容完整度校验；未提供则跳过）")
def import_result(result, output, source_transcript):
    """把网页版 AI 生成的结果解析成标准校验稿，并报告内容完整度"""
    result = _normalize_path(result, "-r ")
    source_transcript = _normalize_path(source_transcript, "-s ")
    if not os.path.exists(result):
        click.echo(f"错误：结果文件不存在: {result}", err=True)
        sys.exit(1)
    with open(result, "r", encoding="utf-8") as f:
        text = f.read()

    data = _parse_web_result(text)
    minutes = _build_minutes_from_result(data)
    if not minutes.sections and not minutes.qa_pairs:
        click.echo("错误：无法从结果中解析出 Q&A 或主题要点。请确认网页版 AI 按输出要求返回了 JSON。", err=True)
        sys.exit(1)

    transcript_raw = ""
    report = None
    if source_transcript:
        try:
            from finminutes.core.preprocessor import Preprocessor
            raw = open(source_transcript, "r", encoding="utf-8").read()
            transcript_raw = Preprocessor().clean(raw)
            checker = FactChecker(transcript_raw, minutes)
            report = checker.check("paragraph", ok_ratio=_coverage_threshold())
        except Exception as e:
            click.echo(f"[警告]事实校验失败: {e}", err=True)
    else:
        click.echo("[提示]未提供 -s 原始转录稿，无法校验内容完整度。建议加 -s 以获得完整度报告与调试 prompt。", err=True)

    review_content = ReviewRenderer(minutes, report, transcript_raw).render()
    review_path = _resolve_output_path(result, output, "_校验稿.md")
    os.makedirs(os.path.dirname(review_path) or ".", exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(review_content)
    click.echo(f"已解析 {len(minutes.qa_pairs)} 组 Q&A，{len(minutes.sections)} 条主题要点。")
    click.echo(f"校验稿已输出至: {review_path}")

    if report is not None and report.coverage is not None:
        cov = report.coverage
        covered = cov.total_chunks - cov.uncovered_chunks
        click.echo(f"[完整度] {cov.ratio * 100:.0f}%（{covered}/{cov.total_chunks} 段转录被提取内容覆盖）")
        if not cov.ok:
            debug_path = _resolve_output_path(result, output, "_调试prompt.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(_build_debug_prompt(minutes, cov.uncovered_snippets))
            click.echo(
                f"[警告]完整度低于阈值 {cov.ok_ratio * 100:.0f}%，已生成调试 prompt: {debug_path}",
                err=True,
            )
            click.echo("       把该 prompt 拖回网页版 AI，将补全后的完整 JSON 重新 import-result 即可。", err=True)


# ---------------------------------------------------------------------------
# config  —  configuration management
# ---------------------------------------------------------------------------


@main.group()
def config():
    """配置管理（show / set / set-key / add / list）"""
    pass


@config.command(name="show")
@click.option("--all", "-a", "show_all", is_flag=True, help="显示全部提供商配置（Key 脱敏）")
def config_show(show_all):
    """显示配置（默认激活项；--all 全量，Key 脱敏）"""
    cfg = _get_config()
    data = cfg.config

    active_llm = cfg.get_active_llm()
    active_asr = cfg.get_active_asr()

    click.echo("")
    click.echo("=== 当前激活配置 ===")
    click.echo("")

    llm_providers = data.get("llm_providers", {})
    asr_providers = data.get("asr_providers", {})

    llm_cfg = llm_providers.get(active_llm, {})
    click.echo(f"[LLM] {active_llm}")
    for k, v in llm_cfg.items():
        if k == "api_key" and v:
            v = _mask_key(v)
        click.echo(f"  {k}: {v}")
    click.echo("")

    asr_cfg = asr_providers.get(active_asr, {})
    click.echo(f"[ASR] {active_asr}")
    for k, v in asr_cfg.items():
        if k == "api_key" and v:
            v = _mask_key(v)
        click.echo(f"  {k}: {v}")

    if show_all:
        click.echo("")
        click.echo("=== 全部 LLM 提供商 ===")
        for name, opts in llm_providers.items():
            marker = " [当前]" if name == active_llm else ""
            click.echo(f"[{name}]{marker}")
            for k, v in opts.items():
                if k == "api_key" and v:
                    v = _mask_key(v)
                click.echo(f"  {k}: {v}")
            click.echo("")
        click.echo("=== 全部 ASR 提供商 ===")
        for name, opts in asr_providers.items():
            marker = " [当前]" if name == active_asr else ""
            click.echo(f"[{name}]{marker}")
            for k, v in opts.items():
                if k == "api_key" and v:
                    v = _mask_key(v)
                click.echo(f"  {k}: {v}")
            click.echo("")
        click.echo("提示：可用 `config set <点路径> <值>` 修改，如 asr_providers.groq.chunk_duration_minutes 15")


def _parse_config_value(s: str):
    """命令行值类型推断：bool / int / float / 字符串原样（保留 ${ENV} 占位符）。"""
    s = s.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """设置配置项，支持点路径

    \b
    示例：finminutes config set active_llm deepseek
          finminutes config set active_asr siliconflow
          finminutes config set coverage_threshold 0.6
          finminutes config set asr_providers.groq.chunk_duration_minutes 15
          finminutes config set asr_providers.groq.overlap_seconds 5
    """
    cfg = _get_config()
    if key in ("active_llm", "active_asr"):
        if key == "active_llm":
            cfg.set_active_llm(value)
            cfg.save()
            click.echo(f"已设置 {key} = {value}")
            _warn_missing_llm_key(cfg, value)
        else:
            cfg.set_active_asr(value)
            cfg.save()
            click.echo(f"已设置 {key} = {value}")
        return
    parsed = _parse_config_value(value)
    ok, msg = cfg.set_by_path(key, parsed)
    if not ok:
        click.echo(f"[错误]{msg}", err=True)
        sys.exit(1)
    cfg.save()
    click.echo(f"已设置 {key} = {parsed}")


def _find_provider_namespace(cfg, name: str) -> str | None:
    """返回提供商 name 所属的命名空间：llm_providers / asr_providers / None。"""
    for ns in ("llm_providers", "asr_providers"):
        if name in cfg.config.get(ns, {}):
            return ns
    return None


def _set_provider_key(cfg, name: str, ns: str, label: str = "") -> bool:
    """交互录入并写回指定 provider 的 api_key。返回是否成功。"""
    providers = cfg.config.setdefault(ns, {})
    providers.setdefault(name, {})
    api_key = click.prompt(f"请输入 {label or name} 的 API Key", hide_input=True)
    if not api_key:
        return False
    providers[name]["api_key"] = api_key
    cfg.save()
    return True


@config.command(name="set-key")
@click.argument("provider")
def config_set_key(provider):
    """设置指定 LLM 或 ASR 提供商的 API Key（自动识别）

    \b
    示例：finminutes config set-key deepseek
          finminutes config set-key groq
          finminutes config set-key siliconflow
    """
    cfg = _get_config()
    ns = _find_provider_namespace(cfg, provider)
    if ns is None:
        click.echo(f"[错误]未知的提供商: {provider}", err=True)
        click.echo(f"   可用（LLM）: {', '.join(cfg.config.get('llm_providers', {}).keys()) or '（无）'}", err=True)
        click.echo(f"   可用（ASR）: {', '.join(cfg.config.get('asr_providers', {}).keys()) or '（无）'}", err=True)
        sys.exit(1)
    if not _set_provider_key(cfg, provider, ns):
        click.echo("未输入 API Key，已取消。", err=True)
        sys.exit(1)
    click.echo(f"已保存 {provider} 的 API Key。")


def _warn_missing_llm_key(cfg, provider: str = ""):
    provider = provider or cfg.get_active_llm()
    try:
        pc = cfg.get_llm_config()
    except FinMinutesError:
        return
    api_key = pc.get("api_key", "")
    if not api_key or "${" in api_key:
        click.echo("")
        click.echo(f"[警告]当前 LLM（{provider}）的 API Key 未配置。")
        click.echo(f"   请运行 `finminutes config set-key {provider}` 配置后重试。")


def _ensure_llm_config(cfg) -> bool:
    """检查 active LLM 是否已配置有效 API Key（非交互，缺失即报错并返回 False）。"""
    try:
        active = cfg.get_active_llm()
        pc = cfg.get_llm_config()
    except FinMinutesError:
        return True
    api_key = pc.get("api_key", "")
    if api_key and "${" not in api_key:
        return True
    click.echo("")
    click.echo(f"[警告]当前 LLM（{active}）的 API Key 未配置。", err=True)
    click.echo(f"   请运行 `finminutes config set-key {active}` 或 `finminutes init` 配置后重试。", err=True)
    return False


_DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
}


@config.command(name="add")
@click.argument("kind", type=click.Choice(["llm", "asr"]))
@click.argument("name")
def config_add(kind, name):
    """交互式新增/配置 LLM 或 ASR 提供商

    内置预设名（如 groq / openrouter_free，已在出厂 config 中定义）→ 只填 API Key；
    全新名称 → 全参数向导。与 init 的手动配置入口共用同一套向导。

    \b
    示例：finminutes config add llm openrouter_free   # 预设：只填 key
          finminutes config add llm myprovider        # 自定义：全参数
          finminutes config add asr myasr
    """
    cfg = _get_config()
    ns = f"{kind}_providers"
    pkg_defaults = cfg._package_default().get(ns, {})
    providers = cfg.config.setdefault(ns, {})
    if name in pkg_defaults:
        # 预设档：内置名（深合并后已在 providers 中）→ 只填 key
        entry = dict(providers.get(name) or pkg_defaults[name])
        _prompt_key_if_missing(kind.upper(), name, entry)
    elif name in providers:
        # 用户自建且已存在
        click.echo(f"[错误]提供商 {name} 已存在，可用 `config set` 修改，或换个名字。", err=True)
        sys.exit(1)
    else:
        # 自定义档：全参数向导
        entry = _custom_provider_wizard(kind, name)
    make_active = click.confirm(f"将 {name} 设为当前激活{kind.upper()}？", default=True)
    providers[name] = entry
    if make_active:
        cfg.config[f"active_{kind}"] = name
    cfg.save()
    click.echo(f"已保存 {kind} 提供商 {name}。")


@config.command(name="remove")
@click.argument("kind", type=click.Choice(["llm", "asr"]))
@click.argument("name")
def config_remove(kind, name):
    """删除用户自定义的 LLM 或 ASR 提供商

    用于清理残留/废弃的 provider

    \b
    示例：finminutes config remove llm ollama
          finminutes config remove asr myasr
    """
    cfg = _get_config()
    ns = f"{kind}_providers"
    pkg_defaults = cfg._package_default().get(ns, {})
    providers = cfg.config.get(ns, {})
    if name not in providers:
        click.echo(f"[错误]提供商 {name} 不存在。", err=True)
        sys.exit(1)
    if name in pkg_defaults:
        click.echo(
            f"[错误]{name} 是内置预设，无法删除（加载时会被出厂默认恢复）。如要停用，请切换 active 后忽略它。",
            err=True,
        )
        sys.exit(1)
    if cfg.config.get(f"active_{kind}") == name:
        click.echo(f"[错误]不能删除当前激活的 {kind.upper()} 提供商 {name}，请先切换 active。", err=True)
        sys.exit(1)
    if len(providers) <= 1:
        click.echo(f"[错误]不能删除最后一个 {kind.upper()} 提供商。", err=True)
        sys.exit(1)
    if not click.confirm(f"确定删除 {kind.upper()} 提供商 {name}？", default=False):
        return
    del providers[name]
    cfg.save()
    click.echo(f"已删除 {kind} 提供商 {name}。")


def _print_all_providers(cfg):
    """打印全部 LLM 与 ASR 提供商（激活项标 [当前]，Key 脱敏）。"""
    active_llm = cfg.get_active_llm()
    active_asr = cfg.get_active_asr()

    click.echo("[LLM 提供商]")
    llm_providers = cfg.config.get("llm_providers", {})
    if not llm_providers:
        click.echo("  （无）")
    for name, opts in llm_providers.items():
        marker = " [当前]" if name == active_llm else ""
        key = _mask_key(opts.get("api_key", ""))
        click.echo(f"  {name}{marker}  模型: {opts.get('model', '?')}  key: {key}")

    click.echo("")
    click.echo("[ASR 提供商]")
    asr_providers = cfg.config.get("asr_providers", {})
    if not asr_providers:
        click.echo("  （无）")
    for name, opts in asr_providers.items():
        marker = " [当前]" if name == active_asr else ""
        key = _mask_key(opts.get("api_key", ""))
        chunk = opts.get("chunk_duration_minutes", "?")
        overlap = opts.get("overlap_seconds", "?")
        click.echo(f"  {name}{marker}  模型: {opts.get('model', '?')}  切分: {chunk}min/{overlap}s  key: {key}")


@config.command(name="list")
def config_list():
    """列出全部 LLM 与 ASR 提供商"""
    _print_all_providers(_get_config())


@config.command(name="list-providers", hidden=True)
def config_list_providers():
    """（旧命令，请用 config list）列出全部 LLM 与 ASR 提供商"""
    _print_all_providers(_get_config())


def _mask_key(key: str) -> str:
    """按 Key 前缀类型脱敏：保留前缀 + `***` + 后3位；其他任意字符串完全隐藏。"""
    if not key:
        return ""
    for prefix in ("gsk_", "sk-or-", "sk-"):
        if key.startswith(prefix):
            tail = key[len(prefix):]
            if len(tail) <= 3:
                return prefix + "***"
            return prefix + "***" + tail[-3:]
    return "***"


# ---------------------------------------------------------------------------
# transcribe  —  ASR speech-to-text
# ---------------------------------------------------------------------------


_ASR_SETUP_GUIDES = {
    "groq": {
        "name": "Groq Cloud",
        "url": "https://console.groq.com/login",
        "key_format": "gsk_xxx",
    },
    "siliconflow": {
        "name": "硅基流动（SiliconFlow）",
        "url": "https://www.siliconflow.cn/",
        "key_format": "sk-xxx",
    },
}


def _ffmpeg_install_guide(reason: str = "音频超过当前 ASR 提供商限制，需要切分转录") -> str:
    """按平台返回 ffmpeg 安装指引；reason 说明缺 ffmpeg 的原因。"""
    head = f"{reason}，但系统缺少 ffmpeg。\n"
    if sys.platform.startswith("win"):
        return (
            head
            + "\n"
            + "请安装 ffmpeg：\n"
            + "\n"
            + "方式一（推荐，winget）：\n"
            + "  winget install Gyan.FFmpeg\n"
            + "\n"
            + "方式二（手动下载）：\n"
            + "  1. 打开 https://www.gyan.dev/ffmpeg/builds/ 下载 release 版压缩包\n"
            + "  2. 解压到本地目录（例如 C:\\ffmpeg）\n"
            + "  3. 将 bin 目录加入 PATH：\n"
            + "     设置 → 系统 → 高级系统设置 → 环境变量 → Path → 新建\n"
            + "     添加：C:\\ffmpeg\\bin\n"
            + "\n"
            + "安装完成后请重新打开终端（让新的 PATH 生效），再重新执行命令。"
        )
    if sys.platform == "darwin":
        return (
            head
            + "\n"
            + "请安装 ffmpeg：\n"
            + "  brew install ffmpeg\n"
            + "\n"
            + "安装完成后请重新打开终端，再重新执行命令。"
        )
    return (
        head
        + "\n"
        + "请安装 ffmpeg：\n"
        + "  Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg\n"
        + "  CentOS/RHEL:   sudo yum install ffmpeg\n"
        + "\n"
        + "安装完成后请重新打开终端，再重新执行命令。"
    )


def _ensure_asr_config(cfg: ConfigManager) -> bool:
    active = cfg.get_active_asr()
    try:
        asr_cfg = cfg.get_asr_config()
        api_key = asr_cfg.get("api_key", "")
        if api_key and "${" not in api_key:
            return True
    except FinMinutesError:
        pass

    guide = _ASR_SETUP_GUIDES.get(active)
    if not guide:
        click.echo(f"[错误]未知的 ASR 提供商: {active}，请先执行 finminutes config set active_asr <提供商>", err=True)
        return False

    click.echo("")
    click.echo(f"[警告]检测到您尚未配置{guide['name']} API Key。")
    click.echo("")
    click.echo("请按以下步骤获取 API Key：")
    click.echo(f"  1. 访问 {guide['url']}")
    click.echo("  2. 注册/登录")
    click.echo("  3. 进入 API Keys 页面创建新密钥")
    click.echo(f"  4. 复制你的 API Key（格式：{guide['key_format']}）")
    click.echo("")

    raw = cfg.config
    if "asr_providers" not in raw:
        raw["asr_providers"] = {}
    if active not in raw["asr_providers"]:
        raw["asr_providers"][active] = {
            "provider": active,
            "api_key": "${%s}" % active.upper() + "_API_KEY",
            "model": "whisper-large-v3-turbo",
            "description": "ASR 提供商",
        }
        raw["active_asr"] = active

    if not _set_provider_key(cfg, active, "asr_providers", label=f"{guide['name']} "):
        click.echo("未输入 API Key，无法使用 ASR 功能。", err=True)
        return False
    click.echo("ASR 配置已保存。")
    return True


def _maybe_extract_audio(audio: str, output: str) -> str:
    """视频文件先提取音轨并保留为 {stem}_音频.mp3（幂等），返回可转写的音频路径。"""
    ext = os.path.splitext(audio)[1].lower()
    if ext not in _VIDEO_EXTS:
        return audio

    from finminutes.core.asr_client import ASRClient, FFmpegMissingError
    try:
        ASRClient._check_ffmpeg()
    except FFmpegMissingError:
        click.echo(_ffmpeg_install_guide("需要从视频提取音轨"), err=True)
        sys.exit(1)

    target = _resolve_output_path(audio, output, "_音频.mp3")
    if os.path.exists(target):
        click.echo(f"[提示]检测到已提取音轨，直接复用: {target}")
        return target

    click.echo(f"正在从视频提取音频 -> {target} ...")
    try:
        from pydub import AudioSegment
    except ImportError:
        click.echo("错误：pydub 未安装，请执行 pip install pydub", err=True)
        sys.exit(1)
    try:
        seg = AudioSegment.from_file(audio)
    except Exception as e:
        click.echo(f"错误：无法读取视频音轨（可能无音轨或格式不支持）: {e}", err=True)
        sys.exit(1)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    seg.export(target, format="mp3", bitrate="64k", parameters=["-ac", "1"])
    click.echo(f"音频已提取并保留: {target}")
    return target


@main.command()
@click.option("--audio", "-a", required=True, help="音频/视频文件路径（.mp3/.wav/.m4a；视频自动提取音轨）")
@click.option("--background", "-b", default="", help="背景信息 YAML 文件路径")
@click.option("--glossary", "-g", default="", help="术语表 YAML 文件路径，或内置标签名（如 semiconductor）")
@click.option("--no-process", is_flag=True, help="仅转录，不自动生成校验稿")
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 音频所在目录）")
@click.option("--skip-rewrite", is_flag=True, help="跳过改写润色阶段（减少 LLM 调用，适合长文本/限流场景）")
@click.option(
    "--format", "-f", "fmt",
    default="qa",
    type=click.Choice(["qa", "speech", "both"], case_sensitive=False),
    help="转录类型：qa 纯对话（默认）/ speech 纯独白 / both 独白+对话",
)
def transcribe(audio, background, glossary, no_process, output, skip_rewrite, fmt):
    """语音转写，语音直接生成校验稿（可选）"""
    audio = _normalize_path(audio, "-a ")
    background = _normalize_path(background, "-b ")
    glossary = _normalize_path(glossary, "-g ")
    output = _normalize_path(output, "-o ")
    if not os.path.exists(audio):
        click.echo(f"错误：音频文件不存在: {audio}", err=True)
        click.echo("提示：若你使用了 \\ 反斜杠路径，请改用 / 或相对路径（Git Bash 中反斜杠会被转义吞掉）。", err=True)
        sys.exit(1)

    effective_audio = _maybe_extract_audio(audio, output)

    file_size_mb = os.path.getsize(effective_audio) / (1024 * 1024)

    cfg = _get_config()
    asr_cfg = cfg.get_asr_config()
    max_file_size_mb = asr_cfg.get("max_file_size_mb", 25)
    needs_chunking = file_size_mb > max_file_size_mb

    if not _ensure_asr_config(cfg):
        sys.exit(1)

    if not no_process and not _ensure_llm_config(cfg):
        sys.exit(1)

    if background:
        click.echo(f"[提示]使用指定的背景信息: {background}")
    else:
        _maybe_show_background_hint("")

    click.echo("正在转写音频...")
    if needs_chunking:
        click.echo(f"  音频大小: {file_size_mb:.2f}MB，超过 {max_file_size_mb}MB 限制，将自动切分转写")

    def _on_chunk_progress(stage, current, total, start_ms, end_ms):
        if stage == "chunk":
            start_sec = start_ms // 1000
            end_sec = end_ms // 1000
            click.echo(f"  [{current}/{total}] 转写片段 {current} ({start_sec // 60}:{start_sec % 60:02d} - {end_sec // 60}:{end_sec % 60:02d})...")
        elif stage == "merging":
            click.echo("  正在合并转录结果...")

    try:
        client = cfg.get_asr_client()
        transcript = client.transcribe(effective_audio, progress_callback=_on_chunk_progress if needs_chunking else None)
    except FFmpegMissingError:
        click.echo(_ffmpeg_install_guide(), err=True)
        sys.exit(1)
    except Exception as e:
        active_asr = cfg.get_active_asr()
        click.echo(f"转写失败: {e}", err=True)
        if active_asr == "siliconflow":
            click.echo("提示：请检查 API Key 是否正确；如需换回 Groq，可执行: finminutes config set active_asr groq", err=True)
        else:
            click.echo("提示：请检查 API Key 是否正确；如需切换 ASR，可执行: finminutes config set active_asr <提供商>", err=True)
        sys.exit(1)

    click.echo(f"转写完成，共 {len(transcript)} 字符。")
    transcript_path = _resolve_output_path(audio, output, "_转录稿.txt")
    os.makedirs(os.path.dirname(transcript_path) or ".", exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    click.echo(f"转录稿已保存至: {transcript_path}")

    if not no_process:
        if fmt == "qa":
            est = _estimate_qa_calls(transcript, chunk_size=cfg.get_qa_chunk_size())
        else:
            est = _estimate_llm_calls(transcript, chunk_size=cfg.get_rewrite_chunk_size()) if not skip_rewrite else 1
        if est > 20:
            click.echo(f"[提示]预计本次 LLM 调用约 {est} 次。OpenRouter 免费档日配额 50 次。", err=True)
        click.echo("")
        click.echo("正在生成校验稿...")
        pipeline = Pipeline(cfg)

        stage_labels = {
            "loading": "[1/6] 加载数据...",
            "preprocess": "[2/6] 预处理文本...",
            "rewrite": "[3/6] 改写润色...",
            "summarize": "[4/6] 生成摘要...",
            "factcheck": "[5/6] 事实校验...",
            "render": "[6/6] 渲染输出...",
            "done": "处理完成。",
        }
        if fmt == "qa" or skip_rewrite:
            stage_labels = {
                "loading": "[1/5] 加载数据...",
                "preprocess": "[2/5] 预处理文本...",
                "summarize": "[3/5] 生成摘要...",
                "factcheck": "[4/5] 事实校验...",
                "render": "[5/5] 渲染输出...",
                "done": "处理完成。",
            }

        def _on_progress(stage: str, message: str = "", index: int = None, total: int = None, elapsed: float = None):
            label = stage_labels.get(stage, stage)
            if index is not None and total:
                done = max(index, 1)
                elapsed_total = float(elapsed or 0)
                per = elapsed_total / done
                remain = per * (total - index)
                remain_text = f"~{remain:.0f}s" if remain < 60 else f"~{remain / 60:.1f}分钟"
                click.echo(f"\r  {label} {index}/{total} 片 · 已用 {elapsed_total:.0f}s · 预计还需 {remain_text}", err=True, nl=False)
                if index >= total:
                    click.echo("", err=True)
            elif message:
                click.echo(f"  {label}（{message}）", err=True)
            else:
                click.echo(f"  {label}", err=True)

        try:
            result = pipeline.run(
                transcript=transcript,
                background_path=background,
                glossary_tag=glossary,
                skip_rewrite=skip_rewrite,
                format=fmt,
                progress_callback=_on_progress,
                on_error=lambda stage, msg: click.echo(f"  [失败] {msg}", err=True),
            )
        except Exception as e:
            click.echo(f"处理失败: {e}", err=True)
            sys.exit(1)

        if result.aborted:
            click.echo("", err=True)
            click.echo("处理中止：关键阶段失败，未生成校验稿（转录稿已保存）。", err=True)
            sys.exit(1)

        _on_progress("done")

        minutes = result.structured_minutes
        if minutes is None:
            from finminutes.core.summarizer import StructuredMinutes
            minutes = StructuredMinutes()

        review_content = ReviewRenderer(minutes, result.fact_check_report, result.transcript_raw).render()
        review_path = _resolve_output_path(audio, output, "_校验稿.md")
        with open(review_path, "w", encoding="utf-8") as f:
            f.write(review_content)
        click.echo(f"校验稿已输出至: {review_path}")


# ---------------------------------------------------------------------------
# glossary  —  glossary management
# ---------------------------------------------------------------------------


@main.group()
def glossary():
    """术语表管理"""
    pass


@glossary.command(name="generate")
@click.option("--from", "-f", "from_file", required=True, help="源文件路径（.txt/.docx/.pdf/.md）")
@click.option("--tag", "-t", required=True, help="术语表标签名")
@click.option("--output", "-o", default="", help="输出 YAML 路径（默认当前目录 <tag>.yaml）")
def glossary_generate(from_file, tag, output):
    """从访谈清单/材料中自动提取术语并生成术语表"""
    from_file = _normalize_path(from_file, "-f ")
    output = _normalize_path(output, "-o ")
    if not os.path.exists(from_file):
        click.echo(f"错误：文件不存在: {from_file}", err=True)
        sys.exit(1)

    click.echo("正在读取材料...")
    try:
        text = _read_text(from_file)
    except Exception as e:
        click.echo(f"读取失败: {e}", err=True)
        sys.exit(1)

    click.echo(f"材料读取完成，共 {len(text)} 字符。正在调用 LLM 提取术语...")
    try:
        cfg = _get_config()
        llm = cfg.get_llm_client()
        data = generate_glossary(llm, text)
    except Exception as e:
        click.echo(f"术语提取失败: {e}", err=True)
        sys.exit(1)

    terms = data.get("terms", [])
    click.echo(f"提取到 {len(terms)} 个术语。")

    if not output:
        output = f"{tag}.yaml"

    save_glossary(data, output)
    click.echo(f"术语表已保存至: {output}")


# ---------------------------------------------------------------------------
# serve  —  launch HTTP API
# ---------------------------------------------------------------------------


@main.command(hidden=True)
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=8000, type=int, help="监听端口")
def serve(host, port):
    """启动 FinMinutes HTTP API 服务"""
    click.echo(f"启动 API 服务: http://{host}:{port}")
    click.echo("接口文档: http://localhost:{}/docs".format(port) if host == "0.0.0.0" else "")
    from finminutes.api.server import app
    import uvicorn
    uvicorn.run(app, host=host, port=port)
