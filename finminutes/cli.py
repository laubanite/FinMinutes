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
from finminutes.core.qa_parser import parse_qa_pairs_from_body, sync_frontmatter_with_body, _split_frontmatter
from finminutes.core.template_loader import TemplateLoader

_CONFIG_PATH = os.path.expanduser("~/.finminutes/config.yaml")

_LLM_CHOICES = {
    "1": ("openrouter_free", "OpenRouter（免费，推荐首选）"),
    "2": ("deepseek", "DeepSeek（低成本付费）"),
    "3": ("openai", "OpenAI（付费，高性能）"),
    "4": ("ollama", "Ollama 本地（数据零上云）"),
}


def _get_config() -> ConfigManager:
    try:
        return ConfigManager(_CONFIG_PATH)
    except FinMinutesError:
        try:
            return ConfigManager()
        except FinMinutesError:
            click.echo("错误：无法加载配置文件。请先运行 `finminutes init`。", err=True)
            sys.exit(1)


def _resolve_output_path(source_file: str, output: str, suffix: str, stem: str | None = None) -> str:
    """统一解析 -o 输出路径规则。

    - 不指定 output: 源文件所在目录 + 源文件名 + suffix
    - output 以 / 或 反斜杠 结尾: 视为目录 -> output/源文件名+suffix
    - output 不含 / 或 反斜杠: 视为前缀 -> 源文件目录/output+suffix
    - output 包含 / 或 反斜杠: 视为目录+前缀 -> output+suffix
    """
    source_dir = os.path.dirname(source_file)
    source_name = stem if stem is not None else os.path.splitext(os.path.basename(source_file))[0]

    if not output:
        return os.path.join(source_dir, source_name + suffix)
    if output.endswith("/") or output.endswith("\\"):
        return os.path.join(output, source_name + suffix)
    if "/" in output or "\\" in output:
        return output + suffix
    return os.path.join(source_dir, output + suffix)


_BACKGROUND_HINT = """ℹ️ 未指定背景信息，将使用默认背景。如需更精准的纪要，建议使用 -b 指定背景文件。

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


# ---------------------------------------------------------------------------
# main group
# ---------------------------------------------------------------------------


class _MainGroup(click.Group):
    """自定义 group：按工作流逻辑排序 --help 子命令显示顺序。"""

    _CMD_ORDER = ("config", "init", "glossary", "transcribe", "process", "render")

    def list_commands(self, ctx):
        return list(self._CMD_ORDER)


@click.group(cls=_MainGroup)
@click.version_option(version=__version__, prog_name="finminutes")
def main():
    pass


# ---------------------------------------------------------------------------
# init  —  interactive cloud-first onboarding wizard
# ---------------------------------------------------------------------------


@main.command()
@click.option("--force", is_flag=True, help="覆盖已有配置")
def init(force):
    """交互式初始化向导"""
    cfg_path = _CONFIG_PATH

    auto_force = False
    if os.path.exists(cfg_path) and not force:
        try:
            cfg = ConfigManager(cfg_path)
            active = cfg.get_active_llm()
            pc = cfg.get_llm_config()
            api_key = pc.get("api_key", "")
            if not api_key or "${" in api_key:
                click.echo("检测到配置不完整（API Key 为空），自动进入配置引导...")
                auto_force = True
            else:
                click.echo("正在检测现有配置...")
                client = LLMFactory.create(pc)
                ok, msg, latency = client.test_connection()
                if ok:
                    click.echo(f"配置已就绪！({active}，延迟 {latency}ms)")
                    return
                click.echo(f"检测到连接失败: {msg}")
                if click.confirm("是否重新配置？", default=True):
                    auto_force = True
                else:
                    return
        except FinMinutesError:
            click.echo("检测到配置文件异常，自动进入配置引导...")
            auto_force = True

    should_enter = force or auto_force or not os.path.exists(cfg_path)

    existing_raw = None
    if os.path.exists(cfg_path) and should_enter:
        try:
            existing_raw = ConfigManager(cfg_path).config.copy()
        except FinMinutesError:
            pass

    if existing_raw:
        click.echo("检测到已有配置。")
    click.echo("请选择 LLM 提供商：")
    for key, (_, label) in _LLM_CHOICES.items():
        click.echo(f"  [{key}] {label}")

    if existing_raw:
        current_preset = existing_raw.get("active_llm", "")
        current_key = None
        for key, (name, _) in _LLM_CHOICES.items():
            if name == current_preset:
                current_key = key
                break
        default_choice = current_key or "1"
        click.echo(f"当前配置为: {current_preset}")
    else:
        default_choice = "1"

    choice = click.prompt("请输入编号", type=click.Choice(list(_LLM_CHOICES)), default=default_choice)
    preset_name, _ = _LLM_CHOICES[choice]

    if existing_raw and preset_name == existing_raw.get("active_llm", ""):
        providers = existing_raw.get("llm_providers", {})
        provider_config = providers.get(preset_name, {}).copy()
    else:
        provider_config = _get_preset_defaults(preset_name)

    if preset_name != "ollama":
        api_key = click.prompt(
            "请输入 API Key（留空则使用当前配置或环境变量）",
            default="",
            hide_input=True,
        )
        if api_key:
            provider_config["api_key"] = api_key

    click.echo("")
    click.echo("正在测试连接...")
    try:
        client = LLMFactory.create(provider_config)
        ok, msg, latency = client.test_connection()
        if ok:
            click.echo(f"连接成功！延迟: {latency}ms")
        else:
            click.echo(f"连接测试返回: {msg}")
            if not click.confirm("连接测试未通过，仍然保存配置？", default=False):
                return
    except Exception as e:
        click.echo(f"连接测试异常: {e}")
        if not click.confirm("仍然保存配置？", default=False):
            return

    if existing_raw:
        config_data = existing_raw
        config_data["llm_providers"][preset_name] = provider_config
        config_data["active_llm"] = preset_name
    else:
        config_data = {
            "active_llm": preset_name,
            "llm_providers": {preset_name: provider_config},
            "logging": {
                "level": "INFO",
                "file": "finminutes.log",
                "redact_transcript": True,
                "redact_api_key": True,
            },
        }

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    import yaml
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, allow_unicode=True, default_flow_style=False)

    click.echo("")
    click.echo(f"配置文件已保存至: {cfg_path}")
    click.echo("")
    click.echo("后续使用指南：")
    click.echo("  finminutes process -t 转录文件.txt    # 快速处理转录")
    click.echo("  finminutes process --help            # 查看完整选项")
    click.echo("  finminutes transcribe --help         # 语音转写")
    click.echo("")


def _get_preset_defaults(preset: str) -> dict:
    defaults = {
        "openrouter_free": {
            "provider": "openrouter",
            "api_key": "${OPENROUTER_API_KEY}",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "google/gemini-2.0-flash-lite-preview-02-05",
        },
        "deepseek": {
            "provider": "deepseek",
            "api_key": "${DEEPSEEK_API_KEY}",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
        "openai": {
            "provider": "openai",
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
        },
        "ollama": {
            "provider": "ollama",
            "api_key": "",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
        },
    }
    return defaults.get(preset, {}).copy()


# ---------------------------------------------------------------------------
# process  —  run the full pipeline
# ---------------------------------------------------------------------------


@main.command()
@click.option("--transcript", "-t", required=True, help="转录文本文件路径")
@click.option("--background", "-b", default="", help="背景信息 YAML 文件路径")
@click.option("--glossary", "-g", default="", help="术语表标签名")
@click.option(
    "--mode", "-m",
    default="full",
    type=click.Choice(["fast", "standard", "full"], case_sensitive=False),
    help="流水线模式（默认 full）",
)
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 转录稿所在目录）")
def process(transcript, background, glossary, mode, output):
    """从转录稿生成校验稿"""
    if not os.path.exists(transcript):
        click.echo(f"错误：转录文件不存在: {transcript}", err=True)
        sys.exit(1)

    with open(transcript, "r", encoding="utf-8") as f:
        text = f.read()

    cfg = _get_config()
    pipeline = Pipeline(cfg)

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
            "loading": "[1/3] 加载数据...",
            "preprocess": "[2/3] 预处理文本...",
            "render": "[3/3] 渲染输出...",
            "done": "处理完成。",
        }
    elif mode == "standard":
        stage_labels = {
            "loading": "[1/4] 加载数据...",
            "preprocess": "[2/4] 预处理文本...",
            "rewrite": "[3/4] 改写润色...",
            "render": "[4/4] 渲染输出...",
            "done": "处理完成。",
        }

    def _on_progress(stage: str):
        label = stage_labels.get(stage, stage)
        click.echo(f"  {label}", err=True)

    try:
        result = pipeline.run(
            transcript=text,
            background_path=background,
            glossary_tag=glossary,
            mode=mode,
            progress_callback=_on_progress,
        )
        _on_progress("done")
    except Exception as e:
        click.echo(f"处理失败: {e}", err=True)
        sys.exit(1)

    minutes = result.structured_minutes
    if minutes is None:
        from finminutes.core.summarizer import StructuredMinutes
        minutes = StructuredMinutes()

    review_content = ReviewRenderer(minutes, result.fact_check_report, result.transcript_raw).render()

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
@click.option("--review", "-r", required=True, help="校验稿文件路径")
@click.option("--template", "-T", default="default", help="成品稿模板名称（默认使用 default）")
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 校验稿所在目录）")
def render(review, template, output):
    """从校验稿渲染成品稿"""
    import yaml
    import os

    if not os.path.exists(review):
        click.echo(f"错误：校验稿文件不存在: {review}", err=True)
        sys.exit(1)

    with open(review, "r", encoding="utf-8") as f:
        content = f.read()

    data, body = _split_frontmatter(content)
    if data is None:
        click.echo("错误：校验稿格式不正确（缺少 YAML 头）", err=True)
        sys.exit(1)

    qa_pairs = data.get("qa_pairs", [])

    body_pairs = parse_qa_pairs_from_body(body)
    if body_pairs:
        qa_pairs = body_pairs
        sync_frontmatter_with_body(review, qa_pairs)
    else:
        if not qa_pairs:
            click.echo("错误：校验稿中未找到任何 Q&A 数据（正文解析失败，Front Matter 中也没有 qa_pairs）。请检查校验稿格式是否正确。", err=True)
            sys.exit(1)
        click.echo("⚠️ 正文解析失败，使用 Front Matter 中的数据。请检查校验稿正文格式。", err=True)

    try:
        loader = TemplateLoader(subdir="formal")
        tmpl = loader.load(template)
    except FileNotFoundError:
        click.echo(f"错误：成品稿模板不存在: {template}", err=True)
        click.echo(f"可用模板: {', '.join(loader.list_templates())}", err=True)
        sys.exit(1)

    try:
        cfg = _get_config()
        llm = cfg.get_llm_client()
    except Exception as e:
        click.echo(f"错误：获取 LLM 客户端失败: {e}", err=True)
        sys.exit(1)

    click.echo(f"正在调用 LLM 生成成品稿（{len(qa_pairs)} 组 Q&A，模板: {template}）...")
    generator = FormalGenerator(llm)
    try:
        formal_output = generator.generate(qa_pairs, tmpl)
    except Exception as e:
        click.echo(f"错误：LLM 生成失败: {e}", err=True)
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(review))[0]
    stem = stem.replace("_校验稿", "")
    formal_path = _resolve_output_path(review, output, "_成品稿.md", stem=stem)
    os.makedirs(os.path.dirname(formal_path) or ".", exist_ok=True)
    with open(formal_path, "w", encoding="utf-8") as f:
        f.write(formal_output)
    click.echo(f"成品稿已输出至: {formal_path}")


# ---------------------------------------------------------------------------
# config  —  configuration management
# ---------------------------------------------------------------------------


@main.group()
def config():
    """配置管理"""
    pass


@config.command(name="show")
def config_show():
    """显示当前配置（API Key 脱敏）"""
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


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """设置 active_llm 和 active_asr 两个配置项

    \b
    示例：finminutes config set active_llm deepseek
          finminutes config set active_asr siliconflow
    """
    cfg = _get_config()
    if key not in ("active_llm", "active_asr"):
        click.echo("❌ 错误：仅支持设置 active_llm 和 active_asr", err=True)
        sys.exit(1)
    if key == "active_llm":
        cfg.set_active_llm(value)
    else:
        cfg.set_active_asr(value)
    cfg.save()
    click.echo(f"已设置 {key} = {value}")


@config.command(name="list-providers")
def config_list_providers():
    """列出所有可用的 LLM 提供商"""
    cfg = _get_config()
    providers = cfg.config.get("llm_providers", {})
    active = cfg.get_active_llm()
    click.echo(f"当前使用的提供商: {active}")
    click.echo("")
    for name, opts in providers.items():
        marker = " [当前]" if name == active else ""
        desc = opts.get("description", "")
        click.echo(f"  {name}{marker}")
        if desc:
            click.echo(f"    {desc}")
        click.echo(f"    模型: {opts.get('model', '?')}")
        click.echo(f"    地址: {opts.get('base_url', '?')}")


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


def _ffmpeg_install_guide() -> str:
    if sys.platform.startswith("win"):
        return (
            "音频超过当前 ASR 提供商限制，需要切分转录，但系统缺少 ffmpeg。\n"
            "\n"
            "请安装 ffmpeg：\n"
            "\n"
            "方式一（推荐，winget）：\n"
            "  winget install Gyan.FFmpeg\n"
            "\n"
            "方式二（手动下载）：\n"
            "  1. 打开 https://www.gyan.dev/ffmpeg/builds/ 下载 release 版压缩包\n"
            "  2. 解压到本地目录（例如 C:\\ffmpeg）\n"
            "  3. 将 bin 目录加入 PATH：\n"
            "     设置 → 系统 → 高级系统设置 → 环境变量 → Path → 新建\n"
            "     添加：C:\\ffmpeg\\bin\n"
            "\n"
            "安装完成后请重新打开终端（让新的 PATH 生效），再重新执行命令。"
        )
    if sys.platform == "darwin":
        return (
            "音频超过当前 ASR 提供商限制，需要切分转录，但系统缺少 ffmpeg。\n"
            "\n"
            "请安装 ffmpeg：\n"
            "  brew install ffmpeg\n"
            "\n"
            "安装完成后请重新打开终端，再重新执行命令。"
        )
    return (
        "音频超过当前 ASR 提供商限制，需要切分转录，但系统缺少 ffmpeg。\n"
        "\n"
        "请安装 ffmpeg：\n"
        "  Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg\n"
        "  CentOS/RHEL:   sudo yum install ffmpeg\n"
        "\n"
        "安装完成后请重新打开终端，再重新执行命令。"
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
        click.echo(f"❌ 错误：未知的 ASR 提供商: {active}，请先执行 finminutes config set active_asr <提供商>", err=True)
        return False

    click.echo("")
    click.echo(f"⚠️ 检测到您尚未配置{guide['name']} API Key。")
    click.echo("")
    click.echo("请按以下步骤获取 API Key：")
    click.echo(f"  1. 访问 {guide['url']}")
    click.echo("  2. 注册/登录")
    click.echo("  3. 进入 API Keys 页面创建新密钥")
    click.echo(f"  4. 复制你的 API Key（格式：{guide['key_format']}）")
    click.echo("")

    api_key = click.prompt(f"请粘贴你的 {guide['name']} API Key", hide_input=True)
    if not api_key:
        click.echo("未输入 API Key，无法使用 ASR 功能。", err=True)
        return False

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
    raw["asr_providers"][active]["api_key"] = api_key
    cfg.save()
    click.echo("ASR 配置已保存。")
    return True


@main.command()
@click.option("--audio", "-a", required=True, help="音频文件路径（.mp3/.wav/.m4a）")
@click.option("--background", "-b", default="", help="背景信息 YAML 文件路径")
@click.option("--glossary", "-g", default="", help="术语表标签名")
@click.option("--no-process", is_flag=True, help="仅转录，不自动生成校验稿")
@click.option("--output", "-o", default="", help="输出前缀或目录路径（默认: 音频所在目录）")
def transcribe(audio, background, glossary, no_process, output):
    """语音转写，语音直接生成校验稿（可选）"""
    if not os.path.exists(audio):
        click.echo(f"错误：音频文件不存在: {audio}", err=True)
        sys.exit(1)

    file_size_mb = os.path.getsize(audio) / (1024 * 1024)

    cfg = _get_config()
    asr_cfg = cfg.get_asr_config()
    max_file_size_mb = asr_cfg.get("max_file_size_mb", 25)
    needs_chunking = file_size_mb > max_file_size_mb

    if not _ensure_asr_config(cfg):
        sys.exit(1)

    if background:
        click.echo(f"ℹ️ 使用指定的背景信息: {background}")
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
        transcript = client.transcribe(audio, progress_callback=_on_chunk_progress if needs_chunking else None)
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

        def _on_progress(stage: str):
            label = stage_labels.get(stage, stage)
            click.echo(f"  {label}", err=True)

        try:
            result = pipeline.run(
                transcript=transcript,
                background_path=background,
                glossary_tag=glossary,
                progress_callback=_on_progress,
            )
            _on_progress("done")
        except Exception as e:
            click.echo(f"处理失败: {e}", err=True)
            sys.exit(1)

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
@click.option("--output", "-o", default="", help="输出 YAML 路径（默认 data/glossary/<tag>.yaml）")
def glossary_generate(from_file, tag, output):
    """从访谈清单/材料中自动提取术语并生成术语表"""
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
        output = os.path.join(
            os.path.dirname(__file__), "data", "glossary", f"{tag}.yaml"
        )

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
