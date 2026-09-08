<div align="center">

# FinMinutes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/) [![Powered by OrcaRouter](https://img.shields.io/badge/Powered_by-OrcaRouter-2563eb)](https://www.orcarouter.ai/ref/ref_fc88ee354797b100a58a)

**AI Meeting Minutes Workflow — One-stop tool that turns video/audio/transcripts into professional meeting minutes.**

**Free LLM + Free ASR, one-click pipeline.**

<p align="center">
<a href="#features">Features</a> • <a href="#quick-start">Quick Start</a> • <a href="#free-api-keys">Free API Keys</a> • <a href="#commands">Commands</a> • <a href="#license">License</a>
</p>

<p align="center">
<a href="README.md">中文</a> | English
</p>

</div>

FinMinutes — a one-stop tool that turns **video / audio / transcripts** into professional meeting minutes for investment research, due diligence, roadshows, and expert interviews.

**The problem**: a single interview takes half a day to transcribe, polish, and fact-check by hand. FinMinutes compresses it to three steps — **transcribe/import → listen once and verify against the review draft → deliver the polished final draft**.

**Key advantages**:

- **Runs free end-to-end**: free LLM + free ASR, with a web-AI loop as a fallback when free quotas run out
- **Numbers are never lost**: automatic content-completeness check (number/fact retention); below threshold it auto-writes a debug prompt so you can iterate
- **Two-draft workflow**: the review draft preserves the full Q&A + key points for verification; the final draft is polished and deliverable — edit one file and re-render

## Features

- **Two-draft loop**: fidelity review draft + deliverable final draft; edit one file, re-render
- **Speech-to-text**: free Groq Whisper / SiliconFlow SenseVoice; auto-extracts audio from video; auto-chunks long files
- **Web-AI loop**: 0-cost — export a prompt pack, paste it into Doubao / DeepSeek web AIs, import the result back
- **Completeness check**: number/fact retention scoring (deterministic, no LLM); auto-writes a debug prompt when below threshold
- **Multiple LLMs**: free tier first with auto-fallback on rate limits; supports Anthropic / any OpenAI-compatible endpoint (e.g. OrcaRouter)
- **Glossary + background**: auto-generate glossaries from interview materials, correct ASR errors
- **ASR format normalization**: transcripts from Feishu / iFlytek / Whisper etc. work directly
- **Custom templates**: final-draft structure and style are configurable
- **Zero learning curve**: `finminutes --help` + interactive prompts at every step

## Quick Start

```bash
pip install finminutes
finminutes init                            # setup LLM + ASR providers & API keys
finminutes process -t transcript.txt       # text → review draft
finminutes transcribe -a meeting.m4a       # audio/video → transcript → review draft
finminutes render -r review.md             # review draft → final draft
finminutes render -j result.json           # web-AI JSON → final draft directly
```

## Free API Keys

FinMinutes runs free end-to-end: free-tier LLM + free ASR providers. All keys can be entered interactively via `finminutes init` (recommended), or set as environment variables (see Quick Start).

### LLM (minutes generation)

| Provider | Free tier | Get a key |
| --- | --- | --- |
| [**OpenRouter**](https://openrouter.ai) (default, recommended) | Free tier: 20 req/min, 50 req/day | Sign up at [openrouter.ai](https://openrouter.ai) → Keys → Create Key |
| [**OrcaRouter**](https://www.orcarouter.ai/ref/ref_fc88ee354797b100a58a) | Multi-model gateway, pay-as-you-go | Sign up at [orcarouter.ai](https://www.orcarouter.ai/ref/ref_fc88ee354797b100a58a) → API Keys → Create |
| [**SiliconFlow**](https://cloud.siliconflow.cn) | Free models (e.g. GLM-4-9B) | Sign up at [cloud.siliconflow.cn](https://cloud.siliconflow.cn) → API Keys → Create |
| [**DeepSeek**](https://platform.deepseek.com/api_keys) | Paid | Sign up at [platform.deepseek.com](https://platform.deepseek.com/api_keys) → API Keys → Create |
| [**OpenAI**](https://platform.openai.com) | Paid | [platform.openai.com](https://platform.openai.com) → API Keys → Create |

### ASR (speech-to-text)

| Provider | Free tier | Get a key |
| --- | --- | --- |
| [**Groq**](https://console.groq.com) (default, recommended) | Free Whisper, files ≤ 25MB | Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create API Key |
| **SiliconFlow** | Free SenseVoiceSmall, files ≤ 50MB | Same as above (one SiliconFlow key works for both LLM and ASR) |

> **Zero-cost tip**: to run the whole flow, one **Groq key (ASR)** + one **OpenRouter key (LLM)** is all you need — audio → transcript → review draft → final draft. When free quotas run out, the web-AI loop (`export-prompt` / `import-result`) is a fully free, unlimited fallback.

## Commands

| Command | Description |
| --- | --- |
| `init` | Interactive setup wizard (LLM + ASR) |
| `process` | Transcript → review draft |
| `transcribe` | Audio/video → transcript (+ optional review draft) |
| `render` | Review draft → final draft (`-r` draft, `-j` JSON) |
| `export-prompt` | Build a web-AI prompt pack from a transcript |
| `import-result` | Parse web-AI JSON → review draft + completeness report |
| `config` | show / set / add / remove / set-key / list |
| `glossary` | Generate/manage term glossaries |

## LLM Providers

- **OpenRouter** (`openrouter_free`, default): free tier, 20 req/min · 50 req/day
- **DeepSeek**, **OpenAI**, **SiliconFlow** (free tier), **Anthropic** (Claude)
- Any **OpenAI-compatible** endpoint via a custom provider; **fallback** chain auto-degrades on rate limits

## ASR Providers

- **Groq** (`groq`, default): `whisper-large-v3-turbo`, 25MB file limit
- **SiliconFlow** (`siliconflow`): `FunAudioLLM/SenseVoiceSmall`, 50MB file limit

Switch with `finminutes config set active_asr <provider>`. Video files auto-extract their audio track; files over the limit are auto-chunked (requires ffmpeg).

## Pipeline Modes

| Mode | Stages | LLM calls | Output |
| --- | --- | --- | --- |
| `fast` | preprocess only | No | `{stem}_清洗稿.txt` |
| `full` | + rewrite + summarize + fact-check | Yes | `{stem}_校验稿.md` + coverage report |

> `import-result` reports content completeness (deterministic, no LLM). Below the `coverage_threshold` (default 0.80) it auto-writes a debug prompt for the web-AI loop; `render -j` skips the review-draft intermediate entirely.

## License

MIT
