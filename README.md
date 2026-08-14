<div align="center">

# FinMinutes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

**AI 金融会议纪要 Agent —— 从视频/录音/转录稿到专业会议纪要的一站式工具。**

**免费 LLM + 免费 ASR，一键流水线，拯救小黑工。**

[简介](#简介) • [特性](#特性) • [安装](#安装) • [快速开始](#快速开始) • [免费 API Key](#免费-api-key-获取指南) • [命令详解](#命令详解) • [License](#license)

<p align="center">
中文 | <a href="#english">English</a>
</p>

</div>

## 简介

FinMinutes 面向投研、尽调、路演、专家访谈等会议场景，把**视频 / 录音 / 转录稿**一键整理成专业纪要。

**痛点**：一场访谈听下来，逐字整理、书面化、核对数字，动辄半天。FinMinutes 把它压缩成三步——**转写/导入 → 听一遍录音，对着校验稿核对 → 成品稿直接交付**。

**核心优势**：

- **零成本跑通全流程**：提供免费 LLM + 免费 ASR，另有网页版 AI 协同兜底，免费配额用完也不停工
- **数字不丢**：内容完整度自动检查（数字/事实保留率），不达标自动生成调试 prompt 让你迭代补全
- **双稿闭环**：校验稿保真（完整 Q&A + 主题要点），成品稿支持定制模板、可直接交付；改一份文件即可重渲染

## 特性

- **双稿闭环**：校验稿保真核对 + 成品稿可交付，改一份文件重渲染
- **语音转写**：免费 Groq Whisper / 硅基流动 SenseVoice；视频自动提取音轨；长音频自动切分
- **网页版 AI 协同**：0 成本把转录稿喂给豆包/DeepSeek 等网页版AI，结果导回校验稿
- **内容完整度检查**：数字/事实保留率自动打分，不达标自动生成调试 prompt
- **多 LLM 切换**：免费档优先、限流自动降级（fallback）；支持 Anthropic / 任意 OpenAI 兼容端点
- **术语表 + 背景信息**：从访谈材料自动生成术语表、纠正 ASR 错词，提升纪要准确度
- **ASR 格式归一化**：飞书妙记/讯飞/Whisper 等任意转录稿直接可用
- **模板定制**：成品稿章节结构与语言风格可自定义
- **零学习成本**：`finminutes --help` + 每步交互引导

## 安装

要求：Python 3.10+，[ffmpeg](https://ffmpeg.org/)（视频提取音轨 / 大文件切分时需要）

```bash
git clone <你的仓库地址>
cd finminutes
pip install -r requirements.txt
pip install -e .          # 安装 finminutes 命令
```

### 安装 ffmpeg

ffmpeg 用于**视频提取音轨**与**大文件切分**（音频超过 ASR 提供商限制 Groq 25MB / 硅基流动 50MB 时）。按你的系统选择：

**Windows：**

```powershell
# 方式一（推荐）：winget 一键安装
winget install Gyan.FFmpeg

# 方式二：手动下载
#   1. 打开 https://www.gyan.dev/ffmpeg/builds/ 下载 release 版压缩包
#   2. 解压到本地目录（例如 C:\ffmpeg）
#   3. 将 bin 目录加入 PATH：
#      设置 → 系统 → 高级系统设置 → 环境变量 → Path → 新建 → 添加 C:\ffmpeg\bin
```

**macOS：**

```bash
brew install ffmpeg
```

**Linux：**

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install ffmpeg

# CentOS / RHEL
sudo yum install ffmpeg
```

> 安装完成后请**重新打开终端**，让新的 PATH 生效，再重新执行命令。

### 懒人安装：让 Agent 帮你装

如果你不想手动操作，把下面这段话复制给 Claude Code / Codex / opencode / WorkBuddy 等任意 AI Agent，它会帮你完成全部安装：

> 请帮我克隆并安装配置好 https://github.com/laubanite/FinMinutes 这个 AI 会议纪要工具（含 ffmpeg），验证 `finminutes --help` 能正常运行，然后告诉我下一步怎么做。

## 免费 API Key 获取指南

FinMinutes 全流程可零成本跑通：LLM 用免费档、ASR 用免费提供商。下面列出各提供商的注册与取 key 方式。所有 key 都可通过 `finminutes init` 交互式录入（推荐），或设置环境变量（见「快速开始」）。

### LLM（生成纪要）

| 提供商                                                        | 免费额度                                   | 获取方式                                                                               |
| ------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| [**OpenRouter**](https://openrouter.ai)（默认，推荐）    | 提供免费模型，免费档：20 次/分钟、50 次/天 | 注册[openrouter.ai](https://openrouter.ai) → 右上角头像 → Keys → Create Key          |
| [**硅基流动 SiliconFlow**](https://cloud.siliconflow.cn) | 提供免费模型（如 GLM-4-9B）                | 注册[cloud.siliconflow.cn](https://cloud.siliconflow.cn) → 控制台 → API 密钥 → 新建  |
| [**DeepSeek**](https://platform.deepseek.com/api_keys)   | 付费                                       | 注册[platform.deepseek.com](https://platform.deepseek.com/api_keys) → API Keys → 创建 |
| [**OpenAI**](https://platform.openai.com)                | 付费                                       | [platform.openai.com](https://platform.openai.com) → API Keys → Create                |

### ASR（语音转写）

| 提供商                                                  | 免费额度                             | 获取方式                                                                      |
| ------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------- |
| [**Groq**](https://console.groq.com)（默认，推荐） | Whisper 大模型免费，单文件 ≤ 25MB   | 注册[console.groq.com](https://console.groq.com) → API Keys → Create API Key |
| **硅基流动 SiliconFlow**                          | SenseVoiceSmall 免费，单文件 ≤ 50MB | 同上（硅基流动一个账号一个 key 同时用于 LLM 与 ASR）                          |

> **零成本提示**：想完整跑一遍，只需一个 **Groq key（ASR）+ 一个 OpenRouter key（LLM）**，即可「音频 → 转录稿 → 校验稿 → 成品稿」全流程。免费配额不够时还有网页版 AI 协同兜底（`export-prompt` / `import-result`，完全免费、不限次数）。

## 快速开始

### 1. 初始化配置

```bash
finminutes init
```

交互式选择 LLM 提供商并输入 API Key。配置保存在 `~/.finminutes/config.yaml`（不会写入项目目录）。

> 也可直接设置环境变量：`${OPENROUTER_API_KEY}`、`${DEEPSEEK_API_KEY}`、`${OPENAI_API_KEY}`、`${ANTHROPIC_API_KEY}`、`${GROQ_API_KEY}`、`${SILICONFLOW_API_KEY}`

### 2. 生成会议纪要

```bash
# 方式一：从转录文本直接生成
finminutes process -t 转录文件.txt

# 方式二：从音频/视频开始（语音转写 + 自动生成纪要，视频自动提取音轨）
finminutes transcribe -a 会议录音.m4a

# 方式三：先转写，再单独生成纪要
finminutes transcribe -a 会议录音.m4a --no-process
finminutes process -t 会议录音_转录稿.txt
```

生成结果：

| 产物   | 路径                    | 说明                                  |
| ------ | ----------------------- | ------------------------------------- |
| 转录稿 | `{文件名}_转录稿.txt` | 语音转写的原始文本                    |
| 清洗稿 | `{文件名}_清洗稿.txt` | ASR 格式归一化后的纯文本（fast 模式） |
| 校验稿 | `{文件名}_校验稿.md`  | 保真稿：完整 Q&A + 主题要点，供核对   |
| 成品稿 | `{文件名}_成品稿.md`  | 书面化、结构化，可直接交付            |

### 输出示例（脱敏）

以下是同一段访谈生成的校验稿与成品稿示例（内容均为虚构脱敏数据；示例中公司、技术细节与数字均已脱敏为 XX，实际输出会保留原文数字）。

**校验稿** —— 尽量保真转录：保留完整 Q&A 与主题要点，数字原样保留，供对照录音逐条核对：

```
---
qa_pairs:
- question: 第一个关于新一代产品架构，因为这块我不知道XX公司内部现在是不是有一些技术储备或者专利的一些研究，这块大概是怎么样？您能介绍一下吗？
  answer: 研究的新一代架构已经算是有一定的落地了。最早是在外部晶圆厂里面去做，现在已经有一定的突破了。主要方向是从传统架构演进，单元面积确实是缩减了约 XX，通过这样的方式来解决面积方面的问题。毕竟没有先进工艺嘛，所以说在整个大的方向上面是有一定的突破的。良率比较低，目前可能只有将近 XX% 的良率，处于一个没有量产的阶段，但整个大的方向上来说是没有问题的。东西是有专利的，我们处理这段是比较早的。
  asker: 提问者
  timerange: ''
- question: 那XX因为这段时间他们有发相关的新闻，他们也在走这个技术路线方向，那我是不是也可以理解他们也是在良率这块可能还是有比较大的提升空间，而不是说底层的技术方案还没有定型？
  answer: 方案技术已经定型了，没有疑问。大家都是在提高良率。
  asker: 提问者
  timerange: ''
- question: XX他们因为他们有先进工艺嘛，他们在有先进工艺的情况下，是不是他们的新一代架构方案会比国内这几家会有更优的一些密度、什么各方面一些性能指标？
  answer: 本来不是一个维度，不是一个等级的。密度来说他们只是为了提高上限，而我们是为了通过工艺的限制来跟他们达到同样的制程角度。他们在同样的尺寸的密度上面上限更高，而且性能更优，不是在一个游戏里面去做的。
  asker: 提问者
  timerange: ''
- question: 既然咱们现在也是能够绕开先进工艺的方案，先天来说成本和良率可能会有一些区别。咱们的，因为现在方案定了，咱们的目标良率大概会在多少？
  answer: 目标良率大概是在XX%。
  asker: 提问者
  timerange: ''
- question: 有预计几年能够实现吗？比如说三年还是怎么样？因为我们看XX什么都把新一代架构的量产时间往后挪了。
  answer: 当然需要，最起码需要X到X年左右的时间，这个周期是非常长的，没办法的。
  asker: 提问者
  timerange: ''
- question: 您介绍说是早期是在外部晶圆厂做，现在是回到咱们自己的产线做了？
  answer: 没错。
  asker: 提问者
  timerange: ''
- question: 第三个现在咱们目前高端产品的整体的进度是怎么样？因为我们也有看一些公开信息，说应该是新一代产品是马上要上了，还是怎么样？
  answer: XX是已经量产的。XX今年已经能出，接下来慢慢的就会往XX方向走。如果未来新一代架构的技术比较成熟，也有可能会通过像XX这些技术，可能需要一些时间的。
  asker: 提问者
  timerange: ''
- question: 现在咱们主要是给谁供货，或者说有一些潜在的客户是国内的还是海外的？
  answer: ''
  asker: 提问者
  timerange: ''
---
# 校验稿

**Q**：第一个关于新一代产品架构，因为这块我不知道XX公司内部现在是不是有一些技术储备或者专利的一些研究，这块大概是怎么样？您能介绍一下吗？
**A**：研究的新一代架构已经算是有一定的落地了。最早是在外部晶圆厂里面去做，现在已经有一定的突破了。主要方向是从传统架构演进，单元面积确实是缩减了约 XX，通过这样的方式来解决面积方面的问题。毕竟没有先进工艺嘛，所以说在整个大的方向上面是有一定的突破的。良率比较低，目前可能只有将近 XX% 的良率，处于一个没有量产的阶段，但整个大的方向上来说是没有问题的。东西是有专利的，我们处理这段是比较早的。

**Q**：那XX因为这段时间他们有发相关的新闻，他们也在走这个技术路线方向，那我是不是也可以理解他们也是在良率这块可能还是有很大的提升空间，而不是说底层的技术方案还没有定型？
**A**：方案技术已经定型了，没有疑问。大家都是在提高良率。

**Q**：XX他们因为他们有先进工艺嘛，他们在有先进工艺的情况下，是不是他们的新一代架构方案会比国内这几家会有更优的一些密度、什么各方面一些性能指标？
**A**：本来不是一个维度，不是一个等级的。密度来说他们只是为了提高上限，而我们是为了通过工艺的限制来跟他们达到同样的制程角度。他们在同样的尺寸的密度上面上限更高，而且性能更优，不是在一个游戏里面去做的。

**Q**：既然咱们现在也是能够绕开先进工艺的方案，先天来说成本和良率可能会有一些区别。咱们的，因为现在方案定了，咱们的目标良率大概会在多少？
**A**：目标良率大概是在XX%。

**Q**：有预计几年能够实现吗？比如说X年还是怎么样？因为我们看XX什么都把新一代架构的量产时间往后挪了。
**A**：当然需要，最起码需要X到X年左右的时间，这个周期是非常长的，没办法的。

**Q**：您介绍说是早期是在外部晶圆厂做，现在是回到咱们自己的产线做了？
**A**：没错。

**Q**：第三个现在咱们目前高端产品的整体的进度是怎么样？因为我们也有看一些公开信息，说应该是新一代产品是马上要上了，还是怎么样？
**A**：XX是已经量产的。XX今年已经能出，接下来慢慢的就会往XX方向走。如果未来新一代架构的技术比较成熟，也有可能会通过像XX这些技术，可能需要一些时间的。
```

**成品稿** —— 由 LLM 基于校验稿精炼，书面化、结构化，保留全部问答对与数值：

```
## 总结

- **新一代架构技术进展**：XX公司在新一代架构技术上已有一定落地，研发路径从传统架构演进，旨在通过缩短单元面积（约 XX）来绕开先进工艺的限制。目前该技术已从外部晶圆厂转向自有产线研发，现阶段处于良率突破期（约 XX% 左右），目标良率定为 XX%。预计实现量产规模化需要约 X 至 X 年的周期。
- **行业竞争格局**：XX的方案已定型，重点在于提升良率；而国内厂商通过该方案旨在应对工艺差距，以达到同等制程水平，两者在密度与性能表现上处于不同维度。
- **高端产品进度**：公司已实现 XX 量产。预计今年可实现新一代产品的产出，后续将逐步向 XX 演进。若新一代架构技术趋于成熟，未来有望实现 XX 等更高阶技术的突破。

## Q&A

### 新一代产品架构研发

**Q：关于新一代产品架构，XX公司内部在技术储备或专利研究方面的情况如何？**
A：研究中的新一代架构技术已实现一定程度的落地。研发早期主要在外部晶圆厂进行，目前已实现突破。主要技术方向是从传统架构演进，通过缩短单元面积（约减少 XX）来解决缺乏先进工艺带来的制程问题。目前该技术尚处于良率爬坡阶段，现有良率约为 XX% 左右，但技术大方向明确。公司在这一领域的专利储备较早。

**Q：XX近期发布了相关新闻，表明其也在布局该技术路线。是否可以理解为他们的技术方案尚未定型，而是主要面临良率提升的空间？**
A：XX的技术方案已经定型，目前的重点在于提高良率。

**Q：由于XX拥有先进工艺，其新一代架构方案在密度指标上是否会优于国内厂商？**
A：两者并不在同一维度竞争。XX通过先进工艺提升的是密度上限；而国内厂商采用该方案是为了在缺乏该工艺的情况下，达到与他们相同的制程水平。即在同等尺寸下，XX的密度更高且性能更优。

**Q：既然国内方案是为了绕开先进工艺，那么在成本和良率方面是否存在差异？在方案已定的情况下，目标良率是多少？**
A：目标良率约为 XX%。

**Q：实现目标良率预计需要多少年？是否参考XX推迟量产的时间节点？**
A：预计至少需要 X 到 X 年的时间，由于技术周期较长，难以速成。

**Q：此前提到该技术早期在外部晶圆厂进行，目前是否已转入自有产线？**
A：是的，目前已回到自有产线进行研发。
```

> 说明：**校验稿**尽量保真——保留完整 Q&A 与主题要点、数字原样保留，供你对照录音逐条核对；**成品稿**由 LLM 基于校验稿书面化、结构化精炼，问答对数量守恒，数值原样保留。成品稿无需逐字重听，靠校验稿复核即可。

## 命令详解

每个命令都内置了帮助信息。**先运行 `finminutes --help` 查看全部命令，再对感兴趣的命令运行 `finminutes <命令> --help` 查看具体参数**，即可按提示一步步操作：

```bash
finminutes --help                 # 查看所有命令及用途
finminutes init --help            # 查看初始化命令的参数
finminutes process --help         # 查看处理命令的参数
finminutes transcribe --help      # 查看转写命令的参数
finminutes render --help          # 查看渲染命令的参数
finminutes config --help          # 查看配置子命令
finminutes glossary --help        # 查看术语表子命令
```

命令总览（按工作流顺序）：

| 命令              | 作用                            | 典型用法                                             |
| ----------------- | ------------------------------- | ---------------------------------------------------- |
| `config`        | 配置管理（查看/切换模型与 ASR） | `finminutes config show`                           |
| `init`          | 交互式初始化向导                | `finminutes init`                                  |
| `glossary`      | 术语表管理                      | `finminutes glossary generate -f 材料.txt -t 标签` |
| `transcribe`    | 语音转写，可直接生成校验稿      | `finminutes transcribe -a 音频.m4a`                |
| `process`       | 从转录稿生成校验稿              | `finminutes process -t 转录.txt`                   |
| `render`        | 从校验稿渲染成品稿              | `finminutes render -r 校验稿.md`                   |
| `export-prompt` | 导出网页版 AI prompt 包         | `finminutes export-prompt -t 转录.txt`             |
| `import-result` | 导入网页版 AI 结果并报告完整度  | `finminutes import-result -r 结果.txt -s 转录.txt` |

### `finminutes init`

交互式初始化向导，分两步引导 **LLM**（纪要后处理）与 **ASR**（语音转写）提供商：选提供商 → 填 API Key → 测试连接（LLM 与 ASR 都做轻量校验）。`--force` 强制重新引导。

- 提供商列表**来自配置而非硬编码**：内置预设 + 你 `config add` 的自定义项都会出现；列表没有你要的，选「手动配置」进入自定义向导。
- 菜单标签紧凑：免费提供商标「免费」+ 当前配置模型名 + 限速信息（均读配置，非硬编码）；付费提供商只显示模型名，不再有冗长描述。
- 配置采用**最小写**：只写入你选择的 provider 与 Key，其余高级参数走出厂默认。
- 结尾输出高级参数入口（`config set <点路径>` / `config add` / `config remove`）与**配置分层提醒**（优先读 `~/.finminutes/config.yaml`，项目 `finminutes/config.yaml` 仅默认值）。

### `finminutes config add`（两档）

内置预设名（如 `groq` / `openrouter_free` / `anthropic`，已在出厂 config 定义）→ **只填 API Key**；全新名称 → 全参数向导（含 max_file_size_mb、切分时长等高级参数）。自定义 LLM 提供商的 API 格式二选一：**OpenAI 兼容**（默认）/ **Anthropic**。

### `finminutes config remove`

删除用户自定义的 provider（清理残留，如早期固化进配置的 ollama）：`finminutes config remove llm ollama`。内置预设无法删除（加载时会被出厂默认恢复）；不能删当前激活或最后一个 provider。

### `finminutes process`

从转录文本生成校验稿。

```bash
finminutes process -t 转录文件.txt [-b 背景.yaml] [-g 术语表] [-m 模式] [-o 输出]
```

| 选项                 | 说明                                             |
| -------------------- | ------------------------------------------------ |
| `-t, --transcript` | 转录文本文件路径（必需）                         |
| `-b, --background` | 背景信息 YAML 文件路径                           |
| `-g, --glossary`   | 术语表标签名                                     |
| `-m, --mode`       | 流水线模式：`fast` / `full`（默认 `full`） |
| `-o, --output`     | 输出前缀或目录路径                               |

**流水线模式：**

| 模式     | 阶段                                   | 是否调用 LLM      | 产出                                 |
| -------- | -------------------------------------- | ----------------- | ------------------------------------ |
| `fast` | 预处理（ASR 格式归一化）               | 否                | `{文件名}_清洗稿.txt`              |
| `full` | 预处理 + 改写 + 摘要 + 事实校验 + 渲染 | 是（改写 + 摘要） | `{文件名}_校验稿.md`，并打印完整度 |

**`-o` 输出路径规则：**

| `-o` 的值              | 输出位置                                     |
| ------------------------ | -------------------------------------------- |
| 不指定                   | `{转录稿所在目录}/{文件名}_校验稿.md`      |
| `output/`              | `output/{文件名}_校验稿.md`                |
| `meeting_final`        | `{转录稿所在目录}/meeting_final_校验稿.md` |
| `output/meeting_final` | `output/meeting_final_校验稿.md`           |

### `finminutes transcribe`

语音转写，可选自动生成校验稿。

```bash
finminutes transcribe -a 音频文件.m4a [--no-process] [-b 背景] [-g 术语表] [-o 输出]
```

| 选项                 | 说明                                      |
| -------------------- | ----------------------------------------- |
| `-a, --audio`      | 音频文件路径（.mp3/.wav/.m4a 等）（必需） |
| `--no-process`     | 仅转录，不自动生成校验稿                  |
| `-b, --background` | 背景信息 YAML 文件路径                    |
| `-g, --glossary`   | 术语表标签名                              |
| `-o, --output`     | 输出前缀或目录路径                        |

**ASR 提供商切换：**

```bash
finminutes config show                                    # 查看当前配置
finminutes config set active_asr groq                     # 切换到 Groq（默认）
finminutes config set active_asr siliconflow              # 切换到硅基流动
```

**大文件切分：** 音频超过当前提供商限制时自动切分转录（Groq 25MB / 硅基流动 50MB，可在配置中调整），并显示切分进度。切分需要系统安装 ffmpeg。

### `finminutes render`

从校验稿渲染成品稿（调用 LLM 精炼）。

```bash
finminutes render -r 校验稿.md [-T 模板名] [-o 输出]
finminutes render -j 网页AI结果.txt                 # 直接吃网页版 AI 的原始 JSON，跳过校验稿中间格式
```

| 选项               | 说明                                                                   |
| ------------------ | ---------------------------------------------------------------------- |
| `-r, --review`   | 校验稿文件路径（与`-j` 二选一，必给一个）                            |
| `-j, --json`     | 网页版 AI 返回的原始 JSON，直接渲染成品稿（与`-r` 二选一，必给一个） |
| `-T, --template` | 成品稿模板名称（默认`default`）                                      |
| `-o, --output`   | 输出前缀或目录路径                                                     |

### `finminutes export-prompt` / `import-result`（网页版 AI 协同，0 成本）

转录稿 → 导出 prompt → 粘贴给网页版 AI（豆包/DeepSeek 等）→ 结果导入并生成校验稿 + **内容完整度报告**。

```bash
finminutes export-prompt -t 转录.txt                # 生成 {转录}_prompt.txt，可直接拖进网页版 AI
finminutes import-result -r 结果.txt -s 转录.txt     # 解析结果 → 校验稿 + 完整度 + 调试 prompt
finminutes render -j 结果.txt                        # 满意后直接渲染成品稿（跳过校验稿）
```

`import-result` 的完整度检查是**确定性数字/实体保留率**（零 LLM 调用）：

- 打印 `[完整度] X%（N/M 段转录被提取内容覆盖）`；
- 低于阈值（`coverage_threshold`，默认 **0.80**）自动写 `{结果}_调试prompt.txt`，内含当前已提取内容 + 未覆盖转录片段，拖回网页版 AI 补全后重新 `import-result` 即可；
- 未提供 `-s` 时跳过完整度检查并提示。

> **配置分层提醒**：真正生效的配置是 `~/.finminutes/config.yaml`；仓库里的 `finminutes/config.yaml` 只是「出厂默认值」，会被用户配置覆盖。改阈值请用 `finminutes config set coverage_threshold 0.8`（会写入正确位置），不要直接改仓库配置。

### `finminutes config`

配置管理。

```bash
finminutes config show                 # 显示当前激活配置（API Key 脱敏）
finminutes config set KEY VALUE        # 支持点路径，如 asr_providers.groq.max_file_size_mb 50
finminutes config add llm/asr NAME     # 新增/配置提供商（预设只填 key，自定义走全参数向导）
finminutes config remove llm/asr NAME  # 删除用户自定义 provider（如清理残留 ollama）
finminutes config list-providers       # 列出所有 LLM 提供商
```

### `finminutes glossary`

术语表管理。

```bash
finminutes glossary generate -f 材料文件.txt -t 标签名   # 从材料提取术语生成术语表
```

支持输入 `.txt` / `.md` / `.docx` / `.pdf`。

## 配置文件

所有配置集中在 `~/.finminutes/config.yaml`（由 `init` 生成，或在无此文件时使用项目内置默认 `finminutes/config.yaml`）。

```yaml
active_llm: openrouter_free
active_asr: groq

llm_providers:
  openrouter_free:
    provider: openrouter
    api_key: "${OPENROUTER_API_KEY}"          # 环境变量占位符
    model: google/gemini-2.0-flash-lite-preview-02-05
  deepseek: ...
  openai: ...
  siliconflow: ...

asr_providers:
  groq:
    provider: groq
    api_key: "${GROQ_API_KEY}"
    model: whisper-large-v3-turbo
    max_file_size_mb: 25        # 超过此大小自动切分
    chunk_duration_minutes: 10  # 每个分片时长
    overlap_seconds: 5          # 分片重叠秒数
    max_retries: 3              # 每分片重试次数
  siliconflow:
    provider: siliconflow
    api_key: "${SILICONFLOW_API_KEY}"
    model: FunAudioLLM/SenseVoiceSmall
    max_file_size_mb: 50
```

## 背景信息文件

`-b` 参数指定的背景 YAML 提供会议背景，帮助 LLM 生成更精准的纪要。示例见 `finminutes/data/backgrounds/example.yaml`：

```yaml
company: "某半导体设计公司"
industry: "半导体"
participants: "CEO 张总，CFO 李总"
known_consensus:
  - "公司计划明年申报科创板"
  - "高端存储业务预计明年放量"
meeting_purpose: "Q3 业绩展望及新产品路线图"
```

## 项目结构

```
finminutes/
├── cli.py                    # 命令行入口（init/process/transcribe/render/config/glossary）
├── config.yaml               # 内置默认配置（仅环境变量占位符，可安全公开）
├── api/                      # FastAPI HTTP 服务（serve 命令）
├── core/
│   ├── pipeline.py           # 流水线编排
│   ├── preprocessor.py       # 预处理
│   ├── rewriter.py           # 改写润色
│   ├── summarizer.py         # 结构化摘要
│   ├── fact_checker.py       # 事实校验
│   ├── renderer.py           # Markdown 渲染
│   ├── asr_client.py         # ASR 客户端（Groq / 硅基流动）
│   ├── config_manager.py     # 配置管理
│   ├── glossary_loader.py    # 术语表加载
│   ├── background_loader.py  # 背景加载
│   └── ...
├── data/
│   ├── backgrounds/          # 背景信息示例
│   └── glossary/             # 术语表
├── templates/                # 纪要模板
└── tests/                    # 测试
```

## 开发

```bash
pip install -r requirements.txt
pytest tests/ -q
```

## License

MIT

---

# English

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
- **Multiple LLMs**: free tier first with auto-fallback on rate limits; supports Anthropic / any OpenAI-compatible endpoint
- **Glossary + background**: auto-generate glossaries from interview materials, correct ASR errors
- **ASR format normalization**: transcripts from Feishu / iFlytek / Whisper etc. work directly
- **Custom templates**: final-draft structure and style are configurable
- **Zero learning curve**: `finminutes --help` + interactive prompts at every step

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
finminutes init                            # setup LLM + ASR providers & API keys
finminutes process -t transcript.txt       # text → review draft
finminutes transcribe -a meeting.m4a       # audio/video → transcript → review draft
finminutes render -r review.md             # review draft → final draft
finminutes render -j result.json           # web-AI JSON → final draft directly
```

## Free API Keys

FinMinutes runs free end-to-end: free-tier LLM + free ASR providers. All keys can be entered interactively via `finminutes init` (recommended), or set as environment variables (see Quick Start).

### LLM (minutes generation)

| Provider                                                            | Free tier                                                               | Get a key                                                                                      |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [**OpenRouter**](https://openrouter.ai) (default, recommended) | Free tier: 20 req/min, 50 req/day (accounts with < $10 lifetime credit) | Sign up at[openrouter.ai](https://openrouter.ai) → Keys → Create Key                          |
| [**SiliconFlow**](https://cloud.siliconflow.cn)                | Free models (e.g. GLM-4-9B)                                             | Sign up at[cloud.siliconflow.cn](https://cloud.siliconflow.cn) → API Keys → Create            |
| [**DeepSeek**](https://platform.deepseek.com/api_keys)         | Paid                                                                    | Sign up at[platform.deepseek.com](https://platform.deepseek.com/api_keys) → API Keys → Create |
| [**OpenAI**](https://platform.openai.com)                      | Paid                                                                    | [platform.openai.com](https://platform.openai.com) → API Keys → Create                        |

### ASR (speech-to-text)

| Provider                                                         | Free tier                           | Get a key                                                                           |
| ---------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| [**Groq**](https://console.groq.com) (default, recommended) | Free Whisper, files ≤ 25MB         | Sign up at[console.groq.com](https://console.groq.com) → API Keys → Create API Key |
| **SiliconFlow**                                            | Free SenseVoiceSmall, files ≤ 50MB | Same as above (one SiliconFlow key works for both LLM and ASR)                      |

> **Zero-cost tip**: to run the whole flow, one **Groq key (ASR)** + one **OpenRouter key (LLM)** is all you need — audio → transcript → review draft → final draft. When free quotas run out, the web-AI loop (`export-prompt` / `import-result`) is a fully free, unlimited fallback.

## Commands

| Command           | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `init`          | Interactive setup wizard (LLM + ASR)                    |
| `process`       | Transcript → review draft                              |
| `transcribe`    | Audio/video → transcript (+ optional review draft)     |
| `render`        | Review draft → final draft (`-r` draft, `-j` JSON) |
| `export-prompt` | Build a web-AI prompt pack from a transcript            |
| `import-result` | Parse web-AI JSON → review draft + completeness report |
| `config`        | show / set / add / remove / set-key / list              |
| `glossary`      | Generate/manage term glossaries                         |

## LLM Providers

- **OpenRouter** (`openrouter_free`, default): free tier, 20 req/min · 50 req/day
- **DeepSeek**, **OpenAI**, **SiliconFlow** (free tier), **Anthropic** (Claude)
- Any **OpenAI-compatible** endpoint via a custom provider; **fallback** chain auto-degrades on rate limits

## ASR Providers

- **Groq** (`groq`, default): `whisper-large-v3-turbo`, 25MB file limit
- **SiliconFlow** (`siliconflow`): `FunAudioLLM/SenseVoiceSmall`, 50MB file limit

Switch with `finminutes config set active_asr <provider>`. Video files auto-extract their audio track; files over the limit are auto-chunked (requires ffmpeg).

## Pipeline Modes

| Mode     | Stages                             | LLM calls | Output                                 |
| -------- | ---------------------------------- | --------- | -------------------------------------- |
| `fast` | preprocess only                    | No        | `{stem}_清洗稿.txt`                  |
| `full` | + rewrite + summarize + fact-check | Yes       | `{stem}_校验稿.md` + coverage report |

> `import-result` reports content completeness (deterministic, no LLM). Below the `coverage_threshold` (default 0.80) it auto-writes a debug prompt for the web-AI loop; `render -j` skips the review-draft intermediate entirely.
