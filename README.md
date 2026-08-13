# FinMinutes

AI 金融会议纪要 Agent —— 从录音/转录文本到专业会议纪要的一站式工具。

免费 API + 一站式流水线，拯救万千小黑工。

[English](#english) | 中文

## 简介

FinMinutes 是一个面向金融场景的AI 会议纪要工具，支持从**录音**或**转录稿**两种输入方式，自动生成结构化、可溯源的专业纪要。

 **输入** ：

* 录音文件（`.m4a` / `.mp3` / `.wav` 等）
* 转录稿（来自飞书妙记、讯飞听见、Whisper、Groq 等任意 ASR 工具）

**输出：** 校验稿（完整 Q&A + 数值标记）+ 成品稿（书面化、结构化纪要）

**核心价值：**用户只需要听 1 遍录音，对着校验稿复核，成品稿即可直接交付。


## 特性

- 语音转写：支持Groq Whisper / 硅基流动 SenseVoiceSmall，自动切分长音频
- 术语管理：自动从访谈清单/材料提取术语，纠正ASR识别错误
- 背景信息：提供会议上下文，提升 LLM 理解准确度
- 事实校验：数字溯源 + 异常标记，杜绝纪要凭空捏造数据
- 双稿闭环：修改校验稿 → 重新渲染成品稿，只需改一份文件
- 模板定制：用户可自定义成品稿模板（章节结构、语言风格）

## 安装

要求：Python 3.10+，[ffmpeg](https://ffmpeg.org/)（大文件切分时需要）

```bash
git clone <你的仓库地址>
cd finminutes
pip install -r requirements.txt
pip install -e .          # 安装 finminutes 命令
```

### 安装 ffmpeg

ffmpeg 仅在音频超过 ASR 提供商限制（Groq 25MB / 硅基流动 50MB）需要切分时使用。按你的系统选择：

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

> 请帮我安装并配置 finminutes 项目（一个 AI 会议纪要工具），仓库地址：https://github.com/laubanite/FinMinutes
>
> 1. 从 https://github.com/laubanite/FinMinutes 克隆项目到本地
> 2. 检查 Python 版本是否为 3.10 及以上，不满足则提示我升级
> 3. 检查系统是否已安装 ffmpeg（`ffmpeg -version`），未安装则按我的系统（Windows/macOS/Linux）给出并执行安装命令，安装后提醒我重新打开终端
> 4. 执行 `pip install -r requirements.txt`
> 5. 执行 `pip install -e .`
> 6. 验证 `finminutes --help` 能正常运行，然后告诉我下一步执行 `finminutes init` 完成配置

## 快速开始

### 1. 初始化配置

```bash
finminutes init
```

交互式选择 LLM 提供商并输入 API Key。配置保存在 `~/.finminutes/config.yaml`（不会写入项目目录）。

> 也可直接设置环境变量：`${OPENROUTER_API_KEY}`、`${DEEPSEEK_API_KEY}`、`${OPENAI_API_KEY}`、`${GROQ_API_KEY}`、`${SILICONFLOW_API_KEY}`

### 2. 生成会议纪要

```bash
# 方式一：从转录文本直接生成
finminutes process -t 转录文件.txt

# 方式二：从音频开始（语音转写 + 自动生成纪要）
finminutes transcribe -a 会议录音.m4a

# 方式三：先转写，再单独生成纪要
finminutes transcribe -a 会议录音.m4a --no-process
finminutes process -t 会议录音_转录稿.txt
```

生成结果：

| 产物   | 路径                    |
| ------ | ----------------------- |
| 转录稿 | `{文件名}_转录稿.txt` |
| 校验稿 | `{文件名}_校验稿.md`  |
| 成品稿 | `{文件名}_成品稿.md`  |

### 输出示例（脱敏）

以下是同一段访谈生成的校验稿与成品稿示例（内容均为虚构脱敏数据）。

**校验稿** —— 保留原始 Q&A，数字与原文核对，未在原文出现的数字会标 `❓[待确认]`：

```
---
qa_pairs:
- question: 第一个关于DRAM的4F²架构，因为这块我不知道XX公司内部现在是不是有一些技术储备或者专利的一些研究，这块大概是怎么样？您能介绍一下吗？
  answer: 研究的4F²的技术已经算是有一定的落地了。最早是在外部晶圆厂里面去做，现在已经有一定的突破了。主要方向是从传统的6F2变4F²嘛，单元面积确实是缩减了三分之一，通过这样的方式来解决面积方面的问题。毕竟没有EUV嘛，所以说在整个大的方向上面是有一定的突破的。良率比较低，目前可能只有将近20%几的良率，处于一个没有量产的阶段，但整个大的方向上来说是没有问题的。东西是有专利的，我们处理这段是比较早的。
  asker: 提问者
  timerange: ''
- question: 那XX因为这段时间他们有发相关的新闻，他们也在走这个技术路线方向，那我是不是也可以理解他们也是在良率这块可能还是有比较大的提升空间，而不是说底层的技术方案还没有定型？
  answer: 方案技术已经定型了，没有疑问。大家都是在提高良率。
  asker: 提问者
  timerange: ''
- question: XX他们因为他们有EUV嘛，他们在有EUV的情况下，是不是他们的4F²方案会比国内这几家会有更优的一些密度、什么各方面一些性能指标？
  answer: 本来不是一个维度，不是一个等级的。密度来说他们只是为了提高上限，而我们是为了通过EUV的限制来跟他们达到同样的制程角度。他们在同样的尺寸的密度上面上限更高，而且性能更优，不是在一个游戏里面去做的。
  asker: 提问者
  timerange: ''
- question: 既然咱们现在也是能够绕过EUV的方案，先天来说成本和良率可能会有一些区别。咱们的，因为现在方案定了，咱们的目标良率大概会在多少？
  answer: 目标良率大概是在XX%。
  asker: 提问者
  timerange: ''
- question: 有预计几年能够实现吗？比如说三年还是怎么样？因为我们看XX什么都把4F²的量产时间往后挪了。
  answer: 当然需要，最起码需要X到X年左右的时间，这个周期是非常长的，没办法的。
  asker: 提问者
  timerange: ''
- question: 您介绍说是早期是在外部晶圆厂做，现在是回到咱们自己的产线做了？
  answer: 没错。
  asker: 提问者
  timerange: ''
- question: 第三个现在咱们目前HBM整体的进度是怎么样？因为我们也有看一些公开信息，说应该是HBM3是马上要上了，还是怎么样？
  answer: XX是已经量产的。XX今年已经能出，接下来慢慢的就会往XX方向走。如果未来4F²的技术比较成熟，也有可能会通过像XX这些技术，可能需要一些时间的。
  asker: 提问者
  timerange: ''
- question: 现在咱们主要是给谁供货，或者说有一些潜在的客户是国内的还是海外的？
  answer: ''
  asker: 提问者
  timerange: ''
---
# 校验稿

**Q**：第一个关于DRAM的4F²架构，因为这块我不知道XX公司内部现在是不是有一些技术储备或者专利的一些研究，这块大概是怎么样？您能介绍一下吗？
**A**：研究的4F²的技术已经算是有一定的落地了。最早是在外部晶圆厂里面去做，现在已经有一定的突破了。主要方向是从传统的6F2变4F²嘛，单元面积确实是缩减了三分之一，通过这样的方式来解决面积方面的问题。毕竟没有EUV嘛，所以说在整个大的方向上面是有一定的突破的。良率比较低，目前可能只有将近20%几的良率，处于一个没有量产的阶段，但整个大的方向上来说是没有问题的。东西是有专利的，我们处理这段是比较早的。

**Q**：那XX因为这段时间他们有发相关的新闻，他们也在走这个技术路线方向，那我是不是也可以理解他们也是在良率这块可能还是有很大的提升空间，而不是说底层的技术方案还没有定型？
**A**：方案技术已经定型了，没有疑问。大家都是在提高良率。

**Q**：XX他们因为他们有EUV嘛，他们在有EUV的情况下，是不是他们的4F²方案会比国内这几家会有更优的一些密度、什么各方面一些性能指标？
**A**：本来不是一个维度，不是一个等级的。密度来说他们只是为了提高上限，而我们是为了通过EUV的限制来跟他们达到同样的制程角度。他们在同样的尺寸的密度上面上限更高，而且性能更优，不是在一个游戏里面去做的。

**Q**：既然咱们现在也是能够绕过EUV的方案，先天来说成本和良率可能会有一些区别。咱们的，因为现在方案定了，咱们的目标良率大概会在多少？
**A**：目标良率大概是在XX%。

**Q**：有预计几年能够实现吗？比如说X年还是怎么样？因为我们看XX什么都把4F²的量产时间往后挪了。
**A**：当然需要，最起码需要X到X年左右的时间，这个周期是非常长的，没办法的。

**Q**：您介绍说是早期是在外部晶圆厂做，现在是回到咱们自己的产线做了？
**A**：没错。

**Q**：第三个现在咱们目前HBM整体的进度是怎么样？因为我们也有看一些公开信息，说应该是HBM3是马上要上了，还是怎么样？
**A**：XX是已经量产的。XX今年已经能出，接下来慢慢的就会往XX方向走。如果未来4F²的技术比较成熟，也有可能会通过像XX这些技术，可能需要一些时间的。
```

**成品稿** —— 由 LLM 基于校验稿精炼，书面化、结构化，保留全部问答对与数值：

```
## 总结

- **DRAM 4F² 架构技术进展**：XX公司在 4F² 架构技术上已有一定落地，研发路径从XX演进，旨在通过缩短单元面积（约XX）来绕过 EUV 限制。目前该技术已从外部晶圆厂转向自有产线研发，现阶段处于良率突破期（约 XX左右），目标良率定为 XX%。预计实现量产规模化需要约 X 至 X 年的周期。
- **行业竞争格局**：XX的XX方案已定型，重点在于提升良率；而国内厂商通过该方案旨在应对XX，以达到同等制程角度，两者在XX与性能表现上处于不同维度。
- **HBM 产品进度**：公司已实现 XX 量产。预计今年可实现XX产品的产出，后续将逐步向 XX 演进。若 4F² 技术趋于成熟，未来有望实现 XX 等更高阶技术的突破。

## Q&A

### DRAM 4F² 架构研发

**Q：关于 DRAM 的 4F² 架构，XX公司内部在技术储备或专利研究方面的情况如何？**
A：研究中的 4F² 技术已实现一定程度的落地。研发早期主要在外部晶圆厂进行，目前已实现突破。主要技术方向是从传统的 6F² 向 4F² 演进，通过缩短单元面积（约减少三分之一）来解决因缺乏 EUV 带来的制程问题。目前该技术尚处于良率爬坡阶段，现有良率约为 20% 左右，但技术大方向明确。公司在这一领域的专利储备较早。

**Q：XX近期发布了相关新闻，表明其也在布局该技术路线。是否可以理解为他们的技术方案尚未定型，而是主要面临良率提升的空间？**
A：XX的技术方案已经定型，目前的重点在于提高良率。

**Q：由于XX拥有XX技术，其 4F² 方案在XX指标上是否会优于国内厂商？**
A：两者并不在同一维度竞争。XX通过 XX 提升的是XX；而国内厂商采用该方案是为了在没有XX的情况下，达到与他们相同的制程水平。即在同等尺寸下，XX的XX更高且性能更优。

**Q：既然国内方案是为了绕过 XX，那么在成本和良率方面是否存在差异？在方案已定的情况下，目标良率是多少？**
A：目标良率约为 XX%。

**Q：实现目标良率预计需要多少年？是否参考XX推迟量产的时间节点？**
A：预计至少需要 X 到 X 年的时间，由于技术周期较长，难以速成。

**Q：此前提到该技术早期在外部晶圆厂进行，目前是否已转入自有产线？**
A：是的，目前已回到自有产线进行研发。
```

> 说明：校验稿回答中的数字会与原始转录稿核对——能在原文找到则保留，找不到则加粗并标注 `❓[待确认]`（如上例的 `**20%** ❓[待确认]`）。成品稿由 LLM 精炼，问答对数量守恒，数值原样保留。

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

| 命令           | 作用                            | 典型用法                                             |
| -------------- | ------------------------------- | ---------------------------------------------------- |
| `config`     | 配置管理（查看/切换模型与 ASR） | `finminutes config show`                           |
| `init`       | 交互式初始化向导                | `finminutes init`                                  |
| `glossary`   | 术语表管理                      | `finminutes glossary generate -f 材料.txt -t 标签` |
| `transcribe` | 语音转写，可直接生成校验稿      | `finminutes transcribe -a 音频.m4a`                |
| `process`    | 从转录稿生成校验稿              | `finminutes process -t 转录.txt`                   |
| `render`     | 从校验稿渲染成品稿              | `finminutes render -r 校验稿.md`                   |
| `export-prompt` | 导出网页版 AI prompt 包      | `finminutes export-prompt -t 转录.txt`             |
| `import-result` | 导入网页版 AI 结果并报告完整度 | `finminutes import-result -r 结果.txt -s 转录.txt` |

### `finminutes init`

交互式初始化向导，选择 LLM 提供商、输入 API Key、测试连接。`--force` 可强制重新配置。

### `finminutes process`

从转录文本生成校验稿。

```bash
finminutes process -t 转录文件.txt [-b 背景.yaml] [-g 术语表] [-m 模式] [-o 输出]
```

| 选项                 | 说明                                                            |
| -------------------- | --------------------------------------------------------------- |
| `-t, --transcript` | 转录文本文件路径（必需）                                        |
| `-b, --background` | 背景信息 YAML 文件路径                                          |
| `-g, --glossary`   | 术语表标签名                                                    |
| `-m, --mode`       | 流水线模式：`fast` / `full`（默认 `full`） |
| `-o, --output`     | 输出前缀或目录路径                                              |

**流水线模式：**

| 模式     | 阶段                                   | 是否调用 LLM | 产出                          |
| -------- | -------------------------------------- | ------------ | ----------------------------- |
| `fast`   | 预处理（ASR 格式归一化）               | 否           | `{文件名}_清洗稿.txt`       |
| `full`   | 预处理 + 改写 + 摘要 + 事实校验 + 渲染 | 是（改写 + 摘要） | `{文件名}_校验稿.md`，并打印完整度 |

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

| 选项                 | 说明                                            |
| -------------------- | ----------------------------------------------- |
| `-r, --review`     | 校验稿文件路径（与 `-j` 二选一，必给一个）     |
| `-j, --json`       | 网页版 AI 返回的原始 JSON，直接渲染成品稿（与 `-r` 二选一，必给一个） |
| `-T, --template`   | 成品稿模板名称（默认`default`）                |
| `-o, --output`     | 输出前缀或目录路径                              |

### `finminutes export-prompt` / `import-result`（网页版 AI 协同，0 成本）

转录稿 → 导出 prompt → 粘贴给网页版 AI（豆包/DeepSeek 等）→ 结果导入并生成校验稿 + **内容完整度报告**。

```bash
finminutes export-prompt -t 转录.txt                # 生成 {转录}_prompt.txt，可直接拖进网页版 AI
finminutes import-result -r 结果.txt -s 转录.txt     # 解析结果 → 校验稿 + 完整度 + 调试 prompt
finminutes render -j 结果.txt                        # 满意后直接渲染成品稿（跳过校验稿）
```

`import-result` 的完整度检查是**确定性数字/实体保留率**（零 LLM 调用）：
- 打印 `[完整度] X%（N/M 段转录被提取内容覆盖）`；
- 低于阈值（`coverage_threshold`，默认 **0.60**）自动写 `{结果}_调试prompt.txt`，内含当前已提取内容 + 未覆盖转录片段，拖回网页版 AI 补全后重新 `import-result` 即可；
- 未提供 `-s` 时跳过完整度检查并提示。

> **配置分层提醒**：真正生效的配置是 `~/.finminutes/config.yaml`；仓库里的 `finminutes/config.yaml` 只是「出厂默认值」，会被用户配置覆盖。改阈值请用 `finminutes config set coverage_threshold 0.6`（会写入正确位置），不要直接改仓库配置。

### `finminutes config`

配置管理。

```bash
finminutes config show                 # 显示当前激活配置（API Key 脱敏）
finminutes config set KEY VALUE        # 仅支持 active_llm / active_asr
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
  ollama: ...

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
  - "HBM 业务预计明年放量"
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

FinMinutes — an AI agent that turns audio or transcripts into professional financial meeting minutes.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
finminutes init                            # setup LLM provider + API key
finminutes process -t transcript.txt       # text → review draft
finminutes transcribe -a meeting.m4a       # audio → transcript → review draft
finminutes render -r review.md             # review draft → final draft
```

## Commands

| Command        | Description                                   |
| -------------- | --------------------------------------------- |
| `init`       | Interactive setup wizard                      |
| `process`    | Transcript → review draft                    |
| `transcribe` | Audio → transcript (+ optional review draft) |
| `render`     | Review draft → final draft                   |
| `config`     | Show/set configuration (API keys masked)      |
| `glossary`   | Generate/manage term glossaries               |

## Pipeline Modes

| Mode   | Stages                    | LLM calls | Output                     |
| ------ | ------------------------- | --------- | -------------------------- |
| `fast` | preprocess only           | No        | `{stem}_清洗稿.txt`      |
| `full` | + rewrite + summarize + fact-check | Yes       | `{stem}_校验稿.md` + coverage report |

> `import-result` reports content completeness (deterministic, no LLM). Below the `coverage_threshold` (default 0.90) it auto-writes a debug prompt for the web-AI loop; `render --from-json` skips the review-draft intermediate entirely.

## ASR Providers

- **Groq** (`groq`, default): `whisper-large-v3-turbo`, 25MB file limit
- **SiliconFlow** (`siliconflow`): `FunAudioLLM/SenseVoiceSmall`, 50MB file limit

Switch with `finminutes config set active_asr <provider>`. Files over the limit are auto-chunked (requires ffmpeg).
