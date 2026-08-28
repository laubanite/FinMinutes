# FinMinutes 演示 App

一个用于 **AI 产品经理作品集** 的可点击演示：把一段访谈/路演转录，一键生成「校验稿 + 成品稿 + 内容完整度」，零成本（免费档 LLM + ASR）。

## 它演示了什么（作品集叙事点）

1. **双稿闭环**：校验稿（保真核对，数字原样保留）+ 成品稿（书面化、结构化、可直接交付）。
2. **确定性内容完整度**：用确定性算法（数字保留率）自证"内容没丢"，**不依赖 LLM 给自己打分**——AI PM 面试的高价值反模式。
3. **零成本工程**：免费档优先 + 限流自动降级，全流程不花钱。
4. **交互即产品**：30 秒可点击，面试官无需装 CLI 即可体验。

## 本地运行

前置：`finminutes` 已安装（`pip install -e .`），且已配置免费档 API Key（`finminutes init`，建议 OpenRouter + Groq）。LLM 也可选用 OrcaRouter（OpenAI 兼容，[注册获取 Key](https://www.orcarouter.ai/ref/ref_fc88ee354797b100a58a)）。

```bash
pip install gradio
python demo/app.py
```

浏览器打开 http://127.0.0.1:7860 → 点「生成纪要」。（示例转录已预填，点「载入示例」可恢复。）

> 首次调用 LLM 约 10–30 秒（免费档）。若没配置 Key，界面会提示先运行 `finminutes init`。

## 部署到 HuggingFace Spaces（免费，便于发给面试官）

1. 在 huggingface.co 新建 Space：SDK 选 **Gradio**，硬件选 **CPU basic**（免费）。
2. 上传 `demo/app.py`、`demo/sample_transcript.txt`，并把 `app.py` 移到 Space 根目录。
3. 添加 `requirements.txt`（Space 根目录）：
   ```
   finminutes
   gradio
   ```
4. **配置 API Key（两种方式之一）**：
   - 在 Space **Settings → Variables and secrets** 添加 `OPENROUTER_API_KEY`、`GROQ_API_KEY` 等环境变量；
   - 或在 Space 内放置 `~/.finminutes/config.yaml`（不推荐公开）。
   > 注意：免费 Space 是 CPU，长音频 ASR/大转录会较慢，演示以**粘贴短转录**为主最稳。

## 目录

- `app.py` — Gradio 应用
- `sample_transcript.txt` — 示例转录（含数字，展示"数字不丢"）
