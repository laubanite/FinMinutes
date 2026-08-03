import os
import re

import yaml

from finminutes.core.exceptions import FinMinutesError


def _read_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt" or ext == ".md":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise FinMinutesError("请安装 python-docx: pip install python-docx")
        except Exception as e:
            raise FinMinutesError(f"读取 docx 失败: {e}")

    if ext == ".pdf":
        texts = []
        try:
            import fitz
            doc = fitz.open(path)
            for page in doc:
                texts.append(page.get_text())
            doc.close()
            return "\n".join(texts)
        except ImportError:
            pass
        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            raise FinMinutesError("请安装 PyMuPDF 或 pdfminer.six: pip install PyMuPDF")
        except Exception as e:
            raise FinMinutesError(f"读取 pdf 失败: {e}")

    raise FinMinutesError(f"不支持的文件格式: {ext}（支持: .txt .md .docx .pdf）")


PROMPT_TEMPLATE = """你是一个专业的术语提取助手。从以下材料中提取关键术语。

每个术语需要包含：
- term（正确写法）
- context（业务含义，一句话）
- corrections（可能的错误写法列表，至少3个，包括常见的误写、同音字、ASR 常见错误）

输出 YAML 格式：

```yaml
industry: "领域名称，如：半导体、金融、医疗"
terms:
  - term: "术语"
    context: "含义"
    corrections: ["错误写法1", "错误写法2", "错误写法3"]
```

只输出 YAML，不要包含任何其他文字。

材料内容：
{text}"""


def generate_glossary(llm_client, text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(text=text[:8000])
    raw = llm_client.generate(prompt)
    data = _extract_yaml(raw)
    if not data:
        raise FinMinutesError("LLM 未能生成有效的术语表 YAML")
    return data


def _extract_yaml(text: str) -> dict | None:
    match = re.search(r"```(?:yaml)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            pass
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        pass
    return None


def save_glossary(data: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
