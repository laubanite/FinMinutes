import os
import re

import yaml

_QA_RE = re.compile(
    r"(?<!\w)"
    r"(?P<q_prefix>(?:\*\*)?Q(?:\d*)?(?:\*\*)?[：:]\s*)"
    r"(?P<question>.+?)"
    r"(?=\s*(?:\*\*)?A(?:\d*)?(?:\*\*)?[：:]\s*)"
    r"\s*(?P<a_prefix>(?:\*\*)?A(?:\d*)?(?:\*\*)?[：:]\s*)"
    r"(?P<answer>.+?)"
    r"(?=\s*$|(?:\r?\n){2,}|\s*(?<!\w)(?:\*\*)?Q(?:\d*)?(?:\*\*)?[：:]\s*)",
    re.DOTALL,
)

_CHINESE_RE = re.compile(
    r"(?:【问】\s*)(?P<question>.+?)(?=\s*【答】)"
    r"\s*【答】\s*(?P<answer>.+?)"
    r"(?=\s*$|(?:\r?\n){2,}|\s*【问】)",
    re.DOTALL,
)


def _split_frontmatter(text: str) -> tuple[dict | None, str]:
    lines = text.split("\n")
    fm_start = -1
    fm_end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            if fm_start == -1:
                fm_start = i
            elif fm_end == -1:
                fm_end = i
                break
    if fm_start == -1 or fm_end == -1:
        return None, text

    fm_text = "\n".join(lines[fm_start + 1 : fm_end])
    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError:
        data = None
    body = "\n".join(lines[fm_end + 1 :])
    return data, body


def _assemble_file(data: dict, body: str) -> str:
    yaml_str = yaml.safe_dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{yaml_str}\n---\n\n{body.strip()}\n"


def sync_frontmatter_with_body(file_path: str, qa_pairs: list[dict]):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    data, body = _split_frontmatter(content)
    if data is None:
        return

    data["qa_pairs"] = qa_pairs
    new_content = _assemble_file(data, body)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def parse_qa_pairs_from_body(body_text: str) -> list[dict]:
    result = _parse_standard_format(body_text)
    if result:
        return result
    result = _parse_simple_format(body_text)
    if result:
        return result
    result = _parse_chinese_format(body_text)
    if result:
        return result
    result = _parse_heuristic(body_text)
    if result:
        return result
    return []


def _trim_q_prefix(text: str) -> str:
    return re.sub(
        r"^(?:\*\*)?Q(?:\d*)?(?:\*\*)?[：:]\s*",
        "",
        text,
    ).strip()


def _trim_a_prefix(text: str) -> str:
    return re.sub(
        r"^(?:\*\*)?A(?:\d*)?(?:\*\*)?[：:]\s*",
        "",
        text,
    ).strip()


def _parse_standard_format(text: str) -> list[dict]:
    pairs = []
    for m in _QA_RE.finditer(text):
        question = m.group("question").strip()
        answer = m.group("answer").strip()
        if question or answer:
            pairs.append({"question": question, "answer": answer, "asker": ""})
    return pairs


def _parse_simple_format(text: str) -> list[dict]:
    pairs = []
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        q_match = re.match(r"^Q(?:\d*)?[：:]\s*(.+)", line)
        if q_match:
            question = q_match.group(1).strip()
            answer = ""
            i += 1
            while i < len(lines):
                a_line = lines[i].strip()
                a_match = re.match(r"^A(?:\d*)?[：:]\s*(.+)", a_line)
                if a_match:
                    answer = a_match.group(1).strip()
                    i += 1
                    break
                elif re.match(r"^Q(?:\d*)?[：:]\s*", a_line):
                    break
                else:
                    i += 1
            if answer:
                pairs.append({"question": question, "answer": answer, "asker": ""})
            else:
                pairs.append({"question": question, "answer": "", "asker": ""})
                continue
        else:
            i += 1
    return pairs


def _parse_chinese_format(text: str) -> list[dict]:
    pairs = []
    for m in _CHINESE_RE.finditer(text):
        question = m.group("question").strip()
        answer = m.group("answer").strip()
        if question or answer:
            pairs.append({"question": question, "answer": answer, "asker": ""})
    return pairs


def _parse_heuristic(text: str) -> list[dict]:
    pairs = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        lines = [l.strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue
        question = lines[0]
        if len(question) < 4:
            continue
        if question.lstrip().startswith("#"):
            continue  # 跳过 markdown 标题（如「## 主题要点」下的章节标题）
        label = question.split("：")[0].split(":")[0]
        if re.match(r"^([A-Za-z\u4e00-\u9fff])\1+$", label):
            continue
        answer = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        pairs.append({"question": question, "answer": answer, "asker": ""})
    return pairs
