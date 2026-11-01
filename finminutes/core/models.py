from dataclasses import dataclass, field


@dataclass
class Term:
    term: str
    context: str = ""
    corrections: list[str] = field(default_factory=list)
    # 混淆集自动扩展产物：由 `glossary expand` 按同音字生成，只用于告警、不自动替换。
    # 人工确认某条确实会错后，把它挪进 corrections，才会进入确定性替换。
    auto_variants: list[str] = field(default_factory=list)


@dataclass
class Glossary:
    industry: str = ""
    terms: list[Term] = field(default_factory=list)


@dataclass
class Background:
    company: str = ""
    industry: str = ""
    participants: str = ""
    known_consensus: list[str] = field(default_factory=list)
    meeting_purpose: str = ""


def default_background() -> Background:
    return Background(
        company="访谈对象",
        industry="未知",
        participants="访谈方与被访谈方",
        meeting_purpose="调研访谈",
    )


@dataclass
class Section:
    title: str
    prompt: str = ""


@dataclass
class Template:
    name: str
    description: str = ""
    sections: list[Section] = field(default_factory=list)
