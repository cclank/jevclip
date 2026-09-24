"""The summary: written by a language model from kept segments only, cited by
segment id, with every id turned into a timestamp by code — and every number
and Latin-script name checked against the text it cites. Every kept segment
must be cited; the ones a first pass leaves out get a second, narrower one."""

import re

from .rubric import KINDS
from .subtitles import flat

SYSTEM = "你是严谨的视频内容编辑。只根据给出的字幕片段写总结，不添加片段里没有的信息。"
PROMPT = """视频标题：{title}

下面是从这段视频字幕里筛出来的有价值片段。字幕可能有语音识别错字，请按上下文理解，但不要编造。每段以编号开头。

{blocks}

请用中文输出，只用下面三个标题：
一句话总结：
要点：
可以直接用的做法或结论：

要求：
- 一句话总结接在标题后面写，不超过六十个字。
- 要点按视频顺序写，每条以“- ”开头，结尾用方括号标出依据的片段编号，例如 [S3] 或 [S3][S4]。
- 每个片段都要至少被一条要点引用，不要遗漏；相邻片段讲同一件事可以合成一条。条数按内容来，不设上限，也不要为了凑数重复。
- 可以直接用的做法或结论同样每条标出编号；没有就只写一行“无”。
- 型号、产品名、人名、数字和单位照抄片段原文，不要换成你更熟悉的名称。
- 不要自己加英文翻译或括号注释；原文是中文的术语就写中文。
- 片段里没有说出来的结果不要推测，例如“我们来看看它选了哪个”之后没有交代结果，就不要写结果。
- 只能引用上面出现过的编号。"""

FILL = """视频标题：{title}

下面这些片段在已经写好的总结里还没有提到。请为它们补写要点：每条以“- ”开头，结尾用方括号标出依据的片段编号；相邻片段讲同一件事可以合成一条。只输出这些要点，不要标题和别的内容。
型号、产品名、人名、数字和单位照抄原文；不要自己加英文翻译或括号注释；片段里没说出来的结果不要推测；只能引用下面出现的编号。

{blocks}"""

_CITE = re.compile(r"\[(S\d+)\]")
# Latin-script names (GPT-5.6, typesafe-ai/jev, experimental_evaluate) and
# numbers with at least two digits or a decimal point: the details a writer
# silently "corrects". Single digits are too often a rewritten Chinese numeral.
_FACT = re.compile(r"[A-Za-z]\w*(?:[-./]\w+)*|\d+\.\d+|\d{2,}", re.ASCII)
_BULLET = re.compile(r"^\s*[-*•]\s*")
_NONE = re.compile(r"^\s*[-*•]?\s*无[。.]?\s*$")
_NAMED = re.compile(r"^(一句话总结|要点|可以直接用的做法或结论)\s*[:：]")


def _bare(line):
    return line.strip().strip("#*").strip()


def _heading(line):
    """A section title however it is dressed: "要点：", "**要点：**", "### 做法："."""
    if not line.strip() or _BULLET.match(line):
        return False
    return bool(_NAMED.match(_bare(line))) or _bare(line).endswith((":", "："))


def resolve(text, segments):
    """[S7] -> [03:12–04:05], or [¶3–4] for a script without timestamps.
    Ids that were not handed to the writer are removed and returned, so an
    invented citation can never become a position."""
    unknown = []

    def swap(m):
        seg = segments.get(m.group(1))
        if seg is None:
            unknown.append(m.group(1))
            return ""
        return "[%s]" % seg.where

    return _CITE.sub(swap, text), unknown


def _found(fact, source):
    fact = fact.lower()
    # "Noul/Boolean" names two things; each must be there, not the pair verbatim.
    return fact in source or all(part in source for part in fact.split("/") if part)


def check(text, segments, title="", allowed=None):
    """{line index: [facts]} for lines whose numbers or Latin-script names
    are nowhere near what they cite.

    `segments` is every segment in video order; `allowed` the ids the writer
    was given (default all). A cited line is held against its segments and
    their immediate neighbours — at ~20 s a point often straddles two, and a
    citation one segment off is not a fabrication. A line citing nothing is
    held against every allowed segment. The title always counts."""
    ids = list(segments)
    at = {sid: n for n, sid in enumerate(ids)}
    allowed = set(ids if allowed is None else allowed)
    everything = " ".join(flat(segments[s].text) for s in ids if s in allowed)
    problems = {}
    for i, line in enumerate(text.split("\n")):
        cited = [c for c in _CITE.findall(line) if c in allowed]
        near = sorted({ids[n] for c in cited for n in (at[c] - 1, at[c], at[c] + 1) if 0 <= n < len(ids)}, key=at.get)
        source = (" ".join(flat(segments[s].text) for s in near) if cited else everything) + " " + title
        missing = sorted({f for f in _FACT.findall(_CITE.sub("", line)) if not _found(f, source.lower())})
        if missing:
            problems[i] = missing
    return problems


def mark(text, problems):
    lines = text.split("\n")
    for i, missing in problems.items():
        lines[i] = "%s ⚠ 引用处找不到：%s" % (lines[i], "、".join(missing))
    return "\n".join(lines)


def drop_empty_none(text):
    """A section that has real items and also a stray "无" bullet loses the "无"."""
    lines = text.split("\n")
    out, section = [], []

    def flush():
        has_items = any(_BULLET.match(l) and not _NONE.match(l) for l in section)
        out.extend(l for l in section if not (has_items and _NONE.match(l)))
        section.clear()

    for line in lines:
        if line.strip() and not _BULLET.match(line) and not _NONE.match(line):
            flush()
            out.append(line)
        else:
            section.append(line)
    flush()
    return "\n".join(out)


def merge_points(text, extra, order):
    """Put `extra` bullets into the 要点 section, the whole list in video
    order (by each bullet's first citation; an uncited bullet keeps its
    neighbour's place). No 要点 heading: they go at the end under one."""
    if not extra:
        return text
    lines = text.split("\n")
    heads = [i for i, line in enumerate(lines) if _heading(line)]
    start = next((i for i in heads if _bare(lines[i]).startswith("要点")), None)
    if start is None:
        return text.rstrip() + "\n\n要点：\n" + "\n".join(extra)
    stop = next((i for i in heads if i > start), len(lines))
    body = lines[start + 1 : stop]
    keyed, last = [], -1
    for n, line in enumerate([l for l in body if _BULLET.match(l)] + list(extra)):
        ids = [order[c] for c in _CITE.findall(line) if c in order]
        last = min(ids) if ids else last
        keyed.append((last, n, line))
    others = [l for l in body if l.strip() and not _BULLET.match(l)]
    tail = [""] if stop < len(lines) else []
    return "\n".join(lines[: start + 1] + others + [l for *_, l in sorted(keyed)] + tail + lines[stop:])


def _blocks(verdicts):
    return "\n\n".join("[%s]（%s）%s" % (v.segment.id, KINDS[v.kind][0], flat(v.segment.text)) for v in verdicts)


def summarize(llm, title, verdicts):
    """(summary, unknown ids, number of lines flagged, kept ids never cited).
    No kept segment, no call."""
    kept = [v for v in verdicts if v.keep]
    if not kept:
        return None, [], 0, []
    text = drop_empty_none(llm.chat(SYSTEM, PROMPT.format(title=title, blocks=_blocks(kept))))
    cited = set(_CITE.findall(text))
    missing = [v for v in kept if v.segment.id not in cited]
    if missing:
        wanted = {v.segment.id for v in missing}
        extra = llm.chat(SYSTEM, FILL.format(title=title, blocks=_blocks(missing)))
        bullets = [l.strip() for l in extra.split("\n") if _BULLET.match(l) and set(_CITE.findall(l)) & wanted]
        text = merge_points(text, bullets, {v.segment.id: n for n, v in enumerate(verdicts)})
        cited = set(_CITE.findall(text))
    uncovered = [v.segment.id for v in kept if v.segment.id not in cited]
    given = {v.segment.id: v.segment for v in kept}
    problems = check(text, {v.segment.id: v.segment for v in verdicts}, title, allowed=given)
    text, unknown = resolve(mark(text, problems), given)
    return text, unknown, len(problems), uncovered
