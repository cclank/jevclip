"""The summary: written by a language model from kept segments only, cited by
segment id, with every id turned into a timestamp by code."""

import re

from .rubric import KINDS
from .subtitles import flat, fmt_range

SYSTEM = "你是严谨的视频内容编辑。只根据给出的字幕片段写总结，不添加片段里没有的信息。"
PROMPT = """视频标题：{title}

下面是从这段视频字幕里筛出来的有价值片段。字幕可能有语音识别错字，请按上下文理解，但不要编造。每段以编号开头。

{blocks}

请用中文输出，格式如下：
一句话总结：（不超过 60 字）

要点：
- 每条写一个要点，结尾用方括号标出依据的片段编号，例如 [S3] 或 [S3][S7]。写 3 到 8 条。

可以直接用的做法或结论：
- 同样标出编号。没有就写“无”。

只能引用上面出现过的编号。"""

_CITE = re.compile(r"\[(S\d+)\]")


def resolve(text, segments):
    """[S7] -> [03:12–04:05]. Ids that were not handed to the writer are
    removed and returned, so an invented citation can never become a
    timestamp."""
    unknown = []

    def swap(m):
        seg = segments.get(m.group(1))
        if seg is None:
            unknown.append(m.group(1))
            return ""
        return "[%s]" % fmt_range(seg.start, seg.end)

    return _CITE.sub(swap, text), unknown


def summarize(llm, title, verdicts):
    """(summary, unknown ids). No kept segment, no call."""
    kept = [v for v in verdicts if v.keep]
    if not kept:
        return None, []
    blocks = "\n\n".join("[%s]（%s）%s" % (v.segment.id, KINDS[v.kind][0], flat(v.segment.text)) for v in kept)
    text = llm.chat(SYSTEM, PROMPT.format(title=title, blocks=blocks))
    return resolve(text, {v.segment.id: v.segment for v in kept})
