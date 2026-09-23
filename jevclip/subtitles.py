"""Subtitles in; time-aligned cues and segments out. No network, no model."""

import html
import json
import os
import re
from dataclasses import dataclass

TARGET = 45.0  # seconds a segment aims for
MAXIMUM = 90.0
MINIMUM = 15.0
PAUSE = 0.8  # a gap this long between cues is a natural break
SENTENCE_END = tuple("。！？!?；;….")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def fmt_time(t):
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%02d:%02d" % (m, s)


def fmt_range(a, b):
    return "%s–%s" % (fmt_time(a), fmt_time(b))


_TIME = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?$")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _seconds(value):
    m = _TIME.match(value.strip())
    if not m:
        return None
    h, mins, secs, frac = m.groups()
    return int(h or 0) * 3600 + int(mins) * 60 + int(secs) + (float("0." + frac) if frac else 0.0)


def parse(path):
    """SRT, WebVTT, whisper.cpp JSON (-oj) or a JSON list of
    {start, end, text} segments. Returns cues in time order."""
    suffix = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        raw = fh.read()
    if suffix == ".json":
        cues = _parse_json(json.loads(raw))
    elif suffix in (".srt", ".vtt"):
        cues = _parse_blocks(raw)
    else:
        raise ValueError("%s: expected .srt, .vtt or .json subtitles" % path)
    return _clean(cues)


def _parse_blocks(raw):
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            if "-->" not in line:
                continue
            left, right = line.split("-->", 1)
            start = _seconds(left)
            end = _seconds(right.split()[0]) if right.split() else None
            if start is not None and end is not None:
                cues.append(Cue(start, end, " ".join(lines[i + 1 :])))
            break
    return cues


def _parse_json(data):
    items = data
    if isinstance(data, dict):
        items = data.get("transcription") or data.get("segments") or []
    cues = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("offsets"), dict):  # whisper.cpp -oj, milliseconds
            start, end = item["offsets"].get("from"), item["offsets"].get("to")
            if start is None or end is None:
                continue
            start, end = start / 1000.0, end / 1000.0
        elif "start" in item and "end" in item:
            start, end = float(item["start"]), float(item["end"])
        else:
            continue
        cues.append(Cue(start, end, str(item.get("text", ""))))
    return cues


def _clean(cues):
    out = []
    for cue in sorted(cues, key=lambda c: c.start):
        text = " ".join(_TAG.sub("", html.unescape(cue.text)).split())
        if not text or cue.end <= cue.start:
            continue
        if out and text == out[-1].text:  # repeated cue: extend, don't duplicate
            out[-1].end = max(out[-1].end, cue.end)
            continue
        prev = out[-1].text if out else ""
        if len(prev) >= 4 and text.startswith(prev):  # rolling captions repeat the last line
            text = text[len(prev) :].strip()
            if not text:
                continue
        out.append(Cue(cue.start, cue.end, text))
    return out


def segment(cues, target=TARGET, maximum=MAXIMUM, minimum=None, pause=PAUSE):
    """Group whole cues into segments of roughly `target` seconds: break at a
    pause if there is one, else at a sentence end a little later, else at
    `maximum`. Never inside a cue. A tail shorter than `minimum` (default a
    third of the target) joins the segment before it. Returns (first, stop)
    cue index pairs."""
    soft = min(maximum, target + 15.0)
    minimum = min(MINIMUM, target / 3.0) if minimum is None else minimum
    out, i, n = [], 0, len(cues)
    while i < n:
        start, j = cues[i].start, i + 1
        while j < n:
            span = cues[j - 1].end - start
            gap = cues[j].start - cues[j - 1].end
            if span >= maximum or (span >= target and gap >= pause):
                break
            if span >= soft and cues[j - 1].text.endswith(SENTENCE_END):
                break
            j += 1
        out.append((i, j))
        i = j
    if len(out) > 1 and cues[out[-1][1] - 1].end - cues[out[-1][0]].start < minimum:
        tail = out.pop()
        out[-1] = (out[-1][0], tail[1])
    return out


_WIDE = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")


def flat(text):
    """Cue lines joined for reading: no space where either side is Chinese,
    one space between Latin words."""
    out = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if out and not (_WIDE.match(out[-1]) or _WIDE.match(line[0])):
            out += " "
        out += line
    return out


def _srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as fh:
        for i, cue in enumerate(cues, 1):
            fh.write("%d\n%s --> %s\n%s\n\n" % (i, _srt_time(cue.start), _srt_time(cue.end), cue.text))
