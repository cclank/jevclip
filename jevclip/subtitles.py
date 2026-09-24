"""Subtitles or a plain-text script in; cues and segments out. No network, no model."""

import html
import json
import os
import re
from dataclasses import dataclass

# Seconds a segment aims for. 45 s mixed a demo, a waitlist aside and the
# episode outline into one segment on a real video, and the average dropped
# the demo; at 20 s each landed in its own segment and was judged right.
TARGET = 20.0
MAXIMUM = 90.0
MINIMUM = 15.0
PAUSE = 0.8  # a gap this long between cues is a natural break
# A sentence end is only a fallback break; if a pause follows within this many
# seconds, the segment runs on to it. Breaking at the full stop left a topic's
# last sentence ("所以…没必要加内存。") to be judged with the outro after it.
LOOKAHEAD = 10.0
SENTENCE_END = tuple("。！？!?；;….")

SCRIPT_SUFFIXES = (".txt", ".md")
# A script without timestamps is still cut by length, so its sentences get
# reading-time positions: characters per second of speech. They size the
# segments and are never shown as times.
RATE_CJK = 4.0
RATE_LATIN = 14.0
PARA_GAP = 1.0  # a paragraph break counts as a pause


@dataclass
class Cue:
    start: float
    end: float
    text: str
    para: int = None  # paragraph number in a script without timestamps; None when times are real


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
    """SRT, WebVTT, whisper.cpp JSON (-oj), a JSON list of {start, end, text}
    segments, or a .txt / .md script. Returns cues in order; a script without
    timestamps gives cues with `para` set and no real times."""
    suffix = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        raw = fh.read()
    if suffix in SCRIPT_SUFFIXES:
        return _parse_text(raw)
    if suffix == ".json":
        cues = _parse_json(json.loads(raw))
    elif suffix in (".srt", ".vtt"):
        cues = _parse_blocks(raw)
    else:
        raise ValueError("%s: expected .srt, .vtt, .json subtitles or a .txt / .md script" % path)
    return _clean(cues)


# A timestamp opening a line: "0:01", "12:34", "1:02:03", "[00:01:23]", "(0:05) …", "00:01 - …"
_STAMP = re.compile(
    r"^[\[(（【]?\s*((?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)\s*[\])）】]?(?:\s*[-–—|:：]\s*|\s+|$)(.*)$"
)
_MARKUP = re.compile(r"^(?:#{1,6}\s+|>\s?|[-*+]\s+)")
_ITEM = re.compile(r"^(?:#{1,6}\s|>|[-*+]\s|\d+[.、)）])")  # a line that starts its own unit
# After a Chinese stop (and any closing quote), or after a Latin one followed
# by a space — so 0.042 and v2.3 stay whole.
_BREAK = re.compile(
    r"(?<=[。！？!?；;])(?![”’」』\"')）])|(?<=[。！？!?；;][”’」』\"')）])|(?<=[.!?])\s+"
)


def _join(lines):
    return flat("\n".join(lines))


def _parse_text(raw):
    """Timestamps opening at least two lines, in order, make a timed
    transcript (YouTube's copied transcript, "[00:01] …"); otherwise it is a
    script, cut by paragraphs and sentences. Subtitles saved as .txt are
    read as subtitles."""
    if re.search(r"\d\s*-->\s*\d", raw):
        return _clean(_parse_blocks(raw))
    lines = [line.strip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    entries = []
    for line in lines:
        m = _STAMP.match(line) if line else None
        t = _seconds(m.group(1)) if m else None
        if t is not None and (not entries or t >= entries[-1][0]):
            entries.append((t, [m.group(2)] if m.group(2).strip() else []))
        elif line and entries:
            entries[-1][1].append(line)
    if len(entries) >= 2:
        return _timed_text(entries)
    return _script(raw)


def _rate(text):
    wide = sum(1 for ch in text if _WIDE.match(ch))
    return RATE_CJK if wide >= 0.3 * max(1, len(text.strip())) else RATE_LATIN


def _timed_text(entries):
    """Copied transcripts give only start times. A line ends at the next
    start or after its reading time, whichever is sooner, so a pause the
    speaker left still shows up as a gap."""
    cues = []
    for k, (start, lines) in enumerate(entries):
        text = _join(lines)
        if not text:
            continue
        spoken = max(1.0, len(text) / _rate(text))
        nxt = entries[k + 1][0] if k + 1 < len(entries) else start + spoken
        end = min(nxt, start + spoken) if nxt > start else start + spoken
        cues.append(Cue(start, end, text))
    return cues


def _script(raw):
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:  # no blank lines: every line is a paragraph
        paragraphs = [line for line in text.split("\n") if line.strip()]
    rate = _rate(text)
    cues, t = [], 0.0
    for number, paragraph in enumerate(paragraphs, 1):
        units = []  # wrapped lines join; list items, headings and quotes stand alone
        for line in (l.strip() for l in paragraph.split("\n")):
            if not line:
                continue
            if _ITEM.match(line) or not units:
                units.append([_MARKUP.sub("", line)])
            else:
                units[-1].append(line)
        for unit in units:
            for sentence in (s.strip() for s in _BREAK.split(_join(unit))):
                if sentence:
                    spoken = max(0.5, len(sentence) / rate)
                    cues.append(Cue(t, t + spoken, sentence, number))
                    t += spoken
        t += PARA_GAP
    return cues


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
        # Rolling captions update almost immediately. The same words spoken
        # after a pause are a new cue, even when no other subtitle intervenes.
        rolling = bool(out and cue.start <= out[-1].end + 0.5)
        if rolling and text == out[-1].text:  # repeated cue: extend, don't duplicate
            out[-1].end = max(out[-1].end, cue.end)
            continue
        prev = out[-1].text if out else ""
        if rolling and len(prev) >= 4 and text.startswith(prev):  # repeated last line
            text = text[len(prev) :].strip()
            if not text:
                continue
        out.append(Cue(cue.start, cue.end, text))
    return out


def _pause_soon(cues, j, limit, pause):
    """Index to break at if a pause occurs before `limit` seconds, else None."""
    for m in range(j, len(cues)):
        if cues[m].start > limit:
            return None
        if cues[m].start - cues[m - 1].end >= pause:
            return m
    return None


def segment(cues, target=TARGET, maximum=MAXIMUM, minimum=None, pause=PAUSE):
    """Group whole cues into segments of roughly `target` seconds: break at a
    pause if there is one, else at a sentence end a little later — unless a
    pause comes within LOOKAHEAD, then there — else at `maximum`. Never
    inside a cue. A tail shorter than `minimum` (default a third of the
    target) joins the segment before it. Returns (first, stop) cue index
    pairs."""
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
                later = _pause_soon(cues, j, cues[j - 1].end + LOOKAHEAD, pause)
                if later is not None and cues[later - 1].end - start <= maximum:
                    j = later
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
    one space between Latin words, none after a hyphen a line broke on
    ("confidence-" / "gated routing")."""
    out = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if out and not (_WIDE.match(out[-1]) or _WIDE.match(line[0]) or out.endswith("-")):
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
