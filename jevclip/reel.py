"""Which kept segments make the highlight reel, and cutting it with ffmpeg."""

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .subtitles import Cue

LEAD_IN = 0.5  # below this "standalone", a segment is only shown after the one before it
ADJACENT = 3.0  # seconds; a segment further back than this is not its lead-in
# Openings that only make sense after what came before. On a real narrated
# video Jev scored "而另一个来自于 browser-use" and "第二个场景" 0.56-0.57
# standalone — too close to 0.5 to separate — while the words say it plainly.
_OPENER = re.compile(
    r"^\s*(?:而|所以|但是?|这些|另一|其次|第[二三四五六七八九十]|最后|还有|同样|因此"
    r"|(?:and|but|so|also|another|second(?:ly)?|third(?:ly)?|finally|these)\b)",
    re.I,
)


@dataclass
class Clip:
    start: float
    end: float
    verdicts: list
    out_start: float = 0.0


def pick(verdicts, max_seconds=180.0, pad=0.3, gap=2.0, duration=None):
    """Kept segments for the reel, played in their original order; chosen
    segments less than `gap` apart play as one clip, pause included, rather
    than as a jump cut. `max_seconds` 0 keeps every kept segment.

    With a budget, the chosen set is the one whose value × length adds up
    highest while fitting (a 0/1 knapsack in whole seconds) — ranking and
    filling greedily left budget unused whenever long segments came first.
    A segment that opens mid-thought (a connective such as "第三…", or
    standalone below LEAD_IN) is taken together with the kept segment right
    before it, or not at all.
    """
    order = sorted(verdicts, key=lambda v: v.segment.start)
    chosen = [v for v in order if v.keep] if not max_seconds else _knapsack(order, int(max_seconds), pad)
    clips = []
    for v in chosen:
        start = max(0.0, v.segment.start - pad)
        end = v.segment.end + pad
        if duration:
            end = min(end, duration)
        if clips and start - clips[-1].end <= gap:
            clips[-1].end = max(clips[-1].end, end)
            clips[-1].verdicts.append(v)
        else:
            clips.append(Clip(start, end, [v]))
    return clips


def _needs_previous(order, i):
    v = order[i]
    if i == 0 or v.standalone is None:
        return False
    if v.standalone >= LEAD_IN and not _OPENER.match(v.segment.text):
        return False
    prev = order[i - 1]
    return bool(prev.keep) and v.segment.start - prev.segment.end <= ADJACENT


def _knapsack(order, budget, pad=0.0):
    """Walk the segments in order keeping, for every budget b and whether the
    last segment was taken, the best total of value × seconds. The "last
    taken" bit is what lets a segment require its predecessor. Each segment
    is charged its padding too, so the reel stays within the budget."""
    none = float("-inf")
    best = [[0.0, none] for _ in range(budget + 1)]  # [b][taken?]; nothing taken yet
    steps = []
    for i, v in enumerate(order):
        seconds = v.segment.end - v.segment.start
        weight = math.ceil(seconds + 2 * pad)
        gain = v.highlight * seconds if v.keep else None
        allowed = (1,) if _needs_previous(order, i) else (0, 1)
        nxt = [[none, none] for _ in range(budget + 1)]
        back = {}
        for b in range(budget + 1):
            for c in (0, 1):  # skip this segment
                if best[b][c] > nxt[b][0]:
                    nxt[b][0], back[(b, 0)] = best[b][c], (b, c)
            if gain is not None and b >= weight:  # take it
                for c in allowed:
                    if best[b - weight][c] > none and best[b - weight][c] + gain > nxt[b][1]:
                        nxt[b][1], back[(b, 1)] = best[b - weight][c] + gain, (b - weight, c)
        best = nxt
        steps.append(back)
    state = max(((b, c) for b in range(budget + 1) for c in (0, 1)), key=lambda s: best[s[0]][s[1]])
    taken = []
    for i in range(len(order) - 1, -1, -1):
        if state[1]:
            taken.append(order[i])
        state = steps[i][state]
    return taken[::-1]


def _tool(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError("%s not found; install ffmpeg to cut highlight reels" % name)
    return path


def _run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
        raise RuntimeError("%s failed: %s" % (os.path.basename(cmd[0]), tail))
    return proc.stdout


def probe_duration(path):
    out = _run([_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path])
    return float(out.strip())


def _codec(fast):
    if fast:
        return ["-c:v", "h264_videotoolbox", "-b:v", "10M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]


def cut(video, clips, out_path, fast=False):
    """Re-encode each clip (frame-accurate starts), then join them losslessly.
    Each clip's real length is measured, so `out_start` — and the re-timed
    subtitles built from it — stay in sync. Returns the reel's length."""
    ffmpeg = _tool("ffmpeg")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    offset = 0.0
    with tempfile.TemporaryDirectory(prefix="jevclip-cut-") as tmp:
        parts = []
        for i, clip in enumerate(clips):
            part = os.path.join(tmp, "part%04d.mp4" % i)
            _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                  "-ss", "%.3f" % clip.start, "-i", video, "-t", "%.3f" % (clip.end - clip.start),
                  "-map", "0:v:0", "-map", "0:a:0?", *_codec(fast),
                  "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", part])
            clip.out_start = offset
            offset += probe_duration(part)
            parts.append(part)
        listing = os.path.join(tmp, "parts.txt")
        with open(listing, "w", encoding="utf-8") as fh:
            fh.writelines("file '%s'\n" % p.replace("'", "'\\''") for p in parts)
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
              "-i", listing, "-c", "copy", "-movflags", "+faststart", out_path])
    return offset


def retime(cues, clips):
    """The source subtitles, moved onto the reel's timeline."""
    out = []
    for clip in clips:
        for cue in cues:
            if cue.end <= clip.start or cue.start >= clip.end:
                continue
            start = max(cue.start, clip.start) - clip.start + clip.out_start
            end = min(cue.end, clip.end) - clip.start + clip.out_start
            if end - start >= 0.2:
                out.append(Cue(start, end, cue.text))
    return out
