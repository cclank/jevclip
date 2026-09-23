"""Which kept segments make the highlight reel, and cutting it with ffmpeg."""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from .subtitles import Cue


@dataclass
class Clip:
    start: float
    end: float
    verdicts: list
    out_start: float = 0.0


def pick(verdicts, max_seconds=180.0, pad=0.3, gap=1.0, duration=None):
    """Best kept segments until the budget is spent, played in their original
    order; neighbours merge into one clip. `max_seconds` 0 keeps them all."""
    ranked = sorted((v for v in verdicts if v.keep), key=lambda v: -v.highlight)
    chosen, total = [], 0.0
    for v in ranked:
        length = v.segment.end - v.segment.start
        if max_seconds and chosen and total + length > max_seconds:
            continue
        chosen.append(v)
        total += length
    clips = []
    for v in sorted(chosen, key=lambda v: v.segment.start):
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
