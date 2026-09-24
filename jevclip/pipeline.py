"""Finding videos and their subtitles, and running one video end to end."""

import glob
import json
import os
import re

from . import reel, report, rubric, subtitles
from .llm import LLMError
from .summary import summarize

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".flv", ".ts", ".mts")
# Preference order when several sit next to a video; scripts come last.
SUB_SUFFIXES = (".srt", ".vtt", ".json") + subtitles.SCRIPT_SUFFIXES
# a.zh.srt, a.zh-CN.vtt, a.wav.srt are subtitles of a.mp4; a.1.srt is not
_SUB_TAG = re.compile(r"^(?:[a-z]{2,3}(?:[-_][a-z0-9]{2,4})?|chs|cht|wav|mp3|m4a|flac|audio)$", re.I)


def find_subtitles(video):
    folder, name = os.path.split(video)
    stem = os.path.splitext(name)[0]
    for suffix in SUB_SUFFIXES:
        for candidate in (stem + suffix, name + suffix):
            path = os.path.join(folder, candidate)
            if os.path.isfile(path):
                return path
        for path in sorted(glob.glob(os.path.join(glob.escape(folder), glob.escape(stem) + ".*" + suffix))):
            if _SUB_TAG.match(os.path.basename(path)[len(stem) + 1 : -len(suffix)]):
                return path
    return None


def find_video(subs):
    folder, name = os.path.split(subs)
    base = os.path.splitext(name)[0]
    for stem in (base, os.path.splitext(base)[0]):
        if stem.lower().endswith(VIDEO_SUFFIXES) and os.path.isfile(os.path.join(folder, stem)):
            return os.path.join(folder, stem)
        for suffix in VIDEO_SUFFIXES:
            path = os.path.join(folder, stem + suffix)
            if os.path.isfile(path):
                return path
    return None


def discover(paths, subs=None, exclude=None):
    """(video or None, subtitles or None) pairs. A video with no subtitles
    comes back with None — reported, never guessed at. In a directory a
    .txt / .md counts only when it has a video's name; on its own it may be
    any note, so pass it by path. `exclude` keeps a previous run's output
    (its highlight reels) from being picked up."""
    exclude = os.path.abspath(exclude) if exclude else None
    items, used = [], set()
    for path in paths:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            files = []
            for root, dirs, names in os.walk(path):
                here = os.path.abspath(root)
                if exclude and (here == exclude or here.startswith(exclude + os.sep)):
                    dirs[:] = []
                    continue
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                files += [os.path.join(root, n) for n in sorted(names) if not n.startswith(".")]
            for f in files:
                if f.lower().endswith(VIDEO_SUFFIXES):
                    found = find_subtitles(f)
                    items.append((f, found))
                    used.add(found)
            for f in files:
                if f.lower().endswith((".srt", ".vtt")) and f not in used:
                    items.append((None, f))
                    used.add(f)
        elif path.lower().endswith(VIDEO_SUFFIXES):
            items.append((path, subs or find_subtitles(path)))
        elif path.lower().endswith(SUB_SUFFIXES):
            items.append((find_video(path), path))
        else:
            raise ValueError("%s: not a video, a subtitle file or a directory" % path)
    return items


def process(store, client, video, subs, out_dir, focus=(), policy=None, max_seconds=180.0,
            llm=None, cut_video=True, fast=False, reuse=True, title=None, target=subtitles.TARGET):
    """One video end to end. Writes report.md, segments.json and
    highlights.json, plus highlights.mp4 / highlights.srt when there is a
    video to cut. Returns a summary of what happened."""
    policy = policy or rubric.Policy()
    transcript = store.ingest(subs, video, title=title, target=target)
    before = dict(client.usage)
    verdicts = rubric.assess(rubric.judge(store, client, transcript, focus, reuse=reuse), policy)
    spent = {k: client.usage[k] - before[k] for k in before}

    will_cut = bool(cut_video and video and transcript.timed)
    duration = reel.probe_duration(video) if will_cut else None
    clips = reel.pick(verdicts, max_seconds, duration=duration) if transcript.timed else []
    folder = os.path.join(out_dir, transcript.doc_id)
    os.makedirs(folder, exist_ok=True)

    reel_path = reel_seconds = None
    if will_cut and clips:
        reel_path = os.path.join(folder, "highlights.mp4")
        reel_seconds = reel.cut(video, clips, reel_path, fast=fast)
        subtitles.write_srt(reel.retime(transcript.cues, clips), os.path.join(folder, "highlights.srt"))

    summary, unknown, flagged, note = None, [], 0, None
    if llm is None:
        note = "未配置总结模型，只给原文摘录"
    else:
        try:
            summary, unknown, flagged = summarize(llm, transcript.title, verdicts)
        except LLMError as exc:
            note = "总结模型调用失败：%s" % exc

    with open(os.path.join(folder, "segments.json"), "w", encoding="utf-8") as fh:
        json.dump({"doc_id": transcript.doc_id, "title": transcript.title, "video": video, "subtitles": subs,
                   "rubric": rubric.RUBRIC, "focus": list(focus), "policy": policy.__dict__,
                   "segments": [report.segment_record(v) for v in verdicts]}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "highlights.json"), "w", encoding="utf-8") as fh:
        json.dump([{"out_start": round(c.out_start, 3), "start": round(c.start, 3), "end": round(c.end, 3),
                    "time": subtitles.fmt_range(c.start, c.end), "segments": [v.segment.id for v in c.verdicts]}
                   for c in clips], fh, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report.render(transcript, verdicts, clips, policy, focus, spent, reel_seconds,
                               summary, unknown, note, flagged))

    return {
        "doc_id": transcript.doc_id,
        "title": transcript.title,
        "timed": transcript.timed,
        "folder": folder,
        "segments": len(verdicts),
        "kept": sum(1 for v in verdicts if v.keep),
        "kept_seconds": sum(v.segment.end - v.segment.start for v in verdicts if v.keep),
        "undecided": sum(1 for v in verdicts if v.keep is None),
        "errors": sorted({v.error_code for v in verdicts if v.status != "ok"}),
        "clips": len(clips),
        "reel": reel_path,
        "reel_seconds": reel_seconds,
        "summary": bool(summary),
        "summary_note": note,
        "unknown_citations": unknown,
        "flagged": flagged,
        "usage": spent,
        "reused": sum(1 for v in verdicts if v.reused),
    }
