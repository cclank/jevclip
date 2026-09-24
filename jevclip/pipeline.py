"""Finding videos and their subtitles, and running one video end to end."""

import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

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
            llm=None, cut_video=True, fast=False, reuse=True, title=None, target=subtitles.TARGET,
            full=True):
    """One video end to end. Writes report.md, segments.json and
    highlights.json; with a video to cut, highlights.mp4 / .srt — the best few
    minutes — and full.mp4 / .srt / .json, the whole video with only what was
    judged surely worthless taken out. The summary is written while ffmpeg
    cuts. Returns a summary of what happened."""
    policy = policy or rubric.Policy()
    transcript = store.ingest(subs, video, title=title, target=target)
    before = dict(client.usage)
    verdicts = rubric.assess(rubric.judge(store, client, transcript, focus, reuse=reuse), policy)
    spent = {k: client.usage[k] - before[k] for k in before}

    will_cut = bool(cut_video and video and transcript.timed)
    duration = reel.probe_duration(video) if will_cut else None
    clips = reel.pick(verdicts, max_seconds, duration=duration) if transcript.timed else []
    trimmed = reel.trim(verdicts, duration=duration) if will_cut and full else []
    taken_out = [v for v in reel.removed(verdicts, trimmed) if v.skip]
    # Dropped from the reel and the summary, yet not worthless: say where they are.
    between = [v for v in verdicts if v.keep is False and not v.skip]
    # Only a run that judged every segment may clear out an earlier run's cuts;
    # a missing key or a network outage must not cost the user their videos.
    complete = all(v.keep is not None for v in verdicts)
    folder = os.path.join(out_dir, transcript.doc_id)
    os.makedirs(folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(_summary, llm, transcript.title, verdicts)

        reel_path = reel_seconds = None
        if will_cut and clips:
            reel_path = os.path.join(folder, "highlights.mp4")
            reel_seconds = reel.cut(video, clips, reel_path, fast=fast)
            subtitles.write_srt(reel.retime(transcript.cues, clips), os.path.join(folder, "highlights.srt"))
        elif will_cut and complete:
            _clear(folder, "highlights")

        full_path = full_seconds = full_note = None
        if will_cut and full:
            if all(v.keep is None for v in verdicts):
                full_note = "没有判断结果，未出去水完整版"
            elif not trimmed:
                full_note = "所有片段都判为没价值，没有去水完整版"
            elif not taken_out:
                full_note = "没有要删的片段，去水完整版就是原片"
            if full_note:
                if complete:
                    _clear(folder, "full")
            else:
                full_path = os.path.join(folder, "full.mp4")
                full_seconds = reel.cut(video, trimmed, full_path, fast=fast)
                subtitles.write_srt(reel.retime(transcript.cues, trimmed), os.path.join(folder, "full.srt"))
                _write_json(os.path.join(folder, "full.json"), _clip_records(trimmed))

        summary, unknown, flagged, uncovered, note = writing.result()

    _write_json(os.path.join(folder, "segments.json"), {
        "doc_id": transcript.doc_id, "title": transcript.title, "video": video, "subtitles": subs,
        "rubric": rubric.RUBRIC, "focus": list(focus), "policy": policy.__dict__,
        "segments": [report.segment_record(v) for v in verdicts]})
    _write_json(os.path.join(folder, "highlights.json"), _clip_records(clips))
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report.render(transcript, verdicts, clips, policy, focus, spent, reel_seconds,
                               summary, unknown, note, flagged, uncovered,
                               full_clips=trimmed if full_path else None, full_seconds=full_seconds,
                               full_removed=taken_out, full_note=full_note, duration=duration,
                               full_between=between if (full_path or full_note) else ()))

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
        "reel_segments": sum(len(c.verdicts) for c in clips),
        "full": full_path,
        "full_seconds": full_seconds,
        "full_removed": len(taken_out) if full_path else 0,
        "full_removed_seconds": (duration - full_seconds) if full_path else 0.0,
        "full_note": full_note,
        "full_between": [v.segment.id for v in between] if (full_path or full_note) else [],
        "summary": bool(summary),
        "summary_note": note,
        "unknown_citations": unknown,
        "flagged": flagged,
        "summary_uncovered": uncovered or [],
        "usage": spent,
        "reused": sum(1 for v in verdicts if v.reused),
    }


def _summary(llm, title, verdicts):
    """(summary, unknown, flagged, uncovered, note); a failed call is a note."""
    if llm is None:
        return None, [], 0, None, "未配置总结模型，只给原文摘录"
    try:
        return (*summarize(llm, title, verdicts), None)
    except LLMError as exc:
        return None, [], 0, None, "总结模型调用失败：%s" % exc


def _clip_records(clips):
    return [{"out_start": round(c.out_start, 3), "start": round(c.start, 3), "end": round(c.end, 3),
             "time": subtitles.fmt_range(c.start, c.end), "segments": [v.segment.id for v in c.verdicts]}
            for c in clips]


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _clear(folder, stem):
    """A cut an earlier run made that this run decided against would sit
    next to a report that no longer describes it."""
    for suffix in (".mp4", ".srt", ".json"):
        path = os.path.join(folder, stem + suffix)
        if os.path.isfile(path) and not (stem == "highlights" and suffix == ".json"):
            os.remove(path)
