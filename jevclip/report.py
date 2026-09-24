"""report.md and segments.json: what was kept, what was dropped and why."""

from .rubric import KINDS
from .subtitles import flat, fmt_range, fmt_time

DIGEST = 8  # excerpts shown; on a dense video nearly everything is kept


def _snippet(text, limit=60):
    text = flat(text)
    text = text if len(text) <= limit else text[:limit] + "…"
    return text.replace("|", "\\|")


def render(transcript, verdicts, clips, policy, focus, usage, reel_seconds=None,
           summary=None, unknown=(), summary_note=None, flagged=0):
    kept = [v for v in verdicts if v.keep]
    undecided = [v for v in verdicts if v.keep is None]
    length = verdicts[-1].segment.end if verdicts else 0.0
    kept_seconds = sum(v.segment.end - v.segment.start for v in kept)
    out = ["# %s" % transcript.title, ""]
    line = "时长 %s · 有价值 %d/%d 段（%s）" % (fmt_time(length), len(kept), len(verdicts), fmt_time(kept_seconds))
    if clips:
        line += " · 高亮 %d 段（%s）" % (len(clips), fmt_time(reel_seconds or sum(c.end - c.start for c in clips)))
    out.append(line)
    cached = sum(1 for v in verdicts if v.reused)
    out.append("Jev：%d 次请求，$%.4f%s%s" % (
        usage["requests"], usage["usd"],
        "，%d 段读缓存" % cached if cached else "",
        "；**%d 段未判断**，没有计入取舍" % len(undecided) if undecided else ""))
    if focus:
        out.append("关注点：%s" % "；".join(focus))
    out.append("门槛（临时，待标注校准）：价值 ≥ %.2f，闲话 < %.2f，可疑 < %.2f%s" % (
        policy.threshold, policy.junk_limit, policy.hype_limit,
        "，关注 ≥ %.2f" % policy.focus_min if focus else ""))
    out.append("")

    out += ["## 总结", ""]
    if summary:
        out.append("> 由总结模型根据保留的片段写成，方括号里的时间点由程序从片段编号换算。")
        if flagged:
            out.append("> **%d 行里的数字或英文名称，在它引用的片段和前后相邻片段里都找不到，已在行末标 ⚠，请回原片核对。**" % flagged)
        out.append("")
        out.append(summary)
        if unknown:
            out += ["", "> 删掉了 %d 处不存在或未保留的片段引用：%s" % (len(unknown), "、".join(unknown))]
    else:
        out.append("（%s）" % (summary_note or "没有达到门槛的片段"))
    out.append("")

    best = sorted(sorted(kept, key=lambda v: -v.value)[:DIGEST], key=lambda v: v.segment.start)
    out += ["## 要点摘录（原文%s）" % ("，价值最高的 %d 段" % DIGEST if len(kept) > DIGEST else ""), ""]
    for v in best:
        out.append("- [%s] %s · %.2f — %s" % (
            fmt_range(v.segment.start, v.segment.end), KINDS[v.kind][0], v.value, _snippet(v.segment.text, 90)))
    if not kept:
        out.append("（无）")
    out.append("")

    out += ["## 时间线", "", "| 片段 | 时间 | 类型 | 价值 | 取舍 | 原文 |", "|---|---|---|---|---|---|"]
    for v in verdicts:
        if v.keep is None:
            verdict, kind, value = "? " + "；".join(v.reasons), "-", "-"
        else:
            verdict = "✓ 保留" if v.keep else "✗ " + "；".join(v.reasons)
            kind, value = KINDS[v.kind][0], "%.2f" % v.value
        out.append("| %s | %s | %s | %s | %s | %s |" % (
            v.segment.id, fmt_range(v.segment.start, v.segment.end), kind, value, verdict, _snippet(v.segment.text)))
    out.append("")

    if clips:
        out += ["## 高亮视频", "", "| 高亮中 | 原视频 | 片段 |", "|---|---|---|"]
        for clip in clips:
            out.append("| %s | %s | %s |" % (
                fmt_time(clip.out_start), fmt_range(clip.start, clip.end),
                " ".join(v.segment.id for v in clip.verdicts)))
        out.append("")
    return "\n".join(out)


def segment_record(v):
    record = {
        "id": v.segment.id,
        "start": round(v.segment.start, 3),
        "end": round(v.segment.end, 3),
        "time": fmt_range(v.segment.start, v.segment.end),
        "text": v.segment.text,
        "status": v.status,
        "keep": v.keep,
        "reasons": v.reasons,
    }
    if v.status == "ok":
        record.update({
            "kind": v.kind,
            "kind_probabilities": v.answers["kind"].get("probabilities"),
            "substance": round(v.substance, 3),
            "hype": round(v.hype, 3),
            "standalone": round(v.standalone, 3),
            "focus": None if v.focus is None else round(v.focus, 3),
            "value": round(v.value, 4),
        })
    else:
        record["error"] = v.error_code
    return record
