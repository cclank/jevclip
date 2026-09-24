"""report.md and segments.json: what was kept, what was dropped and why."""

from .rubric import KINDS
from .subtitles import flat, fmt_range, fmt_time

DIGEST = 8  # excerpts shown; on a dense video nearly everything is kept


def _snippet(text, limit=60):
    text = flat(text)
    text = text if len(text) <= limit else text[:limit] + "…"
    return text.replace("|", "\\|")


def render(transcript, verdicts, clips, policy, focus, usage, reel_seconds=None,
           summary=None, unknown=(), summary_note=None, flagged=0, uncovered=None):
    kept = [v for v in verdicts if v.keep]
    undecided = [v for v in verdicts if v.keep is None]
    out = ["# %s" % transcript.title, ""]
    if transcript.timed:
        length = verdicts[-1].segment.end if verdicts else 0.0
        kept_seconds = sum(v.segment.end - v.segment.start for v in kept)
        line = "时长 %s · 有价值 %d/%d 段（%s）" % (fmt_time(length), len(kept), len(verdicts), fmt_time(kept_seconds))
    else:
        chars = sum(len(flat(v.segment.text)) for v in verdicts)
        kept_chars = sum(len(flat(v.segment.text)) for v in kept)
        line = "文字稿约 %d 字 · 有价值 %d/%d 段（约 %d 字）" % (chars, len(kept), len(verdicts), kept_chars)
    in_reel = {v.segment.id for c in clips for v in c.verdicts}
    if clips:
        line += " · 高亮 %d 段（%s），含 %d/%d 个有价值片段" % (
            len(clips), fmt_time(reel_seconds or sum(c.end - c.start for c in clips)), len(in_reel), len(kept))
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
    if not transcript.timed:
        out.append("> 文字稿没有时间点：位置用原稿的段落编号 ¶ 表示，不剪高亮视频。")
    out.append("")

    out += ["## 总结", ""]
    if summary:
        out.append("> 由总结模型根据保留的片段写成，方括号里的%s由程序从片段编号换算。"
                   % ("时间点" if transcript.timed else "段落位置"))
        if flagged:
            out.append("> **%d 行里的数字或英文名称，在它引用的片段和前后相邻片段里都找不到，已在行末标 ⚠，请回原片核对。**" % flagged)
        out.append("")
        out.append(summary)
        if unknown:
            out += ["", "> 删掉了 %d 处不存在或未保留的片段引用：%s" % (len(unknown), "、".join(unknown))]
        if uncovered is not None:
            where = {v.segment.id: v.segment.where for v in verdicts}
            out.append("")
            out.append("> 总结引用了全部 %d 个有价值片段。" % len(kept) if not uncovered else
                       "> **有 %d 个有价值片段没进总结**：%s" % (
                           len(uncovered), "、".join("%s（%s）" % (i, where[i]) for i in uncovered)))
    else:
        out.append("（%s）" % (summary_note or "没有达到门槛的片段"))
    out.append("")

    best = sorted(sorted(kept, key=lambda v: -v.value)[:DIGEST], key=lambda v: v.segment.start)
    out += ["## 要点摘录（原文%s）" % ("，价值最高的 %d 段" % DIGEST if len(kept) > DIGEST else ""), ""]
    for v in best:
        out.append("- [%s] %s · %.2f — %s" % (
            v.segment.where, KINDS[v.kind][0], v.value, _snippet(v.segment.text, 90)))
    if not kept:
        out.append("（无）")
    out.append("")

    reel_col = bool(clips)  # which kept segments made the reel, so none goes missing unnoticed
    out += ["## 时间线", "", "| 片段 | %s | 类型 | 价值 | 取舍 |%s 原文 |" % (
                "时间" if transcript.timed else "位置", " 高亮 |" if reel_col else ""),
            "|---|---|---|---|---|%s---|" % ("---|" if reel_col else "")]
    for v in verdicts:
        if v.keep is None:
            verdict, kind, value = "? " + "；".join(v.reasons), "-", "-"
        else:
            verdict = "✓ 保留" if v.keep else "✗ " + "；".join(v.reasons)
            kind, value = KINDS[v.kind][0], "%.2f" % v.value
        mark = (" %s |" % ("▶" if v.segment.id in in_reel else "")) if reel_col else ""
        out.append("| %s | %s | %s | %s | %s |%s %s |" % (
            v.segment.id, v.segment.where, kind, value, verdict, mark, _snippet(v.segment.text)))
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
    timed = v.segment.paras is None
    record = {
        "id": v.segment.id,
        "start": round(v.segment.start, 3) if timed else None,
        "end": round(v.segment.end, 3) if timed else None,
        "paragraphs": None if timed else list(v.segment.paras),
        "time": v.segment.where,
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
