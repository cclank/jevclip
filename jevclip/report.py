"""report.md and segments.json: what was kept, what was dropped and why."""

from .rubric import KINDS
from .subtitles import flat, fmt_range, fmt_time

DIGEST = 8  # excerpts shown; on a dense video nearly everything is kept


def _snippet(text, limit=60):
    text = flat(text)
    text = text if len(text) <= limit else text[:limit] + "…"
    return text.replace("|", "\\|")


def _span(clip):
    """S3–S9 for a run of segments, S3 S7 when they are not consecutive."""
    ids = [v.segment.id for v in clip.verdicts]
    nums = [int(i[1:]) for i in ids if i[1:].isdigit()]
    if len(ids) > 2 and len(nums) == len(ids) and nums == list(range(nums[0], nums[0] + len(nums))):
        return "%s–%s" % (ids[0], ids[-1])
    return " ".join(ids)


def render(transcript, verdicts, clips, policy, focus, usage, reel_seconds=None,
           summary=None, unknown=(), summary_note=None, flagged=0, uncovered=None,
           full_clips=None, full_seconds=None, full_removed=(), full_note=None, duration=None,
           full_between=()):
    kept = [v for v in verdicts if v.keep]
    undecided = [v for v in verdicts if v.keep is None]
    out = ["# %s" % transcript.title, ""]
    if transcript.timed:
        length = duration or (verdicts[-1].segment.end if verdicts else 0.0)
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
    if full_clips:
        out.append("去水完整版 %s：只删掉 %d 段没用的（共 %s），其余原样保留" % (
            fmt_time(full_seconds or sum(c.end - c.start for c in full_clips)), len(full_removed),
            fmt_time(max(0.0, length - (full_seconds or sum(c.end - c.start for c in full_clips))))))
    elif full_note:
        out.append("去水完整版：%s" % full_note)
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

    if full_clips or full_note:
        out += ["## 去水完整版", ""]
        if full_clips:
            out += ["full.mp4 只删掉下面 %d 段：信息密度不到 %.1f/3、明确是广告（≥ %.2f），或与关注点无关。"
                    "没有字幕的画面、片头片尾和未判断的片段都保留；删点两边不到 2 秒的停顿一起删，"
                    "更长的空白可能是画面演示，保留。" % (len(full_removed), policy.skip_substance, policy.skip_promo),
                    "", "| 片段 | 原视频 | 删掉的原因 | 原文 |", "|---|---|---|---|"]
            for v in full_removed:
                out.append("| %s | %s | %s | %s |" % (
                    v.segment.id, v.segment.where, "；".join(v.reasons), _snippet(v.segment.text)))
            out += ["", "| 完整版中 | 原视频 | 片段 |", "|---|---|---|"]
            for clip in full_clips:
                out.append("| %s | %s | %s |" % (fmt_time(clip.out_start), fmt_range(clip.start, clip.end), _span(clip)))
        else:
            out.append("%s。" % full_note)
        if full_between:
            out += ["", "下面 %d 段没进高亮和总结，但留在完整版里：它们判为寒暄、可疑或价值不够，"
                        "可信息密度有 %.1f/3 以上，整段删掉会漏信息。" % (len(full_between), policy.skip_substance),
                    "", "| 片段 | 原视频 | 没进总结的原因 | 信息密度 | 原文 |", "|---|---|---|---|---|"]
            for v in full_between:
                out.append("| %s | %s | %s | %.1f | %s |" % (
                    v.segment.id, v.segment.where, "；".join(v.reasons), v.substance, _snippet(v.segment.text)))
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
        "skip": v.skip,  # taken out of the full version
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
