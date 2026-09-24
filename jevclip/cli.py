import argparse
import os
import sys

from . import labels, pipeline, reel, rubric, subtitles
from .jev import MODEL, JevClient
from .llm import ChatLLM
from .store import Store

DEFAULT_DB = os.environ.get("JEVCLIP_DB", "~/.jevclip/cache.db")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="jevclip",
        description="Keep the valuable parts of a video: judged timeline, cited summary, highlight reel, "
                    "and the full video minus what is surely worthless.",
    )
    ap.add_argument("--db", default=DEFAULT_DB, help="cache of transcripts and Jev answers (env JEVCLIP_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="video(s), subtitle or script file(s), or a directory of them")
    p.add_argument("paths", nargs="+")
    p.add_argument("--subs", help="subtitles or a .txt/.md script for a single video (default: found next to it)")
    p.add_argument("--title", help="title for a single video (default: file name)")
    p.add_argument("--focus", action="append", default=[], metavar="TEXT",
                   help="what you care about; segments unrelated to every focus are dropped (repeatable, max %d)"
                   % rubric.MAX_FOCUS)
    p.add_argument("--out", default="jevclip-out", help="output folder (default ./jevclip-out)")
    p.add_argument("--max-seconds", type=float, default=180.0,
                   help="highlight reel budget in seconds; 0 keeps every valuable segment")
    p.add_argument("--no-full", action="store_true",
                   help="skip full.mp4, the whole video with only the surely worthless segments taken out")
    p.add_argument("--threshold", type=float, default=rubric.Policy.threshold,
                   help="minimum value to keep a segment (provisional default)")
    p.add_argument("--segment-seconds", type=float, default=subtitles.TARGET)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--no-cut", action="store_true", help="timeline and summary only")
    p.add_argument("--no-summary", action="store_true", help="skip the summary model")
    p.add_argument("--fast", action="store_true",
                   help="hardware H.264, macOS only (VideoToolbox): larger files, and on Apple Silicon "
                        "only a little faster, at 1080p and above; elsewhere the default encoder is used")
    p.add_argument("--no-reuse", action="store_true", help="call Jev even for segments already judged")

    p = sub.add_parser("export", help="judged segments as JSONL with an empty keep/drop label")
    p.add_argument("path")

    p = sub.add_parser("eval", help="read labeled segments back and recommend a threshold")
    p.add_argument("path")
    p.add_argument("--max-error", type=float, default=0.1,
                   help="highest acceptable share of kept segments a human would drop")

    args = ap.parse_args(argv)
    with Store(args.db) as store:
        if args.cmd == "export":
            n = labels.export(store, args.path)
            print("%d segments -> %s   fill in \"label\" with keep / drop" % (n, args.path))
            return 0
        if args.cmd == "eval":
            return _eval(args)
        return _run(store, args)


def _eval(args):
    report = labels.evaluate(labels.load(args.path), max_error=args.max_error)
    if not report["labeled"]:
        print("no labeled rows in %s (%d rows total)" % (args.path, report["total"]))
        return 1
    print("labeled %d of %d   agreement with current policy %.2f"
          % (report["labeled"], report["total"], report["agreement"]))
    print("\nthreshold  kept  error  recall")
    for s in report["sweep"]:
        mark = "  <- recommended (error <= %.2f)" % args.max_error if s["threshold"] == report["recommended"] else ""
        print("  %.2f    %4d   %.2f   %.2f%s" % (s["threshold"], s["kept"], s["error"], s["recall"], mark))
    if report["recommended"] is None:
        print("\nno threshold keeps error <= %.2f on this set" % args.max_error)
    return 0


def _run(store, args):
    if len(args.focus) > rubric.MAX_FOCUS:
        print("at most %d --focus" % rubric.MAX_FOCUS, file=sys.stderr)
        return 2
    single = len(args.paths) == 1 and not os.path.isdir(args.paths[0])
    if (args.subs or args.title) and not single:
        print("--subs and --title apply to a single video", file=sys.stderr)
        return 2

    items = pipeline.discover(args.paths, subs=args.subs, exclude=args.out)
    if not items:
        print("no videos or subtitle files found", file=sys.stderr)
        return 1
    fast = args.fast and not args.no_cut and _fast_available()
    client = JevClient(model=args.model)
    llm = None if args.no_summary else ChatLLM.from_env()
    policy = rubric.Policy(threshold=args.threshold)
    failed = 0
    try:
        for n, (video, subs) in enumerate(items, 1):
            print("[%d/%d] %s" % (n, len(items), os.path.basename(video or subs)))
            if subs is None:
                print("      skipped: no subtitles or script next to it (.srt / .vtt / .json / .txt / .md with the same name)")
                failed += 1
                continue
            try:
                r = pipeline.process(store, client, video, subs, args.out, focus=args.focus, policy=policy,
                                     max_seconds=args.max_seconds, llm=llm, cut_video=not args.no_cut,
                                     fast=fast, reuse=not args.no_reuse, title=args.title,
                                     target=args.segment_seconds, full=not args.no_full)
            except (ValueError, RuntimeError, OSError) as exc:
                print("      failed: %s" % exc)
                failed += 1
                continue
            line = "      %d 段 → 有价值 %d 段" % (r["segments"], r["kept"])
            if r["timed"]:
                line += "（%s）" % subtitles.fmt_time(r["kept_seconds"])
            if not r["timed"]:
                line += " · 文字稿没有时间点，出总结和取舍，不剪视频"
            elif video is None:
                line += " · 只有字幕，未剪视频"
            print(line)
            if r["reel"]:
                print("      高亮 %d 段（%s），含 %d/%d 个有价值片段 → highlights.mp4" % (
                    r["clips"], subtitles.fmt_time(r["reel_seconds"]), r["reel_segments"], r["kept"]))
            if r["full"]:
                print("      去水完整版 %s，删掉 %d 段（%s） → full.mp4" % (
                    subtitles.fmt_time(r["full_seconds"]), r["full_removed"],
                    subtitles.fmt_time(r["full_removed_seconds"])))
            elif r["full_note"]:
                print("      去水完整版：%s" % r["full_note"])
            extra = []
            if r["reused"]:
                extra.append("%d 段读缓存" % r["reused"])
            if r["undecided"]:
                extra.append("%d 段未判断（%s）" % (r["undecided"], ", ".join(r["errors"])))
            if r["summary_note"]:
                extra.append(r["summary_note"])
            if r["unknown_citations"]:
                extra.append("总结里删掉 %d 处无效引用" % len(r["unknown_citations"]))
            if r["full_between"]:
                extra.append("%d 段没进总结但有具体内容，留在完整版里（报告里列出）" % len(r["full_between"]))
            if r["summary_uncovered"]:
                extra.append("总结漏了 %d 个有价值片段（报告里列出）" % len(r["summary_uncovered"]))
            if r["flagged"]:
                extra.append("总结里 %d 行的数字或英文名称在引用处找不到，已标 ⚠" % r["flagged"])
            u = r["usage"]
            print("      Jev %d 次请求 $%.4f%s → %s/"
                  % (u["requests"], u["usd"], "  · " + "；".join(extra) if extra else "", r["folder"]))
    finally:
        client.close()
    u = client.usage
    print("\ntotal: %d video(s), Jev %d requests, %d input tokens, $%.4f%s"
          % (len(items), u["requests"], u["input_tokens"], u["usd"],
             ", %d failed or skipped" % failed if failed else ""))
    return 1 if failed == len(items) else 0


def _fast_available():
    try:
        if reel.can_fast():
            return True
    except RuntimeError:  # no ffmpeg at all; each video says so when it is cut
        return False
    print("--fast 要用 macOS 的 VideoToolbox，这台机器的 ffmpeg 没有，改用默认编码 libx264", file=sys.stderr)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
