"""Compare jevclip's keep/drop on the synthetic demo with the labels it was
written with, second by second: how much of each block was kept. Works at
any segment length, and a block's last sentence swept into the next segment
shows up as a partly kept block rather than hiding behind a midpoint. When
the run cut a full version, how much of each block it still plays.

    python examples/score_demo.py demo jevclip-out
"""
import glob
import json
import os
import sys

demo = sys.argv[1] if len(sys.argv) > 1 else "demo"
out = sys.argv[2] if len(sys.argv) > 2 else "jevclip-out"
with open(os.path.join(demo, "labels.json"), encoding="utf-8") as fh:
    blocks = json.load(fh)
found = glob.glob(os.path.join(out, "*", "segments.json"))
if len(found) != 1:
    sys.exit("expected one segments.json under %s, found %d" % (out, len(found)))
with open(found[0], encoding="utf-8") as fh:
    segments = json.load(fh)["segments"]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


print("块  预设   内容      留下的秒数 / 块长    结果")
good = 0
for b in blocks:
    length = b["end"] - b["start"]
    kept = sum(overlap(b["start"], b["end"], s["start"], s["end"]) for s in segments if s["keep"])
    share = kept / length
    ok = share >= 0.95 if b["label"] == "keep" else share <= 0.05
    good += ok
    print("%2d  %-5s  %-8s  %5.1f / %5.1f s (%3.0f%%)  %s" % (b["block"], b["label"], b["name"], kept, length,
                                                            100 * share, "✓" if ok else "✗"))
print("\n%d/%d 个块判对（该留的留下 ≥95%%，该删的删掉 ≥95%%）" % (good, len(blocks)))

# The full version (full.mp4) must keep every block worth keeping; here every
# block preset as "drop" is pure padding, so ideally each of them is gone too.
full_json = os.path.join(os.path.dirname(found[0]), "full.json")
if os.path.exists(full_json):
    with open(full_json, encoding="utf-8") as fh:
        clips = json.load(fh)
    print("\n去水完整版 full.mp4：")
    print("块  预设   内容      在完整版里的秒数 / 块长")
    kept_whole = removed = 0
    for b in blocks:
        length = b["end"] - b["start"]
        inside = sum(overlap(b["start"], b["end"], c["start"], c["end"]) for c in clips)
        share = inside / length
        kept_whole += b["label"] == "keep" and share >= 0.95
        removed += b["label"] == "drop" and share <= 0.05
        print("%2d  %-5s  %-8s  %5.1f / %5.1f s (%3.0f%%)" % (b["block"], b["label"], b["name"], inside, length,
                                                              100 * share))
    keeps = sum(b["label"] == "keep" for b in blocks)
    print("\n该留的块完整保留 %d/%d；该删的块删掉 %d/%d" % (kept_whole, keeps, removed, len(blocks) - keeps))
