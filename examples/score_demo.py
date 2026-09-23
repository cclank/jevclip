"""Compare jevclip's keep/drop on the synthetic demo with the labels it was
written with.

    python examples/score_demo.py demo jevclip-out
"""
import glob
import json
import os
import sys

demo = sys.argv[1] if len(sys.argv) > 1 else "demo"
out = sys.argv[2] if len(sys.argv) > 2 else "jevclip-out"
with open(os.path.join(demo, "labels.json"), encoding="utf-8") as fh:
    labels = json.load(fh)
found = glob.glob(os.path.join(out, "*", "segments.json"))
if len(found) != 1:
    sys.exit("expected one segments.json under %s, found %d" % (out, len(found)))
with open(found[0], encoding="utf-8") as fh:
    segments = json.load(fh)["segments"]
if len(segments) != len(labels):
    sys.exit("%d segments but %d labelled blocks: segmentation did not follow the blocks" % (len(segments), len(labels)))

right = 0
print("段    预设   内容      价值   结果")
for label, seg in zip(labels, segments):
    ok = (label["label"] == "keep") == bool(seg["keep"])
    right += ok
    value = "%.2f" % seg["value"] if seg.get("value") is not None else "  - "
    print("%-5s %-5s %-8s %s   %s %s  %s" % (seg["id"], label["label"], label["name"], value,
                                             "留" if seg["keep"] else "弃", "✓" if ok else "✗", "；".join(seg["reasons"])))
print("\n与预设一致 %d/%d" % (right, len(labels)))
