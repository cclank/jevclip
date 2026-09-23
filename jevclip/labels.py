"""Human labels in, a threshold out.

`export` writes every judged segment with an empty `label`; a person fills in
keep / drop; `evaluate` reads them back and finds the value threshold whose
kept segments a human would reject no more than `max_error` of the time.
"""

import json

from .rubric import RUBRIC, Policy, Verdict, assess
from .store import Segment
from .subtitles import fmt_range

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def export(store, path, policy=None):
    """Only segments whose transcript is still at the judged version, so the
    labeler reads exactly the text Jev read. Cached re-reads are skipped."""
    policy = policy or Policy()
    rows = store.db.execute(
        """SELECT j.doc_id, j.segment, j.t_start, j.t_end, j.char_start, j.char_end, j.focus,
                  j.answers, t.title, t.text
           FROM judgments j
           JOIN transcripts t ON t.doc_id = j.doc_id AND t.version = j.doc_version
           WHERE j.rubric = ? AND j.status = 'ok' AND j.reused = 0
           ORDER BY j.id""",
        (RUBRIC,),
    ).fetchall()
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            text = r["text"][r["char_start"] : r["char_end"]]
            seg = Segment(r["segment"], r["t_start"], r["t_end"], r["char_start"], r["char_end"], text)
            v = assess([Verdict(seg, "ok", json.loads(r["answers"]))], policy)[0]
            gates = v.p_junk < policy.junk_limit and v.hype < policy.hype_limit and (
                v.focus is None or v.focus >= policy.focus_min)
            fh.write(json.dumps({
                "doc_id": r["doc_id"], "title": r["title"], "segment": r["segment"],
                "time": fmt_range(r["t_start"], r["t_end"]), "focus": json.loads(r["focus"]),
                "text": text, "kind": v.kind, "value": round(v.value, 4), "gates_ok": gates,
                "keep": v.keep, "reasons": v.reasons, "label": None,
            }, ensure_ascii=False) + "\n")
            count += 1
    return count


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(rows, max_error=0.1):
    """At each value threshold: how many kept segments a human would drop
    (error) and how much of what the human wanted survived (recall). The
    junk / hype / focus gates stay as exported; only the threshold moves."""
    labeled = [r for r in rows if r.get("label") in ("keep", "drop")]
    if not labeled:
        return {"labeled": 0, "total": len(rows)}
    wanted = sum(r["label"] == "keep" for r in labeled)
    sweep = []
    for t in THRESHOLDS:
        kept = [r for r in labeled if r.get("gates_ok") and r["value"] >= t]
        wrong = sum(r["label"] == "drop" for r in kept)
        sweep.append({
            "threshold": t,
            "kept": len(kept),
            "error": wrong / len(kept) if kept else 0.0,
            "recall": (len(kept) - wrong) / wanted if wanted else 0.0,
        })
    fit = [s for s in sweep if s["kept"] and s["error"] <= max_error]
    return {
        "labeled": len(labeled),
        "total": len(rows),
        "agreement": sum(bool(r.get("keep")) == (r["label"] == "keep") for r in labeled) / len(labeled),
        "sweep": sweep,
        "recommended": fit[0]["threshold"] if fit else None,
    }
