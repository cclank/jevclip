"""One SQLite file: transcripts, and every answer Jev has given about them.

A transcript is kept verbatim — cue texts one per line — with the cue timings
alongside. Segments are ranges of whole cues, so any segment's text and
position can be replayed exactly from what is stored. A script without
timestamps is stored the same way; its positions are paragraph numbers.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass

from . import subtitles
from .subtitles import Cue, fmt_range

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts(
    doc_id       TEXT PRIMARY KEY,
    version      INTEGER NOT NULL,
    title        TEXT NOT NULL,
    video        TEXT,
    subtitles    TEXT NOT NULL,
    text         TEXT NOT NULL,
    cues         TEXT NOT NULL,
    segments     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    added_at     REAL NOT NULL,
    timed        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS judgments(
    id           INTEGER PRIMARY KEY,
    ts           REAL NOT NULL,
    request_hash TEXT NOT NULL,
    rubric       TEXT NOT NULL,
    doc_id       TEXT NOT NULL,
    doc_version  INTEGER NOT NULL,
    segment      TEXT NOT NULL,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    focus        TEXT NOT NULL DEFAULT '[]',
    status       TEXT NOT NULL,
    answers      TEXT,
    error_code   TEXT,
    model        TEXT,
    reused       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS judgments_request ON judgments(request_hash, status);
"""


@dataclass
class Segment:
    id: str
    start: float
    end: float
    char_start: int
    char_end: int
    text: str
    paras: tuple = None  # (first, last) paragraph of a script without timestamps

    @property
    def where(self):
        """How a person finds this segment: a time range, or ¶ paragraphs."""
        if self.paras:
            first, last = self.paras
            return "¶%d" % first if first == last else "¶%d–%d" % (first, last)
        return fmt_range(self.start, self.end)


@dataclass
class Transcript:
    doc_id: str
    version: int
    title: str
    video: str
    subtitles: str
    text: str
    cues: list
    segments: list
    timed: bool = True


class Store:
    def __init__(self, path=":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            self.path = os.path.expanduser(self.path)
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(transcripts)")}
        if "timed" not in columns:  # a cache made before scripts were supported
            self.db.execute("ALTER TABLE transcripts ADD COLUMN timed INTEGER NOT NULL DEFAULT 1")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def ingest(self, subtitles_path, video=None, doc_id=None, title=None, target=subtitles.TARGET):
        """Store a transcript and its segmentation. Same text and same
        segments is a no-op apart from file locations and title; anything
        else bumps the version."""
        cues = subtitles.parse(subtitles_path)
        if not cues:
            raise ValueError("%s: no text found" % subtitles_path)
        timed = all(c.para is None for c in cues)
        lines, cue_map, pos = [], [], 0
        for cue in cues:
            lines.append(cue.text)
            entry = [pos, pos + len(cue.text), round(cue.start, 3), round(cue.end, 3)]
            cue_map.append(entry if timed else entry + [cue.para])
            pos += len(cue.text) + 1
        text = "\n".join(lines) + "\n"
        ranges = [list(r) for r in subtitles.segment(cues, target=target)]
        # Timings and paragraph positions are part of the transcript too. If
        # only those change, Jev can reuse its text-based answers while cuts
        # and citations must follow the new positions.
        digest = hashlib.sha256(json.dumps(
            [text, cue_map, ranges, timed], ensure_ascii=False).encode("utf-8")).hexdigest()

        source = video or subtitles_path
        doc_id = doc_id or slug(source)
        title = title or os.path.splitext(os.path.basename(source))[0]
        video = os.path.abspath(video) if video else None
        subs = os.path.abspath(subtitles_path)
        row = self.db.execute(
            "SELECT version, content_hash FROM transcripts WHERE doc_id=?", (doc_id,)
        ).fetchone()
        with self.db:
            if row and row["content_hash"] == digest:
                self.db.execute(
                    "UPDATE transcripts SET title=?, video=?, subtitles=? WHERE doc_id=?",
                    (title, video, subs, doc_id),
                )
            else:
                self.db.execute(
                    """INSERT INTO transcripts(doc_id, version, title, video, subtitles, text, cues,
                           segments, content_hash, added_at, timed) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(doc_id) DO UPDATE SET
                           version=excluded.version, title=excluded.title, video=excluded.video,
                           subtitles=excluded.subtitles, text=excluded.text, cues=excluded.cues,
                           segments=excluded.segments, content_hash=excluded.content_hash,
                           added_at=excluded.added_at, timed=excluded.timed""",
                    (doc_id, (row["version"] + 1) if row else 1, title, video, subs, text,
                     json.dumps(cue_map), json.dumps(ranges), digest, time.time(), int(timed)),
                )
        return self.transcript(doc_id)

    def transcript(self, doc_id):
        r = self.db.execute("SELECT * FROM transcripts WHERE doc_id=?", (doc_id,)).fetchone()
        if r is None:
            raise KeyError(doc_id)
        text, cue_map, timed = r["text"], json.loads(r["cues"]), bool(r["timed"])
        cues = [Cue(e[2], e[3], text[e[0] : e[1]], None if timed else e[4]) for e in cue_map]
        segments = []
        for n, (first, stop) in enumerate(json.loads(r["segments"]), 1):
            cs, ce = cue_map[first][0], cue_map[stop - 1][1]
            paras = None if timed else (cue_map[first][4], cue_map[stop - 1][4])
            segments.append(Segment("S%d" % n, cue_map[first][2], cue_map[stop - 1][3], cs, ce,
                                    text[cs:ce], paras))
        return Transcript(r["doc_id"], r["version"], r["title"], r["video"], r["subtitles"],
                          text, cues, segments, timed)

    def cached(self, request_hash):
        """(answers, model) for an identical request answered before, or None.
        Only successful answers count; failures are retried."""
        row = self.db.execute(
            "SELECT answers, model FROM judgments WHERE request_hash=? AND status='ok' "
            "ORDER BY id DESC LIMIT 1",
            (request_hash,),
        ).fetchone()
        return (json.loads(row["answers"]), row["model"] or "") if row else None

    def log(self, transcript, verdicts, focus, rubric):
        now, focus_json = time.time(), json.dumps(list(focus), ensure_ascii=False)
        rows = [
            (now, v.key, rubric, transcript.doc_id, transcript.version, v.segment.id,
             v.segment.start, v.segment.end, v.segment.char_start, v.segment.char_end, focus_json,
             v.status, json.dumps(v.answers, ensure_ascii=False) if v.answers is not None else None,
             v.error_code, v.model, int(v.reused))
            for v in verdicts
        ]
        with self.db:
            self.db.executemany(
                """INSERT INTO judgments(ts, request_hash, rubric, doc_id, doc_version, segment,
                       t_start, t_end, char_start, char_end, focus, status, answers, error_code,
                       model, reused)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )


def slug(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^\w一-鿿.-]+", "-", stem).strip("-") or "video"
    digest = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:6]
    return "%s-%s" % (stem.lower(), digest)
