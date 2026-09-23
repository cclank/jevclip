import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from jevclip import labels, pipeline, reel, rubric, subtitles, summary
from jevclip.jev import JevClient, JudgeError
from jevclip.llm import ChatLLM
from jevclip.store import Segment, Store

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
KIND_KEYS = list(rubric.KINDS)


def choice(kind, p=0.9):
    rest = (1 - p) / (len(KIND_KEYS) - 1)
    return {"type": "choice", "choice": kind, "confidence": p,
            "probabilities": {k: (p if k == kind else rest) for k in KIND_KEYS}}


def answer(kind="evidence", substance=2.6, hype=0.05, standalone=0.9, focus=()):
    out = {
        "kind": choice(kind),
        "substance": {"type": "score", "score": substance, "probabilities": {}},
        "hype": {"type": "noul", "noul": hype},
        "standalone": {"type": "noul", "noul": standalone},
    }
    for i, p in enumerate(focus, 1):
        out["focus%d" % i] = {"type": "noul", "noul": p}
    return out


def scripted(by_text):
    """Answers by the segment text it is sent; unknown text gets a plain
    valuable answer. Records every payload."""
    sent = []

    def transport(payload, api_key, timeout):
        sent.append(payload)
        text = json.loads(payload["state"])["segment"]["text"]
        for needle, ans in by_text.items():
            if needle in text:
                return {"model": "jev-1.13.0", "answers": ans, "usage": {"input_tokens": 400}}
        return {"model": "jev-1.13.0", "answers": answer(), "usage": {"input_tokens": 400}}

    transport.sent = sent
    return transport


def srt(cues):
    def t(x):
        ms = int(round(x * 1000))
        return "%02d:%02d:%02d,%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)
    return "\n".join("%d\n%s --> %s\n%s\n" % (i, t(a), t(b), text) for i, (a, b, text) in enumerate(cues, 1))


def write(folder, name, content):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# Six 10-second blocks separated by 1 s pauses, two 5-second cues each.
BLOCKS = ["寒暄", "数据", "广告", "方法", "空话", "讲解"]
CUES = []
for n, name in enumerate(BLOCKS):
    base = n * 11.0
    CUES.append((base, base + 5.0, "%s块第一句，内容是%s。" % (name, name)))
    CUES.append((base + 5.0, base + 10.0, "%s块第二句，继续讲%s。" % (name, name)))


class Temp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.store = Store()
        self.addCleanup(self.store.close)


class Subtitles(Temp):
    def test_srt_with_bom_crlf_tags_and_multiline(self):
        raw = "﻿1\r\n00:00:01,000 --> 00:00:03,500\r\n<i>第一行</i>\r\n第二行\r\n\r\n2\r\n00:00:04,000 --> 00:00:06,000\r\n{\\an8}下一句 &amp; 结束\r\n"
        cues = subtitles.parse(write(self.tmp, "a.srt", raw))
        self.assertEqual([(c.start, c.end, c.text) for c in cues],
                         [(1.0, 3.5, "第一行 第二行"), (4.0, 6.0, "下一句 & 结束")])

    def test_vtt_header_notes_settings_and_short_timestamps(self):
        raw = ("WEBVTT\n\nNOTE 这是注释\n\ncue-1\n00:01.200 --> 00:03.000 align:start position:10%\n"
               "<v 主持人>大家好</v>\n\n01:02:03.000 --> 01:02:04.500\n最后一句\n")
        cues = subtitles.parse(write(self.tmp, "a.vtt", raw))
        self.assertEqual([(c.start, c.end, c.text) for c in cues],
                         [(1.2, 3.0, "大家好"), (3723.0, 3724.5, "最后一句")])

    def test_whisper_cpp_json(self):
        data = {"transcription": [{"offsets": {"from": 0, "to": 2500}, "text": " 你好"},
                                  {"offsets": {"from": 2500, "to": 4000}, "text": " 世界"}]}
        cues = subtitles.parse(write(self.tmp, "a.json", json.dumps(data)))
        self.assertEqual([(c.start, c.end, c.text) for c in cues], [(0.0, 2.5, "你好"), (2.5, 4.0, "世界")])

    def test_generic_segment_json(self):
        data = {"segments": [{"start": 1, "end": 2, "text": "一"}, {"start": 2, "end": 3.5, "text": "二"}]}
        self.assertEqual([c.text for c in subtitles.parse(write(self.tmp, "a.json", json.dumps(data)))], ["一", "二"])

    def test_rolling_captions_are_not_duplicated(self):
        raw = srt([(0, 2, "今天我们来聊"), (2, 4, "今天我们来聊 本地模型"), (4, 6, "本地模型"), (6, 8, "本地模型")])
        cues = subtitles.parse(write(self.tmp, "a.srt", raw))
        self.assertEqual([c.text for c in cues], ["今天我们来聊", "本地模型"])
        self.assertEqual(cues[-1].end, 8)

    def test_unsupported_suffix(self):
        with self.assertRaises(ValueError):
            subtitles.parse(write(self.tmp, "a.txt", "hello"))

    def test_chinese_lines_join_without_spaces_latin_with_one(self):
        self.assertEqual(subtitles.flat("我们来看测试。\n我用一台 MacBook Air，\n跑的是 4-bit\nversion\n"),
                         "我们来看测试。我用一台 MacBook Air，跑的是 4-bit version")


class Segmenting(unittest.TestCase):
    def cues(self, spec):
        return [subtitles.Cue(a, b, t) for a, b, t in spec]

    def test_breaks_at_a_pause_after_target(self):
        self.assertEqual(subtitles.segment(self.cues(CUES), target=9.0), [(i, i + 2) for i in range(0, 12, 2)])

    def test_never_splits_a_cue_and_covers_everything(self):
        segs = subtitles.segment(self.cues([(i * 3.0, i * 3.0 + 3.0, "第%d句" % i) for i in range(40)]),
                                 target=20.0, maximum=30.0)
        self.assertEqual((segs[0][0], segs[-1][1]), (0, 40))
        for (_, stop), (start, _) in zip(segs, segs[1:]):
            self.assertEqual(stop, start)

    def test_maximum_caps_a_segment_without_pauses(self):
        cues = self.cues([(i * 3.0, i * 3.0 + 3.0, "没有句号的话") for i in range(40)])
        for a, b in subtitles.segment(cues, target=20.0, maximum=30.0):
            self.assertLessEqual(cues[b - 1].end - cues[a].start, 33.0)

    def test_sentence_end_breaks_when_there_is_no_pause(self):
        spec = [(i * 3.0, i * 3.0 + 3.0, "一句话。" if i == 6 else "半句话") for i in range(20)]
        self.assertEqual(subtitles.segment(self.cues(spec), target=5.0, maximum=60.0)[0], (0, 7))

    def test_short_tail_merges_into_the_previous_segment(self):
        cues = self.cues([(0, 20, "一"), (21, 41, "二"), (42, 45, "尾巴")])
        self.assertEqual(subtitles.segment(cues, target=18.0), [(0, 1), (1, 3)])


class Storing(Temp):
    def setUp(self):
        super().setUp()
        self.subs = write(self.tmp, "talk.srt", srt(CUES))

    def test_segments_align_with_cues_and_carry_times(self):
        t = self.store.ingest(self.subs, target=9.0)
        self.assertEqual(len(t.cues), 12)
        self.assertEqual([s.id for s in t.segments], ["S1", "S2", "S3", "S4", "S5", "S6"])
        s2 = t.segments[1]
        self.assertEqual((s2.start, s2.end), (11.0, 21.0))
        self.assertEqual(s2.text, "数据块第一句，内容是数据。\n数据块第二句，继续讲数据。")
        self.assertEqual(t.text[s2.char_start : s2.char_end], s2.text)

    def test_resegmenting_bumps_the_version(self):
        versions = [self.store.ingest(self.subs, target=x).version for x in (9.0, 9.0, 30.0)]
        self.assertEqual(versions, [1, 1, 2])

    def test_moved_files_update_without_a_new_version(self):
        first = self.store.ingest(self.subs, video="/old/place/talk.mp4", doc_id="talk")
        moved = self.store.ingest(self.subs, video="/new/place/talk.mp4", doc_id="talk")
        self.assertEqual((first.version, moved.version), (1, 1))
        self.assertEqual(moved.video, "/new/place/talk.mp4")

    def test_empty_subtitles_are_an_error(self):
        with self.assertRaises(ValueError):
            self.store.ingest(write(self.tmp, "empty.srt", ""))


class Judging(Temp):
    def setUp(self):
        super().setUp()
        self.t = self.store.ingest(write(self.tmp, "talk.srt", srt(CUES)), target=9.0)

    def client(self, transport):
        return JevClient(api_key="k", transport=transport)

    def test_one_request_per_segment_with_the_whole_rubric(self):
        transport = scripted({})
        rubric.judge(self.store, self.client(transport), self.t, focus=["本地部署"])
        self.assertEqual(len(transport.sent), 6)
        qs = transport.sent[0]["questions"]
        self.assertEqual(set(qs), {"kind", "substance", "hype", "standalone", "focus1"})
        self.assertEqual({q["type"] for q in qs.values()}, {"choice", "score", "noul"})
        self.assertIn("本地部署", qs["focus1"]["instructions"])
        self.assertEqual(len(qs["substance"]["criteria"]), 4)

    def test_state_carries_only_title_and_text(self):
        transport = scripted({})
        rubric.judge(self.store, self.client(transport), self.t)
        # requests go out concurrently, so find this segment's by its text
        raw = next(p["state"] for p in transport.sent if "数据块" in p["state"])
        self.assertEqual(json.loads(raw), {"video": {"title": "talk"}, "segment": {"text": self.t.segments[1].text}})
        self.assertNotIn("00:", raw)
        self.assertNotIn("S2", raw)

    def test_same_request_is_never_paid_for_twice(self):
        transport = scripted({})
        client = self.client(transport)
        rubric.judge(self.store, client, self.t)
        again = rubric.judge(self.store, client, self.t)
        self.assertEqual(len(transport.sent), 6)
        self.assertTrue(all(v.reused for v in again))
        rubric.judge(self.store, client, self.t, focus=["新的关注点"])
        self.assertEqual(len(transport.sent), 12)

    def test_transport_error_is_undetermined_not_low_value(self):
        def boom(payload, api_key, timeout):
            raise JudgeError("timeout")

        verdicts = rubric.assess(rubric.judge(self.store, self.client(boom), self.t))
        self.assertTrue(all(v.keep is None for v in verdicts))
        self.assertIn("未判断（timeout）", verdicts[0].reasons)
        self.assertEqual(reel.pick(verdicts), [])

    def test_malformed_answers_are_rejected(self):
        broken = answer()
        del broken["hype"]
        verdicts = rubric.judge(self.store, self.client(scripted({"数据": broken})), self.t)
        self.assertEqual((verdicts[1].status, verdicts[1].error_code), ("error", "bad_answer"))
        self.assertEqual(verdicts[0].status, "ok")

    def test_errors_are_retried_next_time(self):
        def boom(payload, api_key, timeout):
            raise JudgeError("timeout")

        rubric.judge(self.store, self.client(boom), self.t)
        transport = scripted({})
        rubric.judge(self.store, self.client(transport), self.t)
        self.assertEqual(len(transport.sent), 6)


class Policy(unittest.TestCase):
    def verdict(self, **kw):
        seg = Segment("S1", 0.0, 30.0, 0, 10, "文本")
        return rubric.assess([rubric.Verdict(seg, "ok", answer(**kw))])[0]

    def test_valuable_segment_is_kept(self):
        v = self.verdict(kind="evidence", substance=2.7)
        self.assertTrue(v.keep)
        self.assertAlmostEqual(v.value, 0.9 * (1 - v.p_junk), places=6)

    def test_small_talk_is_dropped_with_its_reason(self):
        v = self.verdict(kind="smalltalk", substance=0.2)
        self.assertFalse(v.keep)
        self.assertTrue(v.reasons[0].startswith("寒暄过渡"))

    def test_hype_is_dropped_even_when_dense(self):
        v = self.verdict(kind="insight", substance=2.8, hype=0.85)
        self.assertFalse(v.keep)
        self.assertIn("可疑说法 0.85", v.reasons)

    def test_platitudes_are_dropped_as_low_substance(self):
        v = self.verdict(kind="insight", substance=1.0)
        self.assertFalse(v.keep)
        self.assertIn("信息密度 1.0/3", v.reasons)

    def test_off_focus_is_dropped(self):
        v = self.verdict(kind="method", substance=2.8, focus=(0.1, 0.2))
        self.assertFalse(v.keep)
        self.assertIn("与关注点无关 0.20", v.reasons)

    def test_any_matching_focus_is_enough(self):
        self.assertTrue(self.verdict(kind="method", substance=2.8, focus=(0.1, 0.9)).keep)

    def test_threshold_changes_without_calling_jev_again(self):
        v = self.verdict(kind="explain", substance=1.8)
        self.assertTrue(rubric.assess([v], rubric.Policy(threshold=0.5))[0].keep)
        self.assertFalse(rubric.assess([v], rubric.Policy(threshold=0.7))[0].keep)


class Highlights(unittest.TestCase):
    def kept(self, start, end, value, standalone=1.0):
        seg = Segment("S%d" % int(start), start, end, 0, 1, "x")
        return rubric.Verdict(seg, "ok", keep=True, value=value, standalone=standalone)

    def test_budget_takes_the_best_then_plays_in_order(self):
        vs = [self.kept(0, 40, 0.6), self.kept(50, 90, 0.9), self.kept(100, 140, 0.7)]
        self.assertEqual([(c.start, c.end) for c in reel.pick(vs, max_seconds=80, pad=0)], [(50, 90), (100, 140)])

    def test_neighbours_merge_into_one_clip(self):
        clips = reel.pick([self.kept(0, 40, 0.9), self.kept(40, 80, 0.8)], max_seconds=0, pad=0.3)
        self.assertEqual([(c.start, c.end) for c in clips], [(0.0, 80.3)])

    def test_padding_is_clamped_to_the_video(self):
        clips = reel.pick([self.kept(0.1, 9.9, 0.9)], pad=0.5, duration=10.0)
        self.assertEqual((clips[0].start, clips[0].end), (0.0, 10.0))

    def test_zero_budget_keeps_everything(self):
        self.assertEqual(len(reel.pick([self.kept(i * 100, i * 100 + 60, 0.8) for i in range(5)], max_seconds=0)), 5)

    def test_standalone_segments_rank_higher(self):
        vs = [self.kept(0, 60, 0.8, standalone=0.1), self.kept(100, 160, 0.8, standalone=0.9)]
        self.assertEqual(reel.pick(vs, max_seconds=60, pad=0)[0].start, 100)

    def test_retimed_subtitles_follow_the_reel(self):
        cues = [subtitles.Cue(10, 14, "a"), subtitles.Cue(14, 18, "b"), subtitles.Cue(40, 44, "c")]
        clips = [reel.Clip(12, 18, []), reel.Clip(40, 44, [], out_start=6.0)]
        self.assertEqual([(c.start, c.end, c.text) for c in reel.retime(cues, clips)],
                         [(0, 2, "a"), (2, 6, "b"), (6, 10, "c")])


class Summary(unittest.TestCase):
    def setUp(self):
        self.segs = {"S2": Segment("S2", 11, 21, 0, 1, "数据"), "S4": Segment("S4", 33, 43, 0, 1, "方法")}

    def test_citations_become_times_and_inventions_are_dropped(self):
        text, unknown = summary.resolve("要点一 [S2]\n要点二 [S4][S9]", self.segs)
        self.assertEqual(text, "要点一 [00:11–00:21]\n要点二 [00:33–00:43]")
        self.assertEqual(unknown, ["S9"])

    def test_writer_sees_only_kept_segments_and_thinking_is_stripped(self):
        seen = []

        def transport(url, body, key, timeout):
            seen.append(body["messages"][1]["content"])
            return {"choices": [{"message": {"content": "<think>先想想 [S1]</think>一句话总结：好 [S2][S1]"}}]}

        llm = ChatLLM("https://example.invalid/v1", "k", "m", transport=transport)
        kept = rubric.Verdict(self.segs["S2"], "ok", keep=True, kind="evidence")
        dropped = rubric.Verdict(Segment("S1", 0, 10, 0, 1, "寒暄"), "ok", keep=False, kind="smalltalk")
        text, unknown = summary.summarize(llm, "标题", [dropped, kept])
        self.assertIn("[S2]", seen[0])
        self.assertNotIn("寒暄", seen[0])
        self.assertEqual(text, "一句话总结：好 [00:11–00:21]")
        self.assertEqual(unknown, ["S1"])

    def test_no_kept_segments_means_no_call(self):
        llm = ChatLLM("https://example.invalid/v1", "k", "m", transport=lambda *a: self.fail("called"))
        self.assertEqual(summary.summarize(llm, "标题", []), (None, []))

    def test_bare_host_from_env_gets_v1(self):
        env = {"MINIMAX_BASE_URL": "https://api.minimax.io", "MINIMAX_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            llm = ChatLLM.from_env()
        self.assertEqual((llm.base_url, llm.model), ("https://api.minimax.io/v1", "MiniMax-M2"))

    def test_explicit_endpoint_is_used_as_given(self):
        env = {"JEVCLIP_LLM_BASE_URL": "https://example.com/compatible-mode/v1",
               "JEVCLIP_LLM_API_KEY": "k", "JEVCLIP_LLM_MODEL": "qwen-plus", "MINIMAX_API_KEY": "x"}
        with mock.patch.dict(os.environ, env, clear=True):
            llm = ChatLLM.from_env()
        self.assertEqual((llm.base_url, llm.model), ("https://example.com/compatible-mode/v1", "qwen-plus"))

    def test_nothing_configured_means_no_writer(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ChatLLM.from_env())


class Transport(unittest.TestCase):
    def test_missing_key_is_an_error_not_a_crash(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(JudgeError) as ctx:
                JevClient(api_key="").ask("{}", {})
        self.assertEqual(ctx.exception.code, "no_api_key")

    def test_rate_limit_is_retried_once(self):
        client = JevClient(api_key="k")
        replies = [(429, b""), (200, b'{"answers": {}, "usage": {"input_tokens": 7}}')]
        with mock.patch("jevclip.jev.RETRY_AFTER", 0), \
                mock.patch.object(client, "_send", side_effect=lambda *a: replies.pop(0)):
            self.assertEqual(client.ask("{}", {}), {"answers": {}, "usage": {"input_tokens": 7}})
        self.assertEqual(client.usage["input_tokens"], 7)

    def test_a_dropped_pooled_connection_is_retried_once(self):
        client = JevClient(api_key="k")
        calls = []

        def send(*a):
            calls.append(1)
            if len(calls) == 1:
                raise JudgeError("dropped")
            return 200, b'{"answers": {}}'

        with mock.patch.object(client, "_send", side_effect=send):
            client.ask("{}", {})
        self.assertEqual(len(calls), 2)

    def test_a_second_failure_surfaces(self):
        client = JevClient(api_key="k")
        with mock.patch("jevclip.jev.RETRY_AFTER", 0), \
                mock.patch.object(client, "_send", return_value=(429, b"")):
            with self.assertRaises(JudgeError) as ctx:
                client.ask("{}", {})
        self.assertEqual(ctx.exception.code, "rate_limited")

    def test_https_proxy_is_tunnelled_with_credentials(self):
        with mock.patch.dict(os.environ, {"https_proxy": "http://user:p%40ss@127.0.0.1:7890"}, clear=True):
            conn = JevClient(api_key="k")._connect(5)
        self.assertEqual((conn.host, conn.port), ("127.0.0.1", 7890))
        self.assertEqual(conn._tunnel_host, "api.typesafe.ai")
        self.assertIn("Proxy-Authorization", conn._tunnel_headers)


class Discover(unittest.TestCase):
    def test_pairs_videos_with_their_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.mp4", "a.srt", "b.mp4", "b.zh-CN.srt", "c.srt", "d.mov",
                         "lesson.mp4", "lesson.1.mp4", "lesson.1.srt"):
                write(tmp, name, "")
            out = os.path.join(tmp, "jevclip-out", "x")
            os.makedirs(out)
            write(out, "highlights.mp4", "")
            write(out, "highlights.srt", "")
            items = pipeline.discover([tmp], exclude=os.path.join(tmp, "jevclip-out"))
            got = {(os.path.basename(v) if v else None, os.path.basename(s) if s else None) for v, s in items}
        self.assertEqual(got, {("a.mp4", "a.srt"), ("b.mp4", "b.zh-CN.srt"), (None, "c.srt"),
                               ("d.mov", None), ("lesson.mp4", None), ("lesson.1.mp4", "lesson.1.srt")})


class Labels(Temp):
    def test_export_is_labelable_and_text_is_what_jev_read(self):
        t = self.store.ingest(write(self.tmp, "talk.srt", srt(CUES)), target=9.0)
        transport = scripted({"寒暄": answer(kind="smalltalk", substance=0.2)})
        client = JevClient(api_key="k", transport=transport)
        rubric.judge(self.store, client, t)
        rubric.judge(self.store, client, t)  # cached re-read: not exported twice
        path = os.path.join(self.tmp, "segs.jsonl")
        self.assertEqual(labels.export(self.store, path), 6)
        rows = labels.load(path)
        first = next(r for r in rows if r["segment"] == "S1")
        self.assertIsNone(first["label"])
        self.assertEqual((first["time"], first["kind"], first["keep"]), ("00:00–00:10", "smalltalk", False))
        sent = next(p for p in transport.sent if "寒暄块" in p["state"])
        self.assertEqual(json.loads(sent["state"])["segment"]["text"], first["text"])

    def test_export_skips_transcripts_that_changed(self):
        subs = write(self.tmp, "talk.srt", srt(CUES))
        rubric.judge(self.store, JevClient(api_key="k", transport=scripted({})), self.store.ingest(subs, target=9.0))
        self.store.ingest(subs, target=30.0)
        self.assertEqual(labels.export(self.store, os.path.join(self.tmp, "segs.jsonl")), 0)

    def test_threshold_sweep(self):
        rows = ([{"value": 0.9, "gates_ok": True, "keep": True, "label": "keep"}] * 6
                + [{"value": 0.55, "gates_ok": True, "keep": True, "label": "drop"}] * 2
                + [{"value": 0.55, "gates_ok": True, "keep": True, "label": "keep"}] * 2
                + [{"value": 0.2, "gates_ok": True, "keep": False, "label": "drop"}] * 4
                + [{"value": 0.9, "gates_ok": False, "keep": False, "label": "drop"}]
                + [{"value": 0.9, "gates_ok": True, "keep": True, "label": None}])
        report = labels.evaluate(rows, max_error=0.1)
        self.assertEqual((report["labeled"], report["total"]), (15, 16))
        at = {s["threshold"]: s for s in report["sweep"]}
        self.assertAlmostEqual(at[0.5]["error"], 2 / 10)
        self.assertAlmostEqual(at[0.6]["error"], 0.0)
        self.assertAlmostEqual(at[0.6]["recall"], 6 / 8)
        self.assertEqual(report["recommended"], 0.6)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Cutting(unittest.TestCase):
    """Each block of the test video is a solid colour, so the reel can be
    checked frame by frame: it must show exactly the kept blocks, in order."""

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        inputs, streams = [], []
        for i, (r, g, b) in enumerate(cls.COLORS):
            inputs += ["-f", "lavfi", "-i", "color=c=0x%02x%02x%02x:s=160x90:r=25:d=11" % (r, g, b)]
            streams.append("[%d:v]" % i)
        cls.video = os.path.join(cls.tmp.name, "talk.mp4")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
             "-f", "lavfi", "-i", "sine=frequency=440:duration=66",
             "-filter_complex", "%sconcat=n=6:v=1:a=0[v]" % "".join(streams),
             "-map", "[v]", "-map", "6:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-shortest", cls.video],
            check=True,
        )
        write(cls.tmp.name, "talk.srt", srt(CUES))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def color_at(self, path, t):
        raw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "%.2f" % t, "-i", path,
             "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True,
        ).stdout
        return tuple(raw[:3])

    def closest(self, rgb):
        return min(range(len(self.COLORS)), key=lambda i: sum((a - b) ** 2 for a, b in zip(rgb, self.COLORS[i])))

    def test_reel_contains_exactly_the_kept_blocks(self):
        store = Store()
        self.addCleanup(store.close)
        transport = scripted({"寒暄": answer(kind="smalltalk", substance=0.3),
                              "广告": answer(kind="promo", substance=0.2),
                              "空话": answer(kind="filler", substance=0.4),
                              "讲解": answer(kind="explain", substance=1.0)})
        out = os.path.join(self.tmp.name, "out")
        r = pipeline.process(store, JevClient(api_key="k", transport=transport), self.video,
                             os.path.join(self.tmp.name, "talk.srt"), out, max_seconds=0, target=9.0)
        self.assertEqual((r["kept"], r["clips"]), (2, 2))  # 数据 (block 1) and 方法 (block 3)

        self.assertAlmostEqual(reel.probe_duration(r["reel"]), r["reel_seconds"], delta=0.15)
        self.assertAlmostEqual(r["reel_seconds"], 2 * 10.6, delta=0.3)
        self.assertEqual([self.closest(self.color_at(r["reel"], t)) for t in (1.0, 9.0, 12.0, 20.0)], [1, 1, 3, 3])

        with open(os.path.join(r["folder"], "highlights.json"), encoding="utf-8") as fh:
            clips = json.load(fh)
        self.assertEqual([c["segments"] for c in clips], [["S2"], ["S4"]])
        cues = subtitles.parse(os.path.join(r["folder"], "highlights.srt"))
        self.assertEqual([c.text[:3] for c in cues], ["数据块", "数据块", "方法块", "方法块"])
        self.assertAlmostEqual(cues[2].start, clips[1]["out_start"], delta=0.35)

        with open(os.path.join(r["folder"], "report.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("有价值 2/6 段", text)
        self.assertIn("✗ 推广求关注", text)
        self.assertIn("未配置总结模型", text)


if __name__ == "__main__":
    unittest.main()
