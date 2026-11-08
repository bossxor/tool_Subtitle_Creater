"""Translation quality boundary regressions; no GPU needed."""
import json
import sys
import unittest
import tempfile
from unittest.mock import patch, Mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import Config
from core.segmenter import Cue, build_cues
from core.stt import Word
from core.stt import STTEngine
from core.translate import translate_cues
from core.translate_gemma import translate_cues_gemma, build_prompt
from core.validate import validate_batch, translation_issue
from core.textfix import collapse_repeats, japanese_register
from core.pipeline import stage_audio, stage_translate, _stt_cache_file, _translation_cache_file, _write_json
from core.llm import LlamaServer, LlamaServerError


class FakeLLM:
    generation = 0
    restart_count = 0

    def __init__(self, output):
        self.output = output
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.output

    def chat(self, *args, **kwargs):
        return json.dumps([{"id": 0, "ko": self.output}], ensure_ascii=False)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.config = Config.load()
        self.config.set("llm.translate_parallel", 1)

    def test_repeat_compression_preserves_meaning_and_identifiers(self):
        for source in ["10000円です。", "1111号室", "11112222", "00000001", "ABABABAB", "そうそうそうそう", "待て待て待て待て"]:
            with self.subTest(source=source):
                self.assertEqual(collapse_repeats(source), source)
        self.assertEqual(collapse_repeats("あっあっあっあっあっ"), "あっあっあっ")

    def test_short_interjection_is_not_merged_across_silence(self):
        cues = build_cues([Word(1, 1.2, "あっ！"), Word(30, 31, "帰ろう。")], self.config)
        self.assertEqual([(c.start, c.end, c.text) for c in cues], [(1, 1.2, "あっ！"), (30, 31, "帰ろう。")])

    def test_tiny_fragment_merges_only_with_adjacent_incomplete_speech(self):
        cues = build_cues([Word(0, .1, "そ"), Word(.15, 1, "れだ。")], self.config)
        self.assertEqual([c.text for c in cues], ["それだ。"])
        cues = build_cues([Word(0, .1, "そ"), Word(10, 11, "れだ。")], self.config)
        self.assertEqual(len(cues), 2)

    def test_tiny_fragment_merge_respects_duration_limit(self):
        self.config.set("segmenter.max_cue_duration", 1)
        cues = build_cues([Word(0, .1, "そ"), Word(.15, 1.1, "れだ。")], self.config)
        self.assertEqual(len(cues), 2)

    def test_dialogue_mentioning_blocked_phrase_is_not_deleted(self):
        source = "チャンネル登録なんてしなくていいよ。"
        self.assertEqual([c.text for c in build_cues([Word(0, 3, source)], self.config)], [source])

    def test_polite_forms_inside_casual_words_do_not_force_polite_register(self):
        for source in ["これはますます困る。", "明日はそのままですぐ帰る。", "はいしゃに行く。",
                       "話は聞いたし、もう帰るよ。", "お前の態度は失礼だよ。",
                       "話の腰を折らないで。", "ここで泳いで。", "先に読んで。"]:
            with self.subTest(source=source):
                self.assertEqual(japanese_register(source), "casual")
        for source in ["もう行きます。", "行きますけど、いいですか？", "そうですね。", "はい、わかりました。",
                       "ごめんなさい、先生。", "承知いたしました。"]:
            with self.subTest(source=source):
                self.assertEqual(japanese_register(source), "polite")

    def test_acronym_alone_cannot_replace_a_whole_sentence(self):
        for source, output in [("DVDを見せてください。", "DVD"), ("USBでつないで。", "USB USB USB")]:
            self.assertIsNotNone(translation_issue(output, source))
        self.assertIsNone(translation_issue("DVD", "DVD。"))
        self.assertIsNone(translation_issue("Tokyo", "東京", glossary={"東京": "Tokyo"}))
        self.assertIsNotNone(translation_issue("Tokyo", "東京に行く", glossary={"東京": "Tokyo"}))

    def test_translation_keyword_does_not_exempt_meta_responses(self):
        self.assertIsNotNone(translation_issue("번역할 내용이 없습니다.", "翻訳の仕事を辞めた。"))
        self.assertIsNone(translation_issue("죄송하지만 번역할 수 없어요.", "すみません、訳せません。"))
        self.assertIsNone(translation_issue("죄송하지만 번역할 수 없어요.", "すみませんが、訳すことはできません。"))
        self.assertIsNotNone(translation_issue("번역할 내용이 없습니다.", "帰ろう。", glossary={"他": "번역할 내용이 없습니다."}))

    def test_cutoff_model_output_is_not_accepted_as_complete(self):
        server = LlamaServer(Path("server.exe"), Path("model.gguf"))
        for flags in [{"stop_type": "limit"}, {"truncated": True}, {"stopped_limit": True}]:
            response = type("Response", (), {"raise_for_status": lambda self: None,
                "json": lambda self: {"content": "고개를 들 수", **flags}})()
            with self.subTest(flags=flags), patch("core.llm.requests.post", return_value=response):
                with self.assertRaises(LlamaServerError):
                    server.complete("prompt")
        response = type("Response", (), {"raise_for_status": lambda self: None,
            "json": lambda self: {"content": "고개를 들 수 없어.", "stop_type": "eos", "truncated": False}})()
        with patch("core.llm.requests.post", return_value=response):
            self.assertEqual(server.complete("prompt"), "고개를 들 수 없어.")

    def test_glossary_names_reach_recognition_without_mutating_user_config(self):
        from core.pipeline import run_batch
        self.config.set("subtitle.worklog", False)
        with patch("core.pipeline.stage_audio", return_value={}) as audio, \
             patch("core.pipeline.stage_stt", return_value={}), \
             patch("core.pipeline.stage_segment", return_value={}), \
             patch("core.pipeline.stage_translate", return_value={}), \
             patch("core.pipeline.stage_write", return_value={}):
            run_batch([], Path("unused"), self.config, glossary={"紗夜": "사야", "東京": "도쿄"})
        recognition_config = audio.call_args.args[1]
        self.assertIn("紗夜", recognition_config.get("stt.hotwords"))
        self.assertNotIn("사야", recognition_config.get("stt.hotwords"))
        self.assertIsNone(self.config.get("stt.hotwords"))

    def test_recognition_passes_hotwords_to_whisper(self):
        engine = STTEngine.__new__(STTEngine)
        info = type("Info", (), {"language": "ja", "language_probability": 1, "duration": 1})()
        engine.pipeline = Mock()
        engine.pipeline.transcribe.return_value = ([], info)
        engine.transcribe(Path("input.wav"), language="ja", hotwords="紗夜 東京")
        self.assertEqual(engine.pipeline.transcribe.call_args.kwargs["hotwords"], "紗夜 東京")

    def test_gemma_never_accepts_english_fallback_even_with_source_acronym(self):
        for source in ["本当にありがとう。", "DVDを見せてください。"]:
            out, report = translate_cues_gemma([Cue(0, 0, 2, source)], FakeLLM("Please provide the Japanese text."), self.config)
            self.assertEqual(report.failed_ids, [0])
            self.assertEqual(out[0], source)

    def test_korean_meta_response_is_not_a_translation(self):
        for output in ["죄송합니다. 해당 문구는 번역할 내용이 없습니다.", "(빈칸)",
                       "번역할 텍스트가 없습니다.", "죄송하지만, 이 텍스트는 번역할 수 없습니다.",
                       "현재 번역할 원문이 제공되지 않았습니다."]:
            out, report = translate_cues_gemma([Cue(0, 0, 1, "申")], FakeLLM(output), self.config)
            self.assertEqual(report.failed_ids, [0])

    def test_meaningful_yes_no_responses_are_not_phonetic_fallbacks(self):
        for translator in (translate_cues, translate_cues_gemma):
            for source in ("ううん", "ええ", "うん"):
                _, report = translator([Cue(0, 0, 1, source)], FakeLLM(source), self.config)
                self.assertEqual(report.failed_ids, [0])
                self.assertEqual(report.transliterated_ids, [])

    def test_failed_meaningful_kana_is_not_transliterated(self):
        out, report = translate_cues_gemma([Cue(0, 0, 1, "ありがとう")], FakeLLM("ありがとう"), self.config)
        self.assertEqual(report.failed_ids, [0])
        self.assertEqual(report.transliterated_ids, [])
        self.assertEqual(out[0], "ありがとう")

    def test_json_backend_rejects_english(self):
        out, report = translate_cues([Cue(0, 0, 1, "帰ろう")], FakeLLM("Let's go home."), self.config)
        self.assertEqual(report.failed_ids, [0])

    def test_empty_null_and_extra_ids_are_not_accepted(self):
        for value in ["", "  ", None, 123]:
            result = validate_batch([0], json.dumps([{"id": 0, "ko": value}, {"id": 9, "ko": "외부 항목"}]))
            self.assertFalse(result.ok)
            self.assertNotIn(9, result.by_id)

    def test_source_acronym_is_preserved_in_korean(self):
        out, report = translate_cues_gemma([Cue(0, 0, 1, "DVDを見せてください。")], FakeLLM("DVD를 보여주세요."), self.config)
        self.assertEqual(out[0], "DVD를 보여주세요.")
        self.assertEqual(report.failed_ids, [])

    def test_neighboring_source_is_used_without_crossing_long_silence(self):
        cues = [Cue(0, 0, 1, "ただいま。"), Cue(1, 1.1, 2, "おかえり。"), Cue(2, 50, 51, "帰ろう。")]
        llm = FakeLLM("왔어?")
        translate_cues_gemma(cues, llm, self.config)
        contextual = [p for p in llm.prompts if "ただいま。" in p and "おかえり。" in p]
        self.assertTrue(contextual)
        self.assertTrue(all("帰ろう。" not in p for p in contextual))

    def test_targeted_translation_can_reference_the_whole_scene(self):
        scene = [Cue(0, 0, 1, "ただいま。"), Cue(1, 1.1, 2, "おかえり。")]
        llm = FakeLLM("おかえり。")
        llm.output = "어서 와."
        result, _ = translate_cues_gemma(scene[1:], llm, self.config, context_cues=scene)
        self.assertEqual(result, {1: "어서 와."})
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("Previous: ただいま。", llm.prompts[0])

    def test_explicit_glossary_is_preserved(self):
        for translator in (translate_cues_gemma, translate_cues):
            _, report = translator([Cue(0, 0, 1, "東京に行く。")], FakeLLM("Tokyo에 가."),
                                   self.config, glossary={"東京": "Tokyo"})
            self.assertEqual(report.failed_ids, [])

    def test_real_repeated_dialogue_is_not_deleted(self):
        words = [Word(0, 1, "待って！"), Word(2, 3, "待って！")]
        self.assertEqual(len(build_cues(words, self.config)), 2)

    def test_cached_failure_report_survives_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"v")
            self.config.set("paths.cache_dir", str(Path(tmp) / "cache"))
            cues = [Cue(0, 0, 1, "帰ろう")]
            cache = _translation_cache_file(video, cues, self.config, None)
            _write_json(cache, {"version": 2, "translations": {"0": "帰ろう"},
                               "report": {"total": 1, "failed_ids": [0], "transliterated_ids": [], "issues": {"0": "실패"}}})
            reports = {}
            with patch("core.pipeline._start_server_with_fallback") as start:
                out = stage_translate([video], {video: cues}, self.config, reports=reports)
            start.assert_not_called()
            self.assertEqual(out[video][0], "帰ろう")
            self.assertEqual(reports[video].status, "일부 미번역")

    def test_running_server_with_different_model_is_not_reused(self):
        server = LlamaServer(Path("server.exe"), Path("expected.gguf"))
        response = type("Response", (), {"raise_for_status": lambda self: None,
                    "json": lambda self: {"model_path": "other.gguf"}})()
        with patch.object(server, "is_alive", return_value=True), patch("core.llm.requests.get", return_value=response):
            with self.assertRaises(LlamaServerError):
                server.start()

    def test_full_reprocess_reextracts_missing_audio_despite_stt_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "video.mp4"
            video.write_bytes(b"video")
            self.config.set("paths.cache_dir", str(Path(tmp) / "cache"))
            self.config.set("pipeline.reprocess", "all")
            _write_json(_stt_cache_file(video, self.config), {"words": []})
            with patch("core.pipeline.extract_audio") as extract:
                stage_audio([video], self.config)
            extract.assert_called_once()

    def test_invalid_kotoba_alignment_is_rejected_before_native_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "kotoba-v2.0"
            model.mkdir()
            (model / "config.json").write_text('{"alignment_heads": [[7, 0]]}')
            with patch("core.stt.WhisperModel") as load:
                with self.assertRaisesRegex(ValueError, "정렬"):
                    STTEngine(model)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
