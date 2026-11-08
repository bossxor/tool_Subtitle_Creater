"""Fixed 40.8s literary sample, batch vs sequential. Orthographic CER is not general accuracy.

Reference: Osamu Dazai, public-domain グッド・バイ, first paragraphs (modernized spelling).
https://www.aozora.gr.jp/cards/000035/files/258_20179.html
The sample contains no source text in its recognition prompts.
"""
import json
import argparse
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import Config
from core.stt import STTEngine

REFERENCE = (
    "文壇のある老大家が亡くなってその告別式の終わり頃から雨が降り始めた早春の雨である"
    "その帰り二人の男が相合傘で歩いているいずれもその逝去した老大家にはお義理一ぺん"
    "話題は女についての極めて不謹慎な事紋服の初老の大男は文士それよりずっと若いロイド眼鏡"
    "縞ズボンの好男子は編集者あいつもと文士は言う女が好きだったらしいな"
    "お前もそろそろ年貢の納め時じゃねえのかやつれたぜ全部やめるつもりでいるんです"
)


def normalized(text):
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text))


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [i]
        for j, b in enumerate(right, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["large", "kotoba", "turbo"], default="large")
    args = parser.parse_args()
    assert edit_distance("abc", "adc") == 1 and edit_distance("", "abc") == 3
    config = Config.load()
    output = Path(__file__).resolve().parent / "out"
    clip = output / "stt_reference_40s.wav"
    with wave.open(str(output.parent / "sample" / "sample_10min.wav"), "rb") as stream:
        rate = stream.getframerate()
        stream.setpos(int(20 * rate))
        with wave.open(str(clip), "wb") as target:
            target.setparams(stream.getparams())
            target.writeframes(stream.readframes(int(40.8 * rate)))
    engine = STTEngine(config.resolve_path("stt.models." + args.model), device="cpu", compute_type="int8")
    records = []
    for mode, runner, beam in [("batch", engine.pipeline, 1), ("sequential", engine.model, 5)]:
        started = time.monotonic()
        kwargs = dict(language="ja", task="transcribe", beam_size=beam, word_timestamps=True,
                      condition_on_previous_text=mode == "sequential", vad_filter=True,
                      vad_parameters={"min_silence_duration_ms": 500})
        if mode == "batch":
            kwargs["batch_size"] = 16
        segments, _ = runner.transcribe(str(clip), **kwargs)
        segments = list(segments)
        text = "".join(s.text for s in segments)
        reference = normalized(REFERENCE)
        record = {"model": args.model, "mode": mode, "device": "cpu", "compute_type": "int8", "beam": beam,
                  "seconds": round(time.monotonic() - started, 2), "text": text,
                  "orthographic_cer": round(edit_distance(reference, normalized(text)) / len(reference), 4),
                  "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (output / f"stt_quality_comparison.{args.model}.json").write_text(json.dumps({"reference": REFERENCE, "reference_source": __doc__, "results": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
