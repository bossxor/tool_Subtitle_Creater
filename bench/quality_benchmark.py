"""Compare installed TranslateGemma with/without neighboring dialogue; not an accuracy score.

Run: .venv/Scripts/python.exe bench/quality_benchmark.py
Synthetic dialogue and checks are fixed before inference. Review semantic differences manually.
"""
import json
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import Config
from core.llm import LlamaServer
from core.segmenter import Cue
from core.translate_gemma import translate_cues_gemma

CASES = [
    ("ただいま。", "귀가 인사, 현재 시각 설명 아님"),
    ("おかえり。", "귀가한 상대에게 답하는 인사"),
    ("遅かったね。", "친근한 말투, 상대가 늦게 왔음"),
    ("電車が止まっちゃってさ。", "전철이 멈췄다는 이유, 대화체"),
    ("ご飯できてるよ。先に食べる？", "밥 준비됨, 먼저 먹을지 질문, 친근한 말투"),
    ("うん、ありがとう。", "감사, 발음 음역 아님"),
    ("課長はもういらっしゃいますか？", "상사 도착 여부, 정중함 유지"),
    ("ええ、朝から会議中です。", "아침부터 회의 중, 정중함 유지"),
    ("彼には頭が上がらない。", "상대에게 신세·위축, 존경 때문에 말을 못한다는 추가 설정 금지"),
    ("DVDを見せてください。", "DVD 보존, 정중한 요청"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU when other GPU workloads are idle")
    args = parser.parse_args()
    config = Config.load()
    config.set("llm.max_retries", 1)
    config.set("llm.translate_parallel", 1)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    server = LlamaServer(config.resolve_path("llm.server_exe"), config.resolve_path("llm.translate_model"),
                         port=18094, n_gpu_layers=None if args.gpu else 0, ctx_size=2048,
                         extra_args=[*config.get("llm.translate_extra_args"), "-np", "2"], log_path=out_dir / "quality_comparison.log")
    cues = [Cue(i, i * 2 + (20 if i >= 6 else 0), i * 2 + 1.8 + (20 if i >= 6 else 0), text)
            for i, (text, _) in enumerate(CASES)]
    results = {"kind": "synthetic dialogue, manual semantic review required", "device": "gpu" if args.gpu else "cpu", "criteria": [c[1] for c in CASES]}
    try:
        server.start(timeout=60)
        for context in (0, 2):
            config.set("llm.context_sentences", context)
            started = time.monotonic()
            output, report = translate_cues_gemma(cues, server, config)
            rows = [{"source": c.text, "translation": output[c.id]} for c in cues]
            results[str(context)] = {"seconds": round(time.monotonic() - started, 2), "rows": rows, "report": asdict(report)}
            print(json.dumps({"context": context, **results[str(context)]}, ensure_ascii=False), flush=True)
    finally:
        server.stop()
    (out_dir / "quality_comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
