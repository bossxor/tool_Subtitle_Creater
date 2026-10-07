"""CLI 진입점. 사용 예:

python -m core.cli --videos movie1.mp4 movie2.mp4 --out-dir D:\subs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from core.config import Config, DEFAULT_CONFIG_PATH
from core.pipeline import run_batch


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="영상에서 자막(srt/smi)을 만드는 로컬 파이프라인")
    p.add_argument("--videos", nargs="+", required=True, help="영상 파일 경로 (여러 개 가능)")
    p.add_argument("--out-dir", required=True, help="자막을 저장할 폴더")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config.yaml 경로")
    p.add_argument("--precision", choices=["turbo", "large", "kotoba"], help="STT 정밀도 (기본: config.yaml 값)")
    p.add_argument("--source-lang", help="원어 코드 (기본: config.yaml 값, 예: ja)")
    p.add_argument("--target-lang", help="번역 대상 언어 코드 (기본: config.yaml 값, 예: ko)")
    p.add_argument("--glossary", help="용어집 JSON 파일 경로 ({'원어단어': '번역'} 형식)")
    p.add_argument("--emit-source", action="store_true", help="원문 자막도 함께 생성")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config.load(args.config)
    if args.precision:
        config.set("stt.precision", args.precision)
    if args.source_lang:
        config.set("source_language", args.source_lang)
    if args.target_lang:
        config.set("target_language", args.target_lang)
    if args.emit_source:
        config.set("subtitle.emit_source", True)

    glossary = None
    if args.glossary:
        glossary = json.loads(Path(args.glossary).read_text(encoding="utf-8"))

    missing = [v for v in args.videos if not Path(v).exists()]
    if missing:
        logging.error("파일을 찾을 수 없음: %s", missing)
        return 1

    outputs = run_batch(args.videos, args.out_dir, config, glossary=glossary)

    print("\n=== 완료 ===")
    for video, paths in outputs.items():
        print(f"{video}:")
        for p in paths:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
