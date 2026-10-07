import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
segs = json.loads((ROOT / "bench" / "out" / "large-v3_b16_beam1_float16.json").read_text(encoding="utf-8"))

# large-v3 segments are long (one per ~30s chunk); split into sentence-ish units on Japanese punctuation
import re

items = []
for s in segs:
    parts = re.split(r"(?<=[。！？])", s["text"])
    for p in parts:
        p = p.strip()
        if p:
            items.append(p)

items = items[:20]  # first 20 sentences for the test
numbered = [{"id": i, "ja": t} for i, t in enumerate(items)]

SYSTEM = (
    "당신은 일본어 영상 자막을 한국어로 옮기는 전문 번역가입니다. "
    "입력은 JSON 배열이며 각 항목은 {id, ja} 형식입니다. "
    "각 문장을 자연스러운 한국어 구어체 대사로 번역하고, 어색한 음성인식 오류(동음이의어 오taja 등)가 의심되면 "
    "문맥에 맞게 자연스럽게 다듬으세요. 존댓말/반말은 문맥의 어조를 따르세요. "
    "출력은 반드시 JSON 배열만 반환하고, 각 항목은 {id, ko} 형식이어야 합니다. 입력과 같은 개수, 같은 id 순서를 유지하세요. "
    "설명이나 코드블록 없이 순수 JSON만 출력하세요."
)
USER = json.dumps(numbered, ensure_ascii=False)

t0 = time.time()
r = requests.post(
    "http://127.0.0.1:8090/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "temperature": 0.3,
        "max_tokens": 2000,
    },
    timeout=300,
)
dt = time.time() - t0
data = r.json()
content = data["choices"][0]["message"]["content"]
usage = data.get("usage", {})

print(f"elapsed={dt:.1f}s usage={usage}")
print("--- raw output ---")
print(content)

out_path = ROOT / "bench" / "out" / "translate_test.json"
out_path.write_text(content, encoding="utf-8")

try:
    parsed = json.loads(content)
    print(f"\nparsed OK: {len(parsed)} items (expected {len(numbered)})")
    print("\n--- side by side ---")
    for src, tr in zip(numbered, parsed):
        print(f"[{src['id']}] JA: {src['ja']}")
        print(f"     KO: {tr.get('ko')}")
except Exception as e:
    print("PARSE FAILED:", e)
