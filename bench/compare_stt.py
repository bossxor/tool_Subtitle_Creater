import difflib
import json
from pathlib import Path

o = Path(__file__).resolve().parent / "out"
names = {
    "kotoba": "kotoba-v2.0_b16_beam1_float16",
    "turbo": "large-v3-turbo_b16_beam1_float16",
    "large": "large-v3_b16_beam1_float16",
}
segs = {k: json.loads((o / f"{v}.json").read_text(encoding="utf-8")) for k, v in names.items()}
T = {k: "".join(s["text"] for s in v) for k, v in segs.items()}

for k, v in T.items():
    print(k, "chars:", len(v), "segments:", len(segs[k]))
print()


def diffrate(a, b):
    return 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


for a, b in [("kotoba", "large"), ("turbo", "large"), ("kotoba", "turbo")]:
    print(f"diff {a} vs {b}: {diffrate(T[a], T[b]) * 100:.2f}%")

print("\n--- segment-by-segment first 8 ---")
for i in range(min(8, len(segs["large"]))):
    print(f"\n[{i}] {segs['large'][i]['start']:.1f}-{segs['large'][i]['end']:.1f}s")
    for k in ("large", "turbo", "kotoba"):
        t = segs[k][i]["text"] if i < len(segs[k]) else "(none)"
        print(f"  {k:7s}: {t}")
