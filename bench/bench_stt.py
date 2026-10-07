import argparse
import json
import os
import site
import time
from pathlib import Path

for sp in site.getsitepackages():
    for b in Path(sp, "nvidia").glob("*/bin"):
        os.add_dll_directory(str(b))
        os.environ["PATH"] = str(b) + os.pathsep + os.environ["PATH"]

from faster_whisper import BatchedInferencePipeline, WhisperModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "out"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--audio", default=str(ROOT / "bench" / "sample" / "sample_10min.wav"))
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--beam", type=int, default=1)
ap.add_argument("--compute", default="float16")
ap.add_argument("--nobatch", action="store_true")
args = ap.parse_args()

t0 = time.time()
model = WhisperModel(
    str(ROOT / "models" / "whisper" / args.model),
    device="cuda",
    compute_type=args.compute,
)
load_s = time.time() - t0

kw = dict(
    language="ja",
    beam_size=args.beam,
    condition_on_previous_text=False,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
)
runner = model if args.nobatch else BatchedInferencePipeline(model)
if not args.nobatch:
    kw["batch_size"] = args.batch

t0 = time.time()
segs, info = runner.transcribe(args.audio, **kw)
segs = [dict(start=s.start, end=s.end, text=s.text.strip()) for s in segs]
run_s = time.time() - t0

tag = f"{args.model}_{'seq' if args.nobatch else 'b' + str(args.batch)}_beam{args.beam}_{args.compute}"
(OUT / f"{tag}.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
print(
    f"{tag}: load={load_s:.1f}s transcribe={run_s:.1f}s "
    f"audio={info.duration:.0f}s RTF={run_s / info.duration:.3f} "
    f"({info.duration / run_s:.1f}x realtime) segs={len(segs)}"
)
