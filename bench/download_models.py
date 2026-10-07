from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent / "models"

WHISPER = {
    "kotoba-v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

for name, repo in WHISPER.items():
    print(f"[whisper] {name}", flush=True)
    snapshot_download(repo, local_dir=ROOT / "whisper" / name)

print("[llm] Qwen3-8B Q4_K_M", flush=True)
hf_hub_download(
    "Qwen/Qwen3-8B-GGUF",
    "Qwen3-8B-Q4_K_M.gguf",
    local_dir=ROOT / "llm",
)
print("DONE", flush=True)
