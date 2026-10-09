"""下载固定版本的公开向量模型；不传输用户资料。"""

from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7999e1d3359715c523056ef9478215996d62a620"

if __name__ == "__main__":
    destination = ROOT / "data" / "models" / "bge-small-zh-v1.5"
    snapshot_download(
        "BAAI/bge-small-zh-v1.5",
        revision=REVISION,
        local_dir=str(destination),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
    )
    print(f"模型固定版本已下载：{REVISION}")
    print(destination)
