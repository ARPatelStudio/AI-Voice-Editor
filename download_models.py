import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Pre-quantized ONNX models hosted on Hugging Face Hub mirrors
MODELS = {
    "speaker_encoder.onnx": (
        "https://huggingface.co/pyannote/embedding/resolve/main/pytorch_model.bin"  # Replace with direct ONNX release URL
    ),
    "speaker_extractor.onnx": (
        "https://huggingface.co/alibabaspeech/ClearerVoice-Studio/resolve/main/models/tse_model.onnx"
    ),
}

def download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Already exists: {dest_path.name}")
        return
    print(f"Downloading {dest_path.name}...")
    # Add streaming download with browser-like user agent
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded: {dest_path.name} ({round(dest_path.stat().st_size / (1024 * 1024), 2)} MB)")

if __name__ == "__main__":
    for filename, url in MODELS.items():
        target = MODELS_DIR / filename
        try:
            download_file(url, target)
        except Exception as e:
            print(f"Failed downloading {filename}: {e}")
