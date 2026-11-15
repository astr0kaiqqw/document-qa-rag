"""One-time public model download; no audio or documents are uploaded."""
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    destination = Path(__file__).resolve().parents[1] / 'models/whisper-base'
    snapshot_download('Systran/faster-whisper-base', local_dir=destination)
    print(f'Speech model ready: {destination}')
