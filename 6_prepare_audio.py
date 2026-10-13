"""prepare the dev audio features"""
import pandas as pd
from pathlib import Path
import subprocess

df = pd.read_csv("data/annotations/dev_sent_emo.csv")

video_dir = Path("data/raw/MELD.Raw/dev/dev_splits_complete")
audio_dir = Path("data/audio/dev")
audio_dir.mkdir(parents=True, exist_ok=True)

failed_files = []
for index, row in df.iterrows():
    dia_id = int(row["Dialogue_ID"])
    utt_id = int(row["Utterance_ID"])
    video_path = video_dir / f"dia{dia_id}_utt{utt_id}.mp4"
    audio_path = audio_dir / f"dia{dia_id}_utt{utt_id}.wav"
    if not video_path.exists():
        print(f"Missing video: {video_path.name}")
        continue
    if audio_path.exists():
        print(f"Audio {audio_path.name} already exists")
        continue

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(audio_path)
            ],
            check=True
        )
    except subprocess.CalledProcessError:
        failed_files.append(video_path.name)
        print(f"Conversion failed: {video_path.name}")
        audio_path.unlink(missing_ok=True)
        continue

    print(f"Converted: {audio_path.name}")
    print(failed_files)
