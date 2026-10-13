"""inspect the trian audio features"""
import pandas as pd
import numpy as np
from pathlib import Path
from audio_features import extract_audio_features

train_df = pd.read_csv("data/annotations/train_sent_emo.csv")
feature_lst = []
label_lst = []
row_index_lst = []

audio_dir = Path("data/audio/train")

for index, row in train_df.iterrows():
    dia_id = int(row["Dialogue_ID"])
    utt_id = int(row["Utterance_ID"])
    audio_path = audio_dir / f"dia{dia_id}_utt{utt_id}.wav"

    if not audio_path.exists():
        print(f"Missing audio: {audio_path.name}")
        continue

    feature = extract_audio_features(audio_path)
    feature_lst.append(feature)
    label_lst.append(row["Emotion"])
    row_index_lst.append(index)

X_audio = np.stack(feature_lst, axis=0)
y_audio = np.array(label_lst)
feature_dir = Path("data/features")
feature_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    feature_dir / "train_audio_features.npz",
    X_audio=X_audio,
    y_audio=y_audio,
    row_indices=np.array(row_index_lst)
)

print(f"X_audio shape: {X_audio.shape}")
print(f"y_audio shape: {y_audio.shape}")
print(f"saved samples: {len(row_index_lst)}")
print("saved to data/features/train_audio_features.npz")
print(feature_lst[0])
print(label_lst[0])
