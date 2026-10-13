import pandas as pd
from pathlib import Path
import numpy as np
from audio_features import extract_segmented_audio_features

feature_dir = Path("data/features")
feature_dir.mkdir(parents=True, exist_ok=True)

# process train, dev, and test separately to prepare the segmented features.
for split in ("train", "dev", "test"):
    df = pd.read_csv(f"data/annotations/{split}_sent_emo.csv")
    audio_dir = Path("data/audio") / split
    feature_list = []
    label_list = []
    row_index_list = []

    missing_files = []
    failed_files = []

    print(f"\nProcessing {split}: {len(df)} CSV rows")

    for count, (index, row) in enumerate(df.iterrows(), start=1):
        dia_id = int(row["Dialogue_ID"])
        utt_id = int(row["Utterance_ID"])
        # find the path for the current audio file
        audio_path = audio_dir / f"dia{dia_id}_utt{utt_id}.wav"

        if not audio_path.is_file():
            missing_files.append(audio_path.name)
            print(f"Missing file: {audio_path.name}")
            continue

        try:
            features = extract_segmented_audio_features(audio_path)

            if (features.shape != (78,)):
                raise ValueError(f"Expected shape is (78,), received {features.shape}")

            if not np.isfinite(features).all():
                raise ValueError("Features contain NaN values or infinity.")

        except (OSError, RuntimeError, ValueError) as error:
            failed_files.append(audio_path.name)
            print(f"Failed file: {audio_path.name}: {error}")
            continue

        # append the three records only after successful extraction
        feature_list.append(features)
        label_list.append(row["Emotion"])
        row_index_list.append(index)

        if count % 500 == 0:
            print(f"Processed through CSV row {count}")

    if not feature_list:
        raise RuntimeError(f"No usable audio features found for {split}")

    X_audio = np.stack(feature_list, axis=0)
    y_audio = np.array(label_list)
    row_indices = np.array(row_index_list, dtype=int)

    save_path = feature_dir / f"{split}_audio_feature_3seg.npz"

    # saves the data as 3 new files corresponding to the 3 datasets. train/dev/test
    np.savez(
        save_path,
        X_audio=X_audio,
        y_audio=y_audio,
        row_indices=row_indices,
    )
    print(f"X_audio shape: {X_audio.shape}")
