"""This file inspects the data and gives overall information of the data."""

import pandas as pd
import numpy as np
import librosa

if __name__ == "__main__":
    train_df = pd.read_csv('data/annotations/train_sent_emo.csv')
    print(train_df.shape)
    print(train_df.columns)
    print(train_df["Emotion"].value_counts())
    mask = (train_df["Dialogue_ID"] == 0) & (train_df["Utterance_ID"] == 0)
    filtered_df = train_df[mask]
    new_filtered_df = filtered_df[["Utterance", "Emotion", "Dialogue_ID", "Utterance_ID"]]
    print(new_filtered_df.to_string(index=False))

    y, sr = librosa.load(
        "data/audio/train/dia0_utt0.wav",
        sr=None,
        mono=True
    )
    print(f"shape of y: {y.shape}")
    # Sample rate: how many samples per second. since 16000Hz, should be 16000.
    print(f"sample rate: {sr}")

    print(f"Duration: {len(y)/sr}")
    print(y[:10])

    print("Calculating MFCC...")
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        center=False
    )
    print(f"MFCC shape: {mfcc.shape}")
    mean_coe = np.mean(mfcc, axis=1)
    std_coe = np.std(mfcc, axis=1)
    coe = np.concatenate((mean_coe, std_coe))
    print(f"final vector shape: {coe.shape}")
    print(f"final vector: {coe}")
