"""Audio Feature Extraction"""

import numpy as np
import librosa


def extract_audio_features(audio_path):
    """Extract MFCC features as 26 features from audio_path"""
    # load the audio
    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )
    # calculate mfcc
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        center=False
    )

    mean_coe = np.mean(mfcc, axis=1)
    std_coe = np.std(mfcc, axis=1)
    feature = np.concatenate((mean_coe, std_coe))
    return feature


if __name__ == "__main__":
    features = extract_audio_features('data/audio/train/dia0_utt0.wav')
    print(f"Feature shape: {features.shape}")
    print(f"First five features: {features[:5]}")
