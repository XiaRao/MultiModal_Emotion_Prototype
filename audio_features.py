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


# Divide the MFCC into 3 parts to maintain some level of time sequences.
# the returned value consists of
# 0:13, the means of the first segment. 13:26, the stds of the first segment
# 25:39, the means of the second segment. 39:52, the stds of the second segment
# 52:65, the means of the third segment. 65:78, the stds of the third segment
def extract_segmented_audio_features(audio_path):
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

    if mfcc.shape[1] < 3:
        raise ValueError("Audio has fewer than 3 MFCC frames.")

    segments = np.array_split(mfcc, 3, axis=1)

    feature_parts = []
    for segment in segments:
        feature_parts.append(segment.mean(axis=1))
        feature_parts.append(segment.std(axis=1))

    return np.concatenate(feature_parts)


if __name__ == "__main__":
    features = extract_segmented_audio_features('data/audio/train/dia0_utt0.wav')
    print(f"Feature shape: {features.shape}")
    print("All values finite:", np.isfinite(features).all())
