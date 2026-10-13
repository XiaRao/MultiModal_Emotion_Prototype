"""Train and compare LinearSVC models for multimodal emotion recognition.

Combine TF-IDF text features with the original 26-dimensional audio
features. Compare three C values using the dev set, then save the
model with the highest dev Macro-F1 separately from the baseline.

This file is AI-generated, to check if changing the model to LinearSVC improves the model
performance.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Evaluate C=1.0 first as the default classifier comparison.
C_VALUES = [1.0, 0.1, 10.0]


def load_aligned_data(split):
    """Load original audio features and align their text and labels."""
    csv_path = DATA_DIR / "annotations" / f"{split}_sent_emo.csv"
    features_path = (
        DATA_DIR / "features" / f"{split}_audio_features.npz"
    )

    df = pd.read_csv(csv_path)

    with np.load(features_path) as cached:
        x_audio = cached["X_audio"]
        y_audio = cached["y_audio"]
        row_indices = cached["row_indices"]

    aligned_df = df.iloc[row_indices]

    texts = aligned_df["Utterance"]
    labels = aligned_df["Emotion"].to_numpy()

    if x_audio.ndim != 2 or x_audio.shape[1] != 26:
        raise ValueError(
            f"Expected original 26-dimensional audio features for {split}, "
            f"but got {x_audio.shape}."
        )

    if len(texts) != len(x_audio):
        raise ValueError(f"Text/audio sample counts differ for {split}.")

    if not np.array_equal(labels, y_audio):
        raise ValueError(f"Text/audio labels do not match for {split}.")

    if not np.isfinite(x_audio).all():
        raise ValueError(f"Invalid audio feature values in {split}.")

    print(f"Aligned {split} texts: {texts.shape}")
    print(f"{split.capitalize()} audio: {x_audio.shape}")

    return texts, labels, x_audio


def main():
    print("Loading and aligning data...")
    train_texts, train_labels, train_audio = load_aligned_data("train")
    dev_texts, dev_labels, dev_audio = load_aligned_data("dev")

    print("Label alignment check passed.")

    # Learn the vocabulary and IDF values from train only.
    print("\nCalculating TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )

    x_train_text = vectorizer.fit_transform(train_texts)
    x_dev_text = vectorizer.transform(dev_texts)

    # Learn audio scaling statistics from train only.
    scaler = StandardScaler()
    x_train_audio = scaler.fit_transform(train_audio)
    x_dev_audio = scaler.transform(dev_audio)

    # Keep the original audio weight of 1.0.
    x_train_fused = hstack(
        [x_train_text, x_train_audio],
        format="csr",
    )
    x_dev_fused = hstack(
        [x_dev_text, x_dev_audio],
        format="csr",
    )

    print(f"Fused train shape: {x_train_fused.shape}")
    print(f"Fused dev shape: {x_dev_fused.shape}")

    results = []
    best_model = None
    best_c = None
    best_dev_f1 = -1.0
    best_dev_predictions = None

    for c_value in C_VALUES:
        print(f"\nTraining LinearSVC with C={c_value}...")

        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=10000,
            random_state=0,
        )
        model.fit(x_train_fused, train_labels)

        train_predictions = model.predict(x_train_fused)
        dev_predictions = model.predict(x_dev_fused)

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions,
        )
        train_f1 = f1_score(
            train_labels,
            train_predictions,
            average="macro",
            zero_division=0,
        )

        dev_accuracy = accuracy_score(
            dev_labels,
            dev_predictions,
        )
        dev_f1 = f1_score(
            dev_labels,
            dev_predictions,
            average="macro",
            zero_division=0,
        )

        print(f"Train accuracy: {train_accuracy:.6f}")
        print(f"Train Macro-F1: {train_f1:.6f}")
        print(f"Dev accuracy:   {dev_accuracy:.6f}")
        print(f"Dev Macro-F1:   {dev_f1:.6f}")

        results.append({
            "C": c_value,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "dev_accuracy": dev_accuracy,
            "dev_macro_f1": dev_f1,
        })

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_c = c_value
            best_model = model
            best_dev_predictions = dev_predictions

    results_df = pd.DataFrame(results)

    print("\nComparison:")
    print(results_df.to_string(index=False, float_format="%.6f"))

    print(f"\nSelected C: {best_c}")
    print("Selected model's dev classification report:")
    print(classification_report(
        dev_labels,
        best_dev_predictions,
        digits=3,
        zero_division=0,
    ))

    # Save results and model separately from previous experiments.
    results_dir = DATA_DIR / "results"
    models_dir = DATA_DIR / "models"

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "linear_svc_comparison.csv"
    results_df.to_csv(results_path, index=False)

    model_path = models_dir / "multimodal_model_svc.joblib"

    bundle = {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "model": best_model,
        "audio_weight": 1.0,
        "audio_feature_dim": 26,
        "selected_C": best_c,
    }
    joblib.dump(bundle, model_path)

    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
