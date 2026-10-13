import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from pathlib import Path

# concatenate the features from text and audio
print("Loading and concatenating data...")
train_audio = np.load("data/features/train_audio_features.npz")
dev_audio = np.load("data/features/dev_audio_features.npz")

train_df = pd.read_csv("data/annotations/train_sent_emo.csv")
dev_df = pd.read_csv("data/annotations/dev_sent_emo.csv")

train_aligned = train_df.iloc[train_audio["row_indices"]]
dev_aligned = dev_df.iloc[dev_audio["row_indices"]]

assert np.array_equal(
    train_aligned["Emotion"].to_numpy(),
    train_audio["y_audio"]
)

assert np.array_equal(
    dev_aligned["Emotion"].to_numpy(),
    dev_audio["y_audio"]
)

print("Aligned train texts:", train_aligned["Utterance"].shape)
print("Train audio:", train_audio["X_audio"].shape)
print("Aligned dev texts:", dev_aligned["Utterance"].shape)
print("Dev audio:", dev_audio["X_audio"].shape)
print("Label alignment check passed.")

print("Calculating TFIDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_text = vectorizer.fit_transform(train_aligned["Utterance"])
X_dev_text = vectorizer.transform(dev_aligned["Utterance"])

scaler = StandardScaler()

X_train_audio_scaled = scaler.fit_transform(train_audio["X_audio"])
X_dev_audio_scaled = scaler.transform(dev_audio["X_audio"])

X_train_fused = hstack(
    [X_train_text, X_train_audio_scaled], format="csr"
)

X_dev_fused = hstack(
    [X_dev_text, X_dev_audio_scaled], format="csr"
)

print("Fused train shape:", X_train_fused.shape)
print("Fused dev shape:", X_dev_fused.shape)

#start training logistic regression model
print("Start training...")
y_train = train_audio["y_audio"]
y_dev = dev_audio["y_audio"]

multimodal_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
multimodal_model.fit(X_train_fused, y_train)

dev_pred = multimodal_model.predict(X_dev_fused)
print("Dev accuracy:", accuracy_score(y_dev, dev_pred))
print("Dev F1:", f1_score(y_dev, dev_pred, average="macro"))
print(classification_report(
    y_dev, dev_pred, digits=3, zero_division=0
))

# add text reference
text_reference = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

text_reference.fit(X_train_text, y_train)
text_dev_pred = text_reference.predict(X_dev_text)

print("Text-only dev accuracy:", accuracy_score(y_dev, text_dev_pred))
print("Text-only dev Macro-F1:",
      f1_score(y_dev, text_dev_pred, average="macro"))

# save vectorizer, scaler, and model to further make predictions
model_dir = Path("data/models")
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "model": multimodal_model,
    },
    model_dir / "multimodal_model.joblib"
)
print("saved to data/models/multimodal_model.joblib")
