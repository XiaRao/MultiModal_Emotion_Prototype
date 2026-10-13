import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score, classification_report

# read test and saved model
test_df = pd.read_csv("data/annotations/test_sent_emo.csv")
test_audio = np.load("data/features/test_audio_features.npz")

bundle = joblib.load("data/models/multimodal_model.joblib")
vectorizer = bundle["vectorizer"]
scaler = bundle["scaler"]
model = bundle["model"]

# get text, audio and labels
test_aligned = test_df.iloc[test_audio["row_indices"]]
y_test = test_audio["y_audio"]

assert np.array_equal(
    test_aligned["Emotion"].to_numpy(),
    y_test
)

X_test_text = vectorizer.transform(test_aligned["Utterance"])
X_test_audio = scaler.transform(test_audio["X_audio"])

X_test_fused = hstack(
    [X_test_text, X_test_audio], format="csr"
)

# make predictions and evaluate
test_pred = model.predict(X_test_fused)

print("Total test rows:", len(test_df))
print("Evaluated test rows:", len(y_test))
print("Fused test shape:", X_test_fused.shape)

print("Test accuracy:", accuracy_score(y_test, test_pred))
print("Test Macro-F1:", f1_score(y_test, test_pred, average="macro"))
print(classification_report(
    y_test, test_pred, digits=3, zero_division=0
))
