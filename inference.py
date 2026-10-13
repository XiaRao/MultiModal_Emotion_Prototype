"""Predict utterance-level emotion from text and audio.

Load the saved TF-IDF vectorizer, audio scaler, and multimodal classifier.
Provide predict_emotion(text, audio_path) to extract and combine text/audio
features, then return an emotion prediction and a template-based response.

When run directly, demonstrate prediction on one dev utterance and print
its ground-truth label for comparison.
"""
from audio_features import extract_audio_features
import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack

bundle = joblib.load("data/models/multimodal_model.joblib")
vectorizer = bundle["vectorizer"]
scaler = bundle["scaler"]
model = bundle["model"]


def predict_emotion(text, audio_path):
    """Predict emotion from utterance text and its corresponding audio file."""
    # Transform text using the fitted TF-IDF vectorizer.
    x_text = vectorizer.transform([text])

    # Extract and standardize audio features.
    audio_features = extract_audio_features(audio_path)
    x_audio = scaler.transform(audio_features.reshape(1, -1))

    # Combine text and audio features.
    x_fused = hstack([x_text, x_audio], format="csr")

    # Predict the emotion for this utterance.
    emotion = str(model.predict(x_fused)[0])

    return {
        "emotion": emotion,
        "response": RESPONSE_TEMPLATES[emotion]
    }


RESPONSE_TEMPLATES = {
    "anger": "Would you like to tell me what bothered you?",
    "disgust": "What felt unpleasant about that?",
    "fear": "Would you like to talk through what's worrying you?",
    "joy": "That sounds so exciting! Glad for you!",
    "neutral": "Tell me more.",
    "sadness": "Hey, what happened? Believe me, it's gonna be alright.",
    "surprise": "Woa! Tell me more about it!"
}

if __name__ == "__main__":
    dev_df = pd.read_csv("data/annotations/dev_sent_emo.csv")
    row = dev_df.iloc[0]

    text = row["Utterance"]
    dia_id = int(row["Dialogue_ID"])
    utt_id = int(row["Utterance_ID"])

    audio_path = (
        Path("data/audio/dev")
        / f"dia{dia_id}_utt{utt_id}.wav"
    )

    result = predict_emotion(text, audio_path)

    print("Text:", text)
    print("Audio:", audio_path)
    print("Prediction:", result["emotion"])
    print("Response:", result["response"])
    print("Ground-truth emotion:", row["Emotion"])
