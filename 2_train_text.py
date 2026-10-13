"""Train text model. Prepare the training text and the emotion labels."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


train_df = pd.read_csv("data/annotations/train_sent_emo.csv")
dev_df = pd.read_csv("data/annotations/dev_sent_emo.csv")

train_texts = train_df["Utterance"]
train_labels = train_df["Emotion"]
dev_texts = dev_df["Utterance"]
dev_labels = dev_df["Emotion"]

print(f"Shape of train_texts: {train_texts.shape}")
print(f"Shape of train_labels: {train_labels.shape}")
print(f"Shape of dev_texts: {dev_texts.shape}")
print(f"Shape of dev_labels: {dev_labels.shape}")

print(f"Sample first train_text: {train_texts.iloc[0]}")
print(f"Sample first train_label: {train_labels.iloc[0]}")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

print("Starting TF-IDF...")
X_train_texts = vectorizer.fit_transform(train_texts)
X_dev_texts = vectorizer.transform(dev_texts)
print(f"Shape of X_train_texts: {X_train_texts.shape}")
print(f"Shape of X_dev_texts: {X_dev_texts.shape}")
vectorizer.get_feature_names_out()
vectorizer.get_feature_names_out()

print("Starting Logistic Regression...")
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_texts, train_labels)
dev_pred = model.predict(X_dev_texts)
print(f"dev_pred shape: {dev_pred.shape}")
print(f"dev_pred first 10 predictions: {dev_pred[:10]}")
accuracy = accuracy_score(dev_labels, dev_pred)
f1 = f1_score(dev_labels, dev_pred, average="macro")
report = classification_report(dev_labels, dev_pred, digits=3, zero_division=0)
print(f"Accuracy: {accuracy}")
print(f"F1: {f1}")
print(f"Report: {report}")
