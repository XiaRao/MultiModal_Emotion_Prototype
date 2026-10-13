import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

train_data = np.load("data/features/train_audio_features.npz")
dev_data = np.load("data/features/dev_audio_features.npz")

X_train = train_data["X_audio"]
y_train = train_data["y_audio"]
X_dev = dev_data["X_audio"]
y_dev = dev_data["y_audio"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)

model.fit(X_train, y_train)
y_pred = model.predict(X_dev)

print("Dev accuracy:", accuracy_score(y_dev, y_pred))
print("Dev F1:", f1_score(y_dev, y_pred, average="macro"))
print(classification_report(y_dev, y_pred, zero_division=0))
