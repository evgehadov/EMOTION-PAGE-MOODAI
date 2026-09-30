import pickle

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from evaluate import EmotionMLP

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def main():
    with open("all_data.pkl", "rb") as f:
        data = pickle.load(f)

    X = np.vstack(data["embedengs"].values).astype(np.float32)
    y = np.vstack(data["emotion_onehot"].values).astype(np.float32)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = EmotionMLP()
    model.load_state_dict(torch.load("mlp_multilabel_emotions.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_val))).numpy()

    print(f"{'порог':>6} {'micro-F1':>9} {'macro-F1':>9}")
    for threshold in THRESHOLDS:
        predicted = (probs >= threshold).astype(int)
        micro = f1_score(y_val, predicted, average="micro", zero_division=0)
        macro = f1_score(y_val, predicted, average="macro", zero_division=0)
        print(f"{threshold:>6.2f} {micro:>9.3f} {macro:>9.3f}")


if __name__ == "__main__":
    main()