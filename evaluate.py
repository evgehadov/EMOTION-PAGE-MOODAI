import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class EmotionMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=28):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def main():
    with open("all_data.pkl", "rb") as f:
        data = pickle.load(f)

    X = np.vstack(data["embedengs"].values).astype(np.float32)
    y = np.vstack(data["emotion_onehot"].values).astype(np.float32)

    # То же разбиение, что и при обучении: 20% данных модель не видела.
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = EmotionMLP()
    model.load_state_dict(torch.load("mlp_multilabel_emotions.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_val))).numpy()
    predicted = (probs >= 0.3).astype(int)

    micro = f1_score(y_val, predicted, average="micro", zero_division=0)
    macro = f1_score(y_val, predicted, average="macro", zero_division=0)

    print(f"Примеров в проверке: {len(y_val)}")
    print(f"micro-F1: {micro:.3f}")
    print(f"macro-F1: {macro:.3f}")


if __name__ == "__main__":
    main()