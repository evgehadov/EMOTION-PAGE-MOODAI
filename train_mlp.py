import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

with open("all_data.pkl", "rb") as f:
    data = pickle.load(f)

X = np.vstack(data["embedengs"].values).astype(np.float32)
y = np.vstack(data["emotion_onehot"].values).astype(np.float32)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = EmotionDataset(X_train, y_train)
val_dataset = EmotionDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionMLP(input_dim=768).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    val_loss = 0
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)

            preds = torch.sigmoid(outputs)
            y_pred_list.append(preds.cpu().numpy())
            y_true_list.append(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    y_true_arr = np.vstack(y_true_list)
    y_pred_arr = np.vstack(y_pred_list)
    y_pred_bin = (y_pred_arr >= 0.5).astype(int)

    print(f"Epoch {epoch}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    acc = (y_pred_bin == y_true_arr).mean()
    print(f"Val accuracy (average over labels): {acc:.4f}")

    print(classification_report(y_true_arr[:, :5], y_pred_bin[:, :5], zero_division=0))

    print("-" * 60)

torch.save(model.state_dict(), "mlp_multilabel_emotions.pth")
print("✅ Модель сохранена: mlp_multilabel_emotions.pth")
