
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch.nn.functional as F

# ===== 对比学习编码器 =====
class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)

# ===== 下游分类器模型 =====
class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return self.classifier(z)

# ===== 数据加载 =====
df = pd.read_csv("data/processed/cleaned_data.csv")
target_col = "7d内术后肺部并发症(0=否，1=是)"
df = df.dropna(subset=[target_col])
df = df.fillna(df.median(numeric_only=True))
y = df[target_col].astype(int).values
X = df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# ===== 模型加载与训练 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = ContrastiveEncoder(input_dim=X.shape[1]).to(device)
encoder.load_state_dict(torch.load("outputs/models/contrastive_encoder.pt", map_location=device))

model = DownstreamClassifier(encoder, hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# 训练分类器部分
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f}")

# ===== 测试评估 =====
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor.to(device)).cpu().numpy().ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"Test AUROC: {auc:.4f} | F1-score: {f1:.4f} | Accuracy: {acc:.4f}")
