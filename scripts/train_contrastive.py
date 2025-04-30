
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# ===== 编码器定义 =====
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

# ===== 对比损失函数 =====
def contrastive_loss(z1, z2, temperature=0.5):
    sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
    labels = torch.arange(z1.size(0)).to(z1.device)
    logits = sim / temperature
    return F.cross_entropy(logits, labels)

# ===== 数据集定义 =====
class ContrastiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = np.array(y)

        # 正类 vs 负类划分
        self.pos_idx = np.where(self.y == 1)[0]
        self.neg_idx = np.where(self.y == 0)[0]

    def __len__(self):
        return min(len(self.pos_idx), len(self.neg_idx))

    def __getitem__(self, idx):
        x1 = self.X[self.pos_idx[idx]]
        x2 = self.X[self.neg_idx[idx]]
        return x1, x2

# ===== 数据准备 =====
df = pd.read_csv("data/processed/cleaned_data.csv")
target_col = "7d内术后肺部并发症(0=否，1=是)"
df = df.dropna(subset=[target_col])
df = df.fillna(df.median(numeric_only=True))
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练数据
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
dataset = ContrastiveDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ===== 模型训练 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ContrastiveEncoder(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for x1, x2 in loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1), model(x2)
        loss = contrastive_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Contrastive Loss: {total_loss / len(loader):.4f}")

# 可选：保存编码器
torch.save(model.state_dict(), "outputs/models/contrastive_encoder.pt")
print("对比学习编码器已保存至 outputs/models/contrastive_encoder.pt")
