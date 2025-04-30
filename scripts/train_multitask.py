
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ===== 多任务模型定义 =====
class MultiTaskNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.task1 = nn.Linear(hidden_dim, 1)  # 再插管
        self.task2 = nn.Linear(hidden_dim, 1)  # 呼吸衰竭
        self.task3 = nn.Linear(hidden_dim, 1)  # 肺部并发症

    def forward(self, x):
        shared_out = self.shared(x)
        return {
            "reintubation": torch.sigmoid(self.task1(shared_out)),
            "resp_failure": torch.sigmoid(self.task2(shared_out)),
            "pulm_comp": torch.sigmoid(self.task3(shared_out))
        }

# ===== 数据集封装 =====
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_dict):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_dict = {k: torch.tensor(v.values, dtype=torch.float32).unsqueeze(1)
                       for k, v in y_dict.items()}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {"features": self.X[idx]}
        for k in self.y_dict:
            sample[k] = self.y_dict[k][idx]
        return sample

# ===== 数据加载与预处理 =====
df = pd.read_csv("data/processed/cleaned_data.csv")

target_cols = {
    "reintubation": "再插管（0=否，1=是）",
    "resp_failure": "呼吸衰竭(0=否，1=是)",
    "pulm_comp": "7d内术后肺部并发症(0=否，1=是)"
}

df = df.dropna(subset=target_cols.values())
df = df.fillna(df.median(numeric_only=True))
y_dict = {k: df[v].astype(int) for k, v in target_cols.items()}
X = df.drop(columns=list(target_cols.values()))
X = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据
X_train, X_test, y_train_dict, y_test_dict = train_test_split(
    X_scaled, y_dict, test_size=0.2, stratify=y_dict["pulm_comp"], random_state=42
)
train_dataset = MultiTaskDataset(X_train, {k: v.iloc[y_train_dict[k].index] for k, v in y_train_dict.items()})
test_dataset = MultiTaskDataset(X_test, {k: v.iloc[y_test_dict[k].index] for k, v in y_test_dict.items()})
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# ===== 模型训练 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskNN(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(1, 21):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch["features"].to(device)
        outputs = model(inputs)
        loss = sum(criterion(outputs[k], batch[k].to(device)) for k in target_cols)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ===== 模型评估 =====
model.eval()
all_preds, all_trues = {k: [] for k in target_cols}, {k: [] for k in target_cols}
with torch.no_grad():
    for batch in test_loader:
        inputs = batch["features"].to(device)
        outputs = model(inputs)
        for k in target_cols:
            all_preds[k].extend(outputs[k].cpu().numpy().ravel())
            all_trues[k].extend(batch[k].numpy().ravel())

print("\nEvaluation Metrics:")
for k in target_cols:
    y_true = np.array(all_trues[k])
    y_pred = (np.array(all_preds[k]) >= 0.5).astype(int)
    y_score = np.array(all_preds[k])
    print(f"{k} - AUROC: {roc_auc_score(y_true, y_score):.4f}, F1: {f1_score(y_true, y_pred):.4f}, Acc: {accuracy_score(y_true, y_pred):.4f}")
