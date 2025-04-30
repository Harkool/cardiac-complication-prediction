
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === 参数 ===
DATA_PATH = "data/processed/cleaned_data.csv"  # 替换为实际路径
LABEL_COL = "7d内术后肺部并发症(0=否，1=是)"
SAVE_MODEL_PATH = "outputs/models/gbc_model.pkl"

# === 读取数据 ===
df = pd.read_csv(DATA_PATH)

# 简单预处理：去除含ID、姓名、备注类字段
drop_cols = [col for col in df.columns if any(k in col for k in ["姓名", "住院号", "备注"])]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# 丢弃目标缺失
df = df.dropna(subset=[LABEL_COL])

# 填补缺失
df = df.fillna(df.median(numeric_only=True))

# 提取特征与标签
y = df[LABEL_COL].astype(int)
X = df.drop(columns=[LABEL_COL])

X_numeric = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# 划分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === 模型训练 ===
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# === 模型评估 ===
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auroc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
auprc = auc(recall, precision)

# === 输出结果 ===
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")

# === 保存模型 ===
os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
joblib.dump(model, SAVE_MODEL_PATH)
print(f"Model saved to {SAVE_MODEL_PATH}")
