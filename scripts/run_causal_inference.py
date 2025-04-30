
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# === 参数 ===
DATA_PATH = "data/processed/cleaned_data.csv"
TREAT_COL = "Intervention"
OUTCOME_COL = "7d内术后肺部并发症(0=否，1=是)"

# === 加载并处理数据 ===
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[OUTCOME_COL, TREAT_COL])
df = df.fillna(df.median(numeric_only=True))

# 特征提取
y = df[OUTCOME_COL].astype(int)
t = df[TREAT_COL].astype(int)
X = df.drop(columns=[OUTCOME_COL])

X = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === T-Learner 实现 ===
X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(X_scaled, y, t, test_size=0.2, random_state=42)

# 模型1：对照组（未干预）
model_c = GradientBoostingClassifier(random_state=42)
model_c.fit(X_train[t_train == 0], y_train[t_train == 0])

# 模型2：干预组
model_t = GradientBoostingClassifier(random_state=42)
model_t.fit(X_train[t_train == 1], y_train[t_train == 1])

# 预测个体化效应（ITE）
mu0 = model_c.predict_proba(X_test)[:, 1]
mu1 = model_t.predict_proba(X_test)[:, 1]
ite = mu1 - mu0

# 输出前10位样本的 ITE
print("前10个样本的个体化干预效应 (ITE):")
for i in range(10):
    print(f"Sample {i+1}: ITE = {ite[i]:.4f} (mu1={mu1[i]:.4f}, mu0={mu0[i]:.4f})")

# 估计平均处理效应（ATE）
ate = np.mean(ite)
print(f"平均干预效应 (ATE): {ate:.4f}")
