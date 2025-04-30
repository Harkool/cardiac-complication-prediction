
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === 路径参数 ===
DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "outputs/models/gbc_model.pkl"
SAVE_DIR = "outputs/shap"
LABEL_COL = "7d内术后肺部并发症(0=否，1=是)"

# === 读取数据并处理 ===
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[LABEL_COL])
df = df.fillna(df.median(numeric_only=True))

# 构建特征和标签
y = df[LABEL_COL].astype(int)
X = df.drop(columns=[LABEL_COL])
X = X.select_dtypes(include=[np.number])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 载入模型 ===
model = joblib.load(MODEL_PATH)

# === SHAP 解释 ===
explainer = shap.Explainer(model.predict, X_scaled)
shap_values = explainer(X_scaled)

# === 绘图保存 ===
plt.figure()
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/shap_beeswarm.png")
print(f"SHAP图保存至: {SAVE_DIR}/shap_beeswarm.png")

# 可选：保存summary bar图
plt.figure()
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/shap_summary_bar.png")
print(f"SHAP条形图保存至: {SAVE_DIR}/shap_summary_bar.png")
