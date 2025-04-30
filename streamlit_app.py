
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CS-PRF 术后肺部并发症预测", layout="wide")

# === 模型和标准化器路径 ===
MODEL_PATH = "outputs/models/gbc_model.pkl"
DATA_SAMPLE_PATH = "data/processed/cleaned_data.csv"

# === 加载模型与示例数据 ===
model = joblib.load(MODEL_PATH)
df_sample = pd.read_csv(DATA_SAMPLE_PATH)
feature_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler().fit(df_sample[feature_cols].fillna(df_sample.median(numeric_only=True)))

# === 页面展示 ===
st.title("🫀 心脏术后肺部并发症预测工具")
st.markdown("请输入术前和术中指标，系统将预测该患者发生术后肺部并发症的风险")

# === 用户输入 ===
user_input = {}
col1, col2, col3 = st.columns(3)
for i, col in enumerate(feature_cols):
    if i % 3 == 0:
        container = col1
    elif i % 3 == 1:
        container = col2
    else:
        container = col3
    user_input[col] = container.number_input(col, value=float(df_sample[col].median()), format="%.4f")

# === 预测按钮 ===
if st.button("预测风险"):
    input_array = np.array([user_input[col] for col in feature_cols]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prob = model.predict_proba(input_scaled)[0][1]
    st.success(f"预测该患者术后7日内发生肺部并发症的风险为：**{prob:.2%}**")
