
# 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 拷贝文件
COPY . /app

# 安装依赖
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && mkdir -p outputs/models outputs/figures data/processed

# 运行 Streamlit 应用
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
