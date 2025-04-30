
#!/bin/bash
# 一键构建并运行 Streamlit 服务（默认端口8501）
docker build -t cardiac-app .
docker run -p 8501:8501 cardiac-app
