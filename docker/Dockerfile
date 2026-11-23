# --docker 模式的沙箱映像：只有 Python 與 pytest，執行時加上 --network none。
#   docker build -t mini-agent-sandbox:latest docker/
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest==8.3.3
WORKDIR /work
