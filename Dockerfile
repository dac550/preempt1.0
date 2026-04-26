FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev openssl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件并安装（利用缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制项目代码
COPY generated/ ./generated/
COPY certs/ ./certs/
COPY config.py ./

EXPOSE 50051

CMD ["python", "-u", "generated/tee_server.py"]