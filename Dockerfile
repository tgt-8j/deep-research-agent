# ===== 构建阶段 =====
FROM docker.1ms.run/library/python:3.11-slim AS base

WORKDIR /app

# 系统依赖（Milvus client 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 非 root 用户运行
RUN useradd --create-home appuser
USER appuser

# ===== 生产阶段 =====
FROM base AS production

# 环境变量（由 docker-compose 的 env_file 注入）
ENV APP_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.app_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
