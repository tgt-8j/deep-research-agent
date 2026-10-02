#!/bin/bash
# 一键启动脚本：构建镜像并启动所有服务
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "===== Deep Research Agent — Docker 启动 ====="
echo ""

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "[警告] .env 文件不存在，复制 .env.example"
    cp .env.example .env
fi

# 构建并启动
echo "[1/3] 构建 Docker 镜像..."
docker-compose build --no-cache backend

echo "[2/3] 启动所有服务..."
docker-compose up -d

echo "[3/3] 等待服务健康检查..."
sleep 5

# 检查服务状态
echo ""
echo "===== 服务状态 ====="
docker-compose ps

echo ""
echo "===== 健康检查 ====="
curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "API 尚未就绪，请稍等..."

echo ""
echo "===== 访问地址 ====="
echo "  API 文档:   http://localhost:8000/docs"
echo "  ReDoc:      http://localhost:8000/redoc"
echo "  健康检查:   http://localhost:8000/health"
echo "  Metrics:    http://localhost:8000/metrics"
echo ""
echo "===== 查看日志 ====="
echo "  docker-compose logs -f backend"
echo ""
