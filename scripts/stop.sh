#!/bin/bash
# 停止所有 Docker 服务
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "===== 停止所有服务 ====="
docker-compose down

echo ""
echo "===== 清理数据卷（可选）====="
echo "  # 删除数据卷（会清除所有持久化数据）"
echo "  docker-compose down -v"
echo ""
echo "完成。"
