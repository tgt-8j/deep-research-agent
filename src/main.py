"""主入口：FastAPI 应用启动。

参考 shopkeeper-agent 的 main.py，提供统一的启动入口。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Windows UTF-8 修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

# 加载环境变量
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("research.main")


def create_application():
    """创建并返回 FastAPI 应用实例。"""
    from src.api.app import create_app
    from src.config import AppConfig

    try:
        config = AppConfig.from_file()
    except FileNotFoundError:
        logger.warning("未找到 config.json，尝试从环境变量加载")
        try:
            config = AppConfig.from_env()
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise

    return create_app(config)


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
