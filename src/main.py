"""主入口：FastAPI 应用启动。

项目根路径与 .env 的定位统一由 ``src.settings`` 负责，本文件不再自己算
``parents[n]``（改造前这里把 ``src/`` 当成了项目根，导致根目录的 .env 读不到）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Windows 下强制 UTF-8 输出，避免中文日志乱码。
# 只包装"还不是 UTF-8"的流，避免在 pytest 等已替换 sys.stdout 的环境里二次包装。
if sys.platform == "win32":
    import io

    for _name in ("stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is None or not hasattr(_stream, "buffer"):
            continue
        _encoding = (getattr(_stream, "encoding", "") or "").lower().replace("-", "")
        if _encoding != "utf8":
            setattr(
                sys,
                _name,
                io.TextIOWrapper(_stream.buffer, encoding="utf-8", errors="replace"),
            )

# src/main.py -> parents[1] = 项目根。必须加项目根而非 src/，
# 否则 `python src/main.py` 方式启动时 `import src.*` 会失败。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.settings import CONFIG_PATH, ENV_PATH  # noqa: E402  导入即加载 .env

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
        logger.info("配置来源: %s", CONFIG_PATH)
    except FileNotFoundError:
        logger.warning("未找到 %s，回退到环境变量", CONFIG_PATH.name)
        logger.info("环境变量文件: %s (exists=%s)", ENV_PATH, ENV_PATH.exists())
        config = AppConfig.from_env()

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
