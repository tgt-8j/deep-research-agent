# Deep Research Agent — LangGraph 多智能体深度调研系统

> 输入一个问题，自动完成多源检索 → 证据裁判 → 反思补搜 → 撰写带引用的完整报告

## 快速开始

`ash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入真实的 DASHSCOPE_API_KEY

# 3. 运行测试
python -m pytest tests/ -v

# 4. 启动服务
python -m src.main
# 访问 http://127.0.0.1:8000/docs (Swagger UI)
`

## 架构概览

`
START -> intent -> [direct_answer -> END]  (简单问题)
           -> plan -> web_search ---┐
                    local_rag ----┤-> rerank -> deep_dive -> analyze -> [reflect -> web_search/local_rag] (循环)
                                   ↓
                                 write -> END
`

**12 个 LangGraph 节点：**

| 节点 | 职责 |
|------|------|
| intent | 意图识别（直接回答 vs 深度调研）|
| query_rewrite | Query 改写 + 子问题拆解 |
| plan | 搜索计划生成 |
| web_search | 网页搜索（Bocha API + 重试/超时）|
| local_rag | 本地知识库（Milvus + Hybrid Search）|
| erank | 两级重排序（粗筛 Top-30 + 精排 Top-10）|
| deep_dive | 证据裁判（评分/去重/冲突审计）|
| nalyze | 分析归纳（结论 + 置信度）|
| eflect | 反思补搜（生成 gap 查询）|
| write | 写作（引用溯源 + 参考文献）|
| pproval | 人工审批（证据 >10 时触发）|
| direct_answer | 直接回答（简单问题路径）|

## 技术亮点

### 检索优化链路（三级）

1. **Hybrid Search** — BM25 关键词 + 向量语义 RRF 融合
   - 公式：score = Sum(1 / (k + rank_i))，k=60
   - 降级：无 BM25 库时自动跳过

2. **Rerank 重排序** — 粗筛 + 精排两级策略
   - Embedding 粗筛 Top-30 -> 精排 Top-10
   - 生产环境可替换为 BGE-Reranker Cross-Encoder
   - 降级：Embedding 不可用时直接返回原始顺序

3. **证据裁判** — 来源评分 + 冲突检测
   - local=0.92, .gov.cn/.edu=0.88, 媒体=0.72, 其他=0.58
   - 同 claim 不同 score 标记 disputed=True

### 记忆系统（三级）

| 层级 | 存储 | TTL | 用途 |
|------|------|-----|------|
| 短期 | Redis / Memory | 1h~7d | 当前对话上下文 |
| 长期 | SQLite + Embedding | 永久 | 用户偏好、历史任务 |
| 检索 | Milvus + BM25 | - | 知识库事实检索 |

### 工程化实践

- **分层 State** — ResearchState 只放控制字段，业务数据拆为子类型
- **Runtime Context** — 依赖注入，无全局单例，支持并发测试
- **取消边界** — 节点前后检查 cancellation，支持用户中途停止
- **Checkpointer** — MemorySaver 持久化，支持中断恢复
- **SSE 流式** — 每个节点实时进度上报前端
- **重试/超时** — 指数退避（1s/2s/4s），总超时 60s，并发上限 3
- **SSRF 防护** — 内网 IP / metadata 地址拦截

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/research/run | 同步执行深度调研 |
| POST | /api/v1/research/stream | SSE 流式执行 |
| GET | /api/v1/research/sessions | 列出用户会话历史 |
| GET | /api/v1/research/sessions/{key} | 查询会话状态 |
| POST | /api/v1/research/sessions/{key}/resume | 从 Checkpoint 恢复 |
| GET | /health | 健康检查 |
| GET | /metrics | Prometheus 指标 |

## 测试

`ash
# 运行全部测试
python -m pytest tests/ -v

# 带覆盖率
python -m pytest tests/ --cov=src --cov-report=html
`

## Windows 下运行测试

Windows 系统上 pytest 的 tmp_path fixture 默认使用 %TEMP%\pytest-of-<user> 路径，
部分环境下可能因权限问题报错。如遇 PermissionError，请设置临时目录：

`powershell
 = "C:\Users\AUSU\AppData\Local\Temp\pytest-tmp"
python -m pytest tests/ --ignore=tests/test_api_routes.py -v
`

> 注：test_api_routes.py 中的集成测试依赖 tmp_path fixture，Windows 上建议排除该文件单独运行核心单元测试。

## 已知限制 & 设计取舍

| 项目 | 现状 | 原因 | 面试应对 |
|------|------|------|----------|
| langchain-community 废弃警告 | 正常运行，有 DeprecationWarning | DashScopeEmbeddings 在 langchain-community 中，官方已 sunset | 可答：已关注迁移路径，计划迁至 langchain-dashscope 独立包 |
| approval_node 覆盖率 43% | interrupt() 阻塞等待外部 resume，单测无法触发 | LangGraph 人工审批节点的固有限制 | interrupt() 需要外部 command.resume() 驱动，单测无法覆盖是预期行为 |
| Windows 下 test_api_routes.py 权限报错 | pytest tmp_path 默认路径被 sandbox 进程占用 | 旧进程以其他用户创建了残留目录 | 已配置 conftest.py 自动重定向到项目本地 _tmp，Linux/CI 不受影响 |

## 技术栈

Python 3.10+ / LangGraph / FastAPI / DashScope(Qwen) / Milvus / Redis / PostgreSQL / Docker
