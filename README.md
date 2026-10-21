# Deep Research Agent — LangGraph 多智能体深度调研系统

> 输入一个问题，自动完成多源检索 → 证据裁判 → 反思补搜 → 撰写带引用的完整报告

## 快速开始

```bash
# 1. 安装依赖（含开发依赖）
pip install -e ".[dev]"

# 2. 配置：两种方式任选
#    a) 复制 .env.example 为 .env 并填入真实 Key（推荐）
#    b) 复制 config.json.example 为 config.json 并填入 Key
#    优先级：环境变量 > config.json > 默认值
cp .env.example .env

# 3. 运行测试
python -m pytest tests/ -q

# 4. 启动服务
python -m src.main
# 访问 http://127.0.0.1:8000/docs (Swagger UI)
```

`.env` 与 `config.json` 统一由 `src/settings.py` 定位到**项目根目录**，
所有入口（`python -m src.main`、`python src/main.py`、测试）行为一致。

## 目录说明（很重要）

| 目录 | 状态 | 说明 |
|------|------|------|
| `src/` | **当前运行时** | 唯一在跑的实现：FastAPI + LangGraph + 检索 + 记忆 |
| `app/` | **legacy，待删除** | 上一版实现（旧 graph/nodes/memory/tools），仍被少量老测试引用 |
| `front/agent_front` | 前端 | Vue 3 + Vite |
| `tests/` | 测试 | 610 条，pytest + pytest-asyncio |

`src/` 已不再 import `app/` 任何模块（可用
`python -c "import src.main, sys; print('app' in sys.modules)"` 验证为 `False`）。
删除 `app/` 前需要先把 `tests/test_e2e_workflow.py`、`test_tools.py` 等几个
老测试迁到 `src/` 对应模块。

## 架构概览

```
START -> intent -> query_rewrite -> [direct_answer -> END]        (简单问题)
                                -> plan -> web_search --┐
                                         local_rag -----┤-> rerank -> deep_dive -> analyze
                                                        ↑                                   │
                                          reflect <-(需补搜)                                 │
                                                        └───────────────────────────────────┘
                                                                   analyze -> write -> END
```

**12 个 LangGraph 节点：**

| 节点 | 职责 |
|------|------|
| intent | 意图识别（直接回答 vs 深度调研）|
| query_rewrite | Query 改写 + 子问题拆解 |
| plan | 搜索计划生成 |
| web_search | 网页搜索（Bocha API + 重试/超时）|
| local_rag | 本地知识库（Milvus + Hybrid Search）|
| rerank | RRF 双路融合 + 相关性重排序 |
| deep_dive | 证据裁判（评分/去重/冲突审计）|
| analyze | 分析归纳（结论 + 置信度）|
| reflect | 反思补搜（生成 gap 查询）|
| write | 写作（引用溯源 + 参考文献）|
| approval | 人工审批（`interrupt()`，证据 >10 时触发）|
| direct_answer | 直接回答（简单问题路径）|

## 技术亮点

### 检索优化链路（三级）

1. **Hybrid Search** — BM25 关键词 + 向量语义 RRF 融合
   - 公式：`score = Σ(1 / (k + rank_i))`，k=60
   - 降级：无 BM25 库时自动跳过

2. **Rerank 重排序** — 两级用的是**不同信号**（不是同一个 embedding 算两遍）
   - 第一级：BM25 词法粗筛 Top-30，零 API 调用，只排序不丢弃
   - 第二级：语义精排 Top-10 —— 默认 Embedding 余弦；装 `sentence-transformers`
     并设 `RERANKER_BACKEND=cross_encoder` 可换成 BGE-Reranker Cross-Encoder
   - 降级：精排不可用时返回粗筛顺序，实际用了哪级写在 `rerank_stats.fine_stage`

3. **证据裁判** — 来源评分 + 冲突检测
   - local=0.92, .gov.cn/.edu=0.88, 媒体=0.72, 其他=0.58
   - 同 claim 不同 score 标记 `disputed=True`

### 记忆系统（三级）

| 层级 | 存储 | TTL | 用途 |
|------|------|-----|------|
| 短期 | Redis / Memory | 1h~7d | 当前对话上下文 |
| 长期 | SQLite + Embedding | 永久 | 用户偏好、历史任务 |
| 检索 | Milvus + BM25 | - | 知识库事实检索 |

### 工程化实践

- **分层 State** — `ResearchState` 只放控制字段，业务数据拆为子 TypedDict
- **Runtime Context** — 依赖注入，无全局单例，支持并发测试
- **取消边界** — 每个节点前后检查停止信号；信号走 `ContextVar`（请求级、不进 checkpoint），
  客户端断开或调用取消接口后，在下一个节点边界协作式停止
- **Checkpointer** — `MemorySaver` 持久化，支持中断恢复
- **SSE 流式** — 事件以 JSON 推送（`data: {...}\n\n`），前端直接 `JSON.parse`
- **重试/超时** — 指数退避（1s/2s/4s），单次请求 30s，并发上限 3
- **SSRF 防护** — 内网 IP / metadata 地址拦截
- **Prometheus** — `/metrics` 暴露节点调用次数与耗时

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/research/run | 同步执行深度调研 |
| POST | /api/v1/research/stream | SSE 流式执行 |
| POST | /api/v1/research/sessions/{key}/cancel | 请求取消正在运行的任务（协作式，节点边界生效）|
| GET | /api/v1/research/sessions | 列出用户会话历史 |
| GET | /api/v1/research/sessions/{key} | 查询会话状态 |
| POST | /api/v1/research/sessions/{key}/resume | 从 Checkpoint 恢复 |
| GET | /health | 健康检查 |
| GET | /metrics | Prometheus 指标 |

## 测试

```bash
# 全部测试
python -m pytest tests/ -q

# 带覆盖率
python -m pytest tests/ --cov=src --cov-report=html
```

`tests/conftest.py` 自己实现了 `tmp_path`：pytest 内部的临时目录固定用
`mode=0o700` 创建，在 Windows / 加固环境下会变成「连自己都读不了」的目录，
导致所有用 `tmp_path` 的用例在 setup 阶段就 PermissionError。改用默认权限 +
每次会话一个全新根目录后，该问题消失。

## 已知限制 & 设计取舍

| 项目 | 现状 | 原因 / 处理 |
|------|------|-------------|
| `app/` 仍在仓库里 | 只被少量老测试引用，`src/` 已不依赖它 | 迁移那几个测试后即可整目录删除 |
| 默认精排是 Embedding 近似 | 不是真正的 cross-encoder | 设 `RERANKER_BACKEND=cross_encoder` + 装 `sentence-transformers` 即启用；`rerank_stats.fine_stage` 可自查 |
| 鉴权未接入路由 | `src/api/app.py` 只挂了 tracing + 限流，JWT 中间件仍在 `app/backend` | 多租户字段目前由请求体传入，接入前不要对外暴露 |
| `approval_node` 覆盖率低 | 43% | `interrupt()` 需外部 `Command(resume=...)` 驱动，单测难覆盖 |
| langchain-community 废弃警告 | 正常运行，仅 DeprecationWarning | DashScopeEmbeddings 仍在该包，计划迁至独立包 |
| 中文相关性用 bigram 统计 | 同义改写（"电动车" vs "新能源汽车"）仍可能被判低分 | 词法预筛宁松勿严 + 官方域名豁免；要更强召回可换 jieba 分词 + 同义词扩展 |

## 技术栈

Python 3.10+ / LangGraph / FastAPI / DashScope(Qwen) / Milvus / Redis / PostgreSQL / Docker
