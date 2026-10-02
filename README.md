# Deep Research Agent — LangGraph 多智能体深度调研系统

> 输入一个问题，自动完成多源检索 → 证据裁判 → 反思补搜 → 撰写带引用的完整报告

## 快速开始

```bash
cd deep_research/deep_research
pip install -e ".[dev]"
cp .env.example .env
python -m pytest tests/ -v
python -m src.main
```

## 架构概览

```
START → intent → [direct_answer → END]  (简单问题)
               → plan → web_search ──┐
                       local_rag ──┤→ rerank → deep_dive → analyze → [reflect → web_search/local_rag] (循环)
                       (并行)                            ↓
                                                              write → END
```

**9 个 LangGraph 节点：**
| 节点 | 职责 |
|------|------|
| `intent` | 意图识别（直接回答 vs 深度调研） |
| `query_rewrite` | Query 改写 + 子问题拆解 |
| `plan` | 搜索计划生成 |
| `web_search` | 网页搜索（Bocha API + 重试/超时） |
| `local_rag` | 本地知识库（Milvus + Hybrid Search） |
| `rerank` | 两级重排序（粗筛 Top-30 + 精排 Top-10） |
| `deep_dive` | 证据裁判（评分/去重/冲突审计） |
| `analyze` | 分析归纳（结论 + 置信度） |
| `reflect` | 反思补搜（生成 gap 查询） |
| `write` | 写作（引用溯源 + 参考文献） |

## 技术亮点

### 检索优化链路（三级）

1. **Hybrid Search** — BM25 关键词 + 向量语义 RRF 融合
   - `src/retrieval/hybrid_search.py`: BM25Okapi + Reciprocal Rank Fusion
   - 公式：`score = Σ(1 / (k + rank_i))`，k=60
   - 降级：无 BM25 库时自动跳过

2. **Rerank 重排序** — 粗筛 + 精排两级策略
   - `src/retrieval/reranker.py`: Embedding 粗筛 Top-30 → 精排 Top-10
   - 生产环境可替换为 BGE-Reranker Cross-Encoder
   - 降级：Embedding 不可用时直接返回原始顺序

3. **证据裁判** — 来源评分 + 冲突检测
   - local=0.92, .gov.cn/.edu=0.88, 媒体=0.72, 其他=0.58
   - 同 claim 不同 score 标记 `disputed=True`

### 记忆系统（三级）

| 层级 | 存储 | TTL | 用途 |
|------|------|-----|------|
| 短期 | Redis / Memory | 1h~7d | 当前对话上下文 |
| 长期 | SQLite + Embedding | 永久 | 用户偏好、历史任务 |
| 检索 | Milvus + BM25 | - | 知识库事实检索 |

- 偏好提取：从对话中自动归纳 `.gov.cn` / `Agent` / `中文` 等偏好
- 语义召回：Embedding 余弦相似度匹配用户历史记忆

### 工程化实践

- **分层 State** — `ResearchState` 只放控制字段，业务数据拆为子类型
- **Runtime Context** — 依赖注入，无全局单例，支持并行测试
- **取消边界** — 节点前后检查 cancellation，支持用户中途停止
- **Checkpointer** — MemorySaver 持久化，支持中断恢复
- **SSE 流式** — 每个节点实时进度上报前端
- **重试/超时** — 指数退避（1s/2s/4s），总超时 60s，并发上限 3
- **SSRF 防护** — 内网 IP / metadata 地址拦截

### 引用溯源

- 引用格式：`[PREFIX_SEQ-ID]`（如 `[WEB_001-001]`）
- 三步保护：证据裁剪 → 字段补全 → 引用校验
- 非法引用自动移除，不存在的 source_id 报错

## 性能数据

```
Query: "LangGraph 反思循环的实现原理"
  Round 1: 1.02s, 2409 tokens, 2 iterations
  Round 2: 1.01s, 2405 tokens, 2 iterations
  Avg: 1.01s, 2407 tokens, 2.0 iterations

Concurrent (5 users): all completed, no state leakage
```

Benchmark 脚本：`study/benchmark_research.py`

## 项目结构

```
deep_research/
├── src/
│   ├── graph/
│   │   ├── graph.py           # StateGraph 构建 + 条件路由
│   │   └── nodes/             # 9 个 LangGraph 节点
│   │       ├── intent_node.py
│   │       ├── query_rewrite_node.py
│   │       ├── plan_node.py
│   │       ├── web_search_node.py
│   │       ├── local_rag_node.py
│   │       ├── rerank_node.py      # ← Day 4 新增
│   │       ├── deep_dive_node.py
│   │       ├── analyze_node.py
│   │       ├── reflect_node.py
│   │       └── write_node.py
│   ├── retrieval/
│   │   ├── web_search.py       # Bocha Web Search（含 SSRF + 重试）
│   │   ├── knowledge_base.py   # KnowledgeBaseClient
│   │   ├── hybrid_search.py    # BM25 + RRF 融合
│   │   └── reranker.py         # ← Day 4 新增：两级重排序
│   ├── memory/
│   │   ├── short_term.py       # ← Day 6 重写：Redis + Memory 双后端
│   │   └── long_term.py        # ← Day 7 重写：SQLite + Embedding
│   ├── state.py                # ResearchState + 子类型
│   ├── context.py              # ResearchRuntimeContext
│   └── services/workflow.py    # WorkflowService
├── tests/
│   ├── test_hybrid_search.py   # 11 cases
│   ├── test_reranker.py        # 16 cases
│   ├── test_retrieval_quality.py # 24 cases
│   ├── test_short_term_memory.py # 23 cases
│   ├── test_long_term_memory_new.py # 26 cases
│   ├── test_memory_system.py   # 21 cases
│   ├── test_web_search_robustness.py # 27 cases
│   └── test_graph.py           # 8 cases
└── study/
    ├── Day4_Rerank_重排序模块.md
    ├── Day5_RAG检索优化对比测试.md
    ├── Day6-8_记忆系统升级.md
    ├── Day9-12_性能与可靠性.md
    └── benchmark_research.py
```

## 测试结果

```
511 passed, 7 skipped in ~90s
```

| 测试文件 | 说明 |
|---------|------|
| test_graph.py | 图构建 + 路由逻辑（8） |
| test_e2e_workflow.py | 端到端工作流（37） |
| test_e2e_realistic.py | 真实场景E2E测试（32） |
| test_llm_adapter.py | LLM多Provider测试（14） |
| test_prompt_models.py | Pydantic模型验证（29） |
| test_coverage_gap.py | 节点覆盖补全（32） |
| test_api_routes.py | API路由测试（13） |
| test_* | 其余单元测试（346） |

## API 端点

```bash
# 同步调研
curl -X POST http://localhost:8000/api/v1/research/run \
  -H "Content-Type: application/json" \
  -d '{"query": "LangGraph 反思循环实现原理"}'

# SSE 流式
curl -N -X POST http://localhost:8000/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "LangGraph 反思循环实现原理"}'

# 健康检查
curl http://localhost:8000/health
```

## 🎯 面试重点（开发者自述）

### 可以聊的技术亮点

1. **LangGraph 工作流设计**
   - 12 个节点的有向图，支持条件路由和循环（reflect → web_search/local_rag）
   - 手写 `_with_cancellation_boundary` 装饰器，实现节点级取消检查
   - StateGraph 分层 State 设计：主 State 只放控制字段，业务数据拆为 EvidenceItem、Finding 等子类型

2. **检索优化链路**
   - Hybrid Search：BM25 关键词 + 向量语义 RRF 融合，降级策略保证可用性
   - 两级 Rerank：Embedding 粗筛 Top-30 → 精排 Top-10
   - 证据裁判：来源评分（gov.cn 0.88 > 新闻 0.72 > 普通 0.58）+ 冲突检测

3. **工程化实践**
   - Token Bucket 限流中间件：按用户分桶，防恶意刷接口
   - ContextVar 请求追踪：并发请求的日志隔离
   - SSRF 防护：拦截内网 IP、云元数据地址
   - 指数退避重试 + 总超时控制

4. **记忆系统**
   - 三级记忆：短期（Redis/Memory）+ 长期（SQLite + Embedding）+ 检索（Milvus）
   - 用户偏好自动提取：从对话中归纳 `.gov.cn` / `中文` 等偏好标签

### 踩过的坑

- Windows + pytest 9.1.1 缓存兼容性问题 → 通过 `-p no:cacheprovider` 解决
- LangGraph Checkpointer 在异步环境下的状态隔离 → 使用 `thread_id` 区分并发会话
- RAG 检索零结果时 LLM 幻觉 → 添加相似度阈值 + 明确降级文案
- monkeypatch.delenv 全量测试时的环境泄漏 → 改用 patch.dict 隔离

## License

Educational project — for interview preparation only.

## Docker 快速启动

### 前置条件
- Docker Desktop 已安装并运行
- Docker Compose v2+

### 一键启动

```bash
cd deep_research/deep_research
bash scripts/start.sh
```

### 手动启动

```bash
# 构建并启动
docker-compose up -d --build

# 查看日志
docker-compose logs -f backend

# 停止
docker-compose down
```

### 访问地址

| 服务 | 地址 |
|------|------|
| API 文档 | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| 健康检查 | http://localhost:8000/health |
| Prometheus 指标 | http://localhost:8000/metrics |
| 前端 | http://localhost:80 |

### 服务架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Backend    │────▶│  PostgreSQL │
│  (Nginx)    │     │  (FastAPI)   │     │  (长期记忆)  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Redis   │ │  Milvus  │ │   Etcd   │
        │(短期记忆) │ │(向量检索) │ │(Milvus)  │
        └──────────┘ └──────────┘ └──────────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │  MinIO   │
                                    │(对象存储) │
                                    └──────────┘
```
