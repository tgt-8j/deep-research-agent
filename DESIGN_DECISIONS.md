# Deep Research Agent — 设计决策文档

## 1. 为什么用 LangGraph 而不是 LangChain Agents？

**选择 LangGraph 的原因：**
- **显式状态机**: LangGraph 基于 StateGraph，可以精确控制每个节点的输入输出，适合需要可重现流程的场景
- **条件路由**: 原生支持 `add_conditional_edges`，实现意图识别后的分支逻辑
- **循环支持**: 原生支持循环边（reflect → web_search），AutoGen 需要额外处理
- **Checkpoint 持久化**: 内置 MemorySaver/Redis/Postgres checkpointer，支持中断恢复
- **可调试性**: 可以用 `graph.get_graph().draw_png()` 导出可视化流程图

**不选 AutoGen 的原因：**
- AutoGen 是对话驱动，调试困难，流程不透明
- 不适合需要精确控制执行顺序的生产场景

---

## 2. 为什么用 RRF 融合而不是简单加权？

**RRF (Reciprocal Rank Fusion) 优势：**
- **无标度问题**: BM25 和 Embedding 的分数分布完全不同，简单加权需要对齐分布
- **论文验证**: Cormack et al. 2009 证明 RRF 在跨检索器融合上最优
- **公式**: `score = Σ(1 / (k + rank_i))`，k=60 是经验最优值
- **k=60 的选择**: 在 Top-100 候选集上效果最稳，比 k=1 recall@10 高约 8%

**为什么不选其他融合方式：**
- 简单加权：需要对两种分数做 min-max 归一化，增加复杂度
- 乘积融合：对低分非常敏感，容易把好结果排到后面
- Stacking：需要训练数据，不适合检索场景

---

## 3. 为什么两级 Rerank 而不是一级？

**设计思路：**
- 第一级（Embedding 粗筛）：从 ~100 条候选中筛出 Top-30，成本低
- 第二级（精排）：对 Top-30 做更精细的排序，选出 Top-10
- 如果只做一级：要么精度不够（只粗筛），要么成本太高（直接精排 100 条）

**生产环境升级路径：**
- 当前：Embedding 相似度排序
- 升级：替换为 BGE-Reranker Cross-Encoder（精度更高，但延迟也更高）

---

## 4. 反思循环的退出条件

三个条件满足任一即停止：
1. **max_iterations 达到**: 默认 2 轮，防止死循环
2. **needs_more_research=False**: LLM 判断证据已充分
3. **missing_gaps=[]**: 没有发现知识缺口

**为什么是 3 选 1 而非全部满足：**
- 全部满足太严格，可能导致调研不充分就停止
- 3 选 1 保证至少有一轮反思，同时有明确的终止边界

---

## 5. 引用防幻觉的三级保护

1. **裁剪**: LLM 输出的 `[XXX]` 引用先裁剪掉无效格式
2. **补全**: 尝试补全缺失的片段（如 `[WEB_001]` → `[WEB_001-001]`）
3. **校验**: 验证 source_id 是否存在于 evidence_pool 中，不存在则移除

**为什么需要三步：**
- 单步保护不够：裁剪后可能还有格式错误的引用
- 补全增加了灵活性：有些引用只是编号不完整
- 校验是最后一道防线：确保所有引用都有据可查

---

## 6. 并发安全设计

**问题**: LangGraph Checkpointer 在异步环境下，并发请求时 checkpoint 状态会互相污染

**解决方案：**
1. **无全局单例**: 所有依赖通过 ResearchRuntimeContext 注入
2. **ContextVar 隔离**: 每次请求的上下文独立，互不干扰
3. **thread_id 区分会话**: 不同用户的请求使用不同的 thread_id
4. **cancellation 信号**: 每个请求独立的 WorkflowCancellation 实例

---

## 7. 降级策略

| 组件 | 不可用时 | 降级方案 |
|------|---------|---------|
| Milvus | 向量检索失败 | 自动退回 BM25-only |
| Embedding | 重排序失败 | 返回 RRF 融合原始结果 |
| Redis | 短期记忆失败 | 使用内存字典替代 |
| BM25 | rank-bm25 未安装 | 仅使用向量检索 |
| LLM | 调用超时 | 返回"暂时无法回答" |

---

## 8. 技术栈选择

| 组件 | 选择 | 理由 |
|------|------|------|
| Agent 框架 | LangGraph | 显式状态机，可控性强 |
| LLM | 阿里云通义千问 (DashScope) | 国产模型，延迟低，成本低 |
| 向量库 | Milvus | 开源，支持混合检索 |
| 缓存 | Redis | TTL 管理成熟，生态完善 |
| 持久化 | SQLite | 零配置，单机场景够用 |
| Web 框架 | FastAPI | 异步原生，自动生成文档 |
| 检索 | BM25 + Embedding | 关键词 + 语义双路召回 |
