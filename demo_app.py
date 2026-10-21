"""Streamlit 前端 Demo：Deep Research Agent"""
import streamlit as st
import httpx
import json
import asyncio

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Deep Research Agent", page_icon="🔬", layout="wide")
st.title("🔬 Deep Research Agent")
st.markdown("基于 LangGraph 的多智能体深度调研系统")

with st.sidebar:
    st.markdown("### 设置")
    query = st.text_area("研究问题", height=100, key="query")
    user_id = st.text_input("用户 ID", value="demo_user", key="user_id")
    max_iter = st.slider("最大反思轮次", 1, 5, 2, key="max_iter")
    run_btn = st.button("🚀 开始调研", type="primary")

if run_btn and query:
    with st.spinner("正在执行深度调研..."):
        try:
            resp = httpx.post(
                f"{API_BASE}/api/v1/research/run",
                json={
                    "query": query,
                    "user_id": user_id,
                    "max_iterations": max_iter,
                },
                timeout=120.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success("✅ 调研完成")
                st.markdown("### 研究报告")
                st.markdown(data.get("final", ""))
                st.markdown(f"**意图**: {data.get('intent', 'N/A')}")
            else:
                st.error(f"请求失败: {resp.status_code}")
                st.json(resp.json())
        except httpx.ConnectError:
            st.error("无法连接到服务，请确认 `python -m src.main` 已启动")
        except Exception as e:
            st.error(f"错误: {e}")

st.markdown("---")
st.markdown("""
### 系统架构
- **12 节点 LangGraph 工作流**: intent → plan → web_search + local_rag → rerank → deep_dive → analyze → reflect → write
- **三级检索优化**: BM25 + Embedding RRF 融合 → 两级 Rerank → 证据裁判
- **三层记忆**: 短期(Redis) / 长期(SQLite+Embedding) / 检索层(Milvus)
- **工程化**: JWT认证 + Token Bucket限流 + SSRF防护 + OpenTelemetry追踪
""")
