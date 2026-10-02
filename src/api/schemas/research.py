"""API Schema 模块：定义请求和响应的数据模型。

参考 shopkeeper-agent 的 api/schemas/ 目录，将数据校验集中在 schema 层。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """深度调研请求体。"""

    query: str = Field(..., description="用户的研究问题")
    user_id: str = Field(default="default_user", description="用户 ID")
    thread_id: str = Field(default="default_thread", description="会话线程 ID")
    tenant_id: str = Field(default="default_tenant", description="租户 ID")
    max_iterations: int | None = Field(default=None, ge=1, le=5, description="最大反思循环次数")
    enable_memory: bool | None = Field(default=None, description="是否启用记忆功能")


class ResearchResponse(BaseModel):
    """深度调研响应体。"""

    query: str
    user_id: str
    thread_id: str
    tenant_id: str
    final: str
    intent: str = Field(default="multiagent", description="路由结果：direct | multiagent")


class StreamResponse(BaseModel):
    """SSE 流式响应的事件体。"""

    type: str  # "status" | "progress" | "delta" | "route" | "final" | "error" | "__done__"
    message: str | None = None
    node: str | None = None
    status: str | None = None
    content: str | None = None
    query: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    tenant_id: str | None = None
    final: str | None = None
