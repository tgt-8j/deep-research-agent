"""SSE 事件聚合器：把后端事件流聚合为前端可渲染的时间线。

参考 Paper-Agent 的 ChatStreamAggregator：
- 将 delta/reasoning/message/tool 等碎片事件整合成稳定消息
- 支持前端乐观更新和流式渲染
- 提供 snapshot() 供前端实时获取当前状态
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("research.presentation")


@dataclass
class StreamMessage:
    """一条可渲染的消息。"""
    id: str
    role: str          # "user" | "assistant" | "system"
    content: str = ""
    reasoning: str = ""
    is_streaming: bool = False
    tool_calls: List[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class StreamAggregator:
    """SSE 事件聚合器。

    将后端推送的细粒度事件聚合成前端可渲染的消息时间线。
    """

    def __init__(self):
        self.messages: List[StreamMessage] = []
        self.is_streaming = False
        self.error: Optional[dict] = None
        self._active_assistant_id: Optional[str] = None

    def add_user_message(self, content: str) -> StreamMessage:
        """添加用户消息（乐观更新）。"""
        msg = StreamMessage(id=str(uuid.uuid4())[:8], role="user", content=content)
        self.messages.append(msg)
        self.is_streaming = True
        return msg

    def _ensure_assistant(self) -> StreamMessage:
        """确保当前有 assistant 消息承接增量内容。"""
        if self._active_assistant_id:
            for m in reversed(self.messages):
                if m.id == self._active_assistant_id:
                    return m
        msg = StreamMessage(
            id=str(uuid.uuid4())[:8],
            role="assistant",
            is_streaming=True,
        )
        self.messages.append(msg)
        self._active_assistant_id = msg.id
        return msg

    def apply(self, event: dict) -> None:
        """应用单个 SSE 事件，增量更新时间线。"""
        event_type = event.get("type") or event.get("event")

        if event_type == "delta":
            assistant = self._ensure_assistant()
            assistant.content += event.get("content", "")

        elif event_type == "progress":
            # 进度事件不改变消息内容，只记录节点状态
            node = event.get("node", "")
            status = event.get("status", "")
            logger.info("[Aggregator] progress | node=%s | status=%s", node, status)

        elif event_type == "reasoning_delta":
            assistant = self._ensure_assistant()
            assistant.reasoning += event.get("content", "")

        elif event_type == "reasoning_end":
            assistant = self._ensure_assistant()
            # reasoning 字段已由 delta 填充

        elif event_type == "message":
            role = event.get("role", "assistant")
            content = event.get("content", "")
            msg = StreamMessage(
                id=str(uuid.uuid4())[:8],
                role=role,
                content=content,
            )
            self.messages.append(msg)

        elif event_type == "tool_call":
            assistant = self._ensure_assistant()
            assistant.tool_calls.append(event.get("tool", {}))

        elif event_type == "error":
            self.error = event
            self.is_streaming = False

        elif event_type in ("final", "__done__", "route"):
            # 终态事件：结束流式
            assistant = self._ensure_assistant()
            assistant.is_streaming = False
            self.is_streaming = False

    def snapshot(self) -> dict:
        """返回前端可渲染的时间线快照。"""
        return {
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "reasoning": m.reasoning,
                    "is_streaming": m.is_streaming,
                    "tool_calls": m.tool_calls,
                    "created_at": m.created_at,
                }
                for m in self.messages
            ],
            "is_streaming": self.is_streaming,
            "error": self.error,
        }

    def reset(self) -> None:
        """重置聚合器状态。"""
        self.messages.clear()
        self.is_streaming = False
        self.error = None
        self._active_assistant_id = None


class NodeReporter:
    """节点级进度上报器。

    参考 Paper-Agent 的 WorkflowNodeReporter：
    - 每个节点开始时发送 "started" 事件
    - 执行中发送 "progress" 事件
    - 完成时发送 "completed" 事件
    - 失败时发送 "failed" 事件
    """

    def __init__(self, node_key: str, node_title: str, emit_callback):
        self.node_key = node_key
        self.node_title = node_title
        self._emit = emit_callback

    def started(self, message: str = "") -> None:
        self._emit({
            "type": "progress",
            "node": self.node_key,
            "message": message or f"{self.node_title}已开始执行",
            "status": "running",
        })

    def progress(self, message: str) -> None:
        self._emit({
            "type": "progress",
            "node": self.node_key,
            "message": message,
            "status": "running",
        })

    def completed(self, message: str = "") -> None:
        self._emit({
            "type": "progress",
            "node": self.node_key,
            "message": message or f"{self.node_title}已完成",
            "status": "success",
        })

    def failed(self, message: str) -> None:
        self._emit({
            "type": "progress",
            "node": self.node_key,
            "message": message,
            "status": "error",
        })
