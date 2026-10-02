from pydantic import BaseModel, Field, field_validator
import re

# 租户/用户标识格式：字母数字 + 连字符 + 下划线，最大 64 字符
_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="研究问题")
    user_id: str = Field(default="default_user", min_length=1, max_length=64)
    thread_id: str = Field(default="default_thread", min_length=1, max_length=64)
    tenant_id: str = Field(default="default_tenant", min_length=1, max_length=64)
    max_iterations: int | None = Field(default=None, ge=1, le=6, description="最大迭代次数")
    enable_memory: bool | None = Field(default=None, description="是否启用记忆系统")

    @field_validator("user_id", "thread_id", "tenant_id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError("标识符只能包含字母、数字、下划线和连字符，长度 1-64")
        return v


class ResearchResponse(BaseModel):
    query: str
    user_id: str
    thread_id: str
    tenant_id: str
    final: str
