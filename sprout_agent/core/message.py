import json
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "system", "tool", "assistant", "summary"]
    content: str
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        message = {
            "role": self.role,
            "content": self.content,
        }

        # assistant 消息负责记录模型发起的工具调用。
        if self.role == "assistant" and self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls

        # tool 消息通过调用 ID 将执行结果关联到对应的工具请求。
        if self.role == "tool" and self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id

        return message

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        role = self.role.upper()
        message = f"[{ts}] [{role}] {self.content}"

        if self.role == "assistant" and self.tool_calls:
            tool_calls = json.dumps(self.tool_calls, ensure_ascii=False)
            message = f"{message}\n工具调用请求：{tool_calls}"

        return message
