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

    def to_str(self) -> str:
        """将消息体格式化为文本。

        仅负责输出消息自身的结构化内容（Role、可选的 Tool Call ID、
        Content 以及 Tool Calls），不包含任何序号或包裹开头，方便各
        Agent 在构建提示词时按需添加自己的标题。
        """
        lines = [f"Role: {self.role}"]
        if self.tool_call_id:
            lines.append(f"Tool Call ID: {self.tool_call_id}")
        lines.extend(["Content:", self.content or "（空）"])

        if not self.tool_calls:
            return "\n".join(lines)

        # 工具调用属于 assistant 消息，使用缩进保留这一层级关系。
        lines.append("Tool Calls:")
        for tool_call in self.tool_calls:
            function = tool_call.get("function", {})
            arguments = function.get("arguments", "{}")
            lines.extend(
                [
                    f"  - Call ID: {tool_call.get('id', '')}",
                    f"    Tool: {function.get('name', '')}",
                    f"    Arguments: {arguments}",
                ]
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        role = self.role.upper()
        message = f"[{ts}] [{role}] {self.content}"

        if self.role == "assistant" and self.tool_calls:
            tool_calls = json.dumps(self.tool_calls, ensure_ascii=False)
            message = f"{message}\n工具调用请求：{tool_calls}"

        return message
