from abc import ABC, abstractmethod
from typing import Any

from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message
from sprout_agent.context.history import HistoryManager
from sprout_agent.tools.base import Tool, ToolManager, ToolResult


class Agent(ABC):
    """所有 Agent 的基类，定义通用接口和基础能力。"""

    def __init__(
        self,
        llm: SproutLLM,
        system_prompt: str | None = None,
    ) -> None:
        """初始化 Agent 基类。

        Args:
            llm: 大语言模型客户端实例。
            system_prompt: 系统提示词，用于设定 Agent 的角色和行为规范。
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = ToolManager()
        self.history = HistoryManager()

        if system_prompt:
            self.history.append(Message(role="system", content=system_prompt))

    @abstractmethod
    def run(self, input: str, **kwargs: Any) -> str:
        """执行 Agent 的核心逻辑。

        子类必须实现此方法以定义具体的 Agent 行为。

        Args:
            input: 用户输入的任务或问题。
            **kwargs: 额外的参数，供子类扩展使用。

        Returns:
            Agent 执行后的响应结果。
        """
        ...

    def clear_history(self) -> None:
        """清除 Agent 的对话历史。"""
        self.history.clear()
    
    def add_message(self, message: Message) -> None:
        """添加一条消息到 Agent 的对话历史。"""
        self.history.append(message)

    def add_tool(self, tool: Tool) -> None:
        """添加一个工具到 Agent 的工具管理器。"""
        self.tools.add_tool(tool)
    
    def remove_tool(self, tool_name: str) -> None:
        """从 Agent 的工具管理器中移除一个工具。"""
        self.tools.remove_tool(tool_name)

    def execute_tool_calls(self, tool_calls: list[Any]) -> list[ToolResult]:
        """执行模型发起的工具调用，并返回参数、结果和执行状态。"""
        return self.tools.execute_tool_calls(tool_calls)
