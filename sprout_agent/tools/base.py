from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel


class ToolParams(BaseModel):
    """工具参数定义模型。"""

    name: str
    type: Literal["object", "string", "number", "integer", "boolean", "array"]
    description: str
    required: bool = False
    default: Any = None


class ToolResult(BaseModel):
    """工具调用结果模型。"""

    name: str
    tool_call_id: str
    content: str


class Tool(ABC):
    """所有工具的抽象基类。

    自定义工具需继承此类并实现 execute 和 get_params 方法。
    """

    def __init__(self, name: str, description: str) -> None:
        """初始化工具实例。

        Args:
            name: 工具名称，唯一标识。
            description: 工具功能描述。
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, params: dict) -> str:
        """执行工具调用并返回结果。"""
        ...
    
    @abstractmethod
    def get_params(self) -> list[ToolParams]:
        """获取工具参数定义列表。"""
        ...
    

class ToolManager:
    """工具管理器，负责注册、查询、执行工具及 schema 转换。"""

    def __init__(self, tools: list[Tool] = []) -> None:
        """初始化工具管理器。

        Args:
            tools: 初始工具列表。
        """
        self._tools = {}
        for tool in tools:
            self.add_tool(tool)
    
    def add_tool(self, tool: Tool) -> None:
        """添加工具。

        Raises:
            ValueError: 工具名称已存在时抛出。
        """
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已存在")
        self._tools[tool.name] = tool
    
    def remove_tool(self, tool_name: str) -> None:
        """移除工具。
        工具不存在时仅打印警告信息，不抛出异常。
        """
        if not self.check_tool(tool_name):
            print(f"⚠️ 警告: 工具 {tool_name} 不存在，无法移除")
            return
        del self._tools[tool_name]
    
    def check_tool(self, tool_name: str) -> bool:
        """检查工具是否存在。"""
        return tool_name in self._tools
    
    def get_tool(self, tool_name: str) -> Tool:
        """获取工具实例。

        Raises:
            ValueError: 工具不存在时抛出。
        """
        if not self.check_tool(tool_name):
            raise ValueError(f"工具 {tool_name} 不存在")
        return self._tools[tool_name]

    def execute_tool(self, tool_name: str, params: dict) -> str:
        """执行指定工具并返回结果。"""
        tool = self.get_tool(tool_name)
        return tool.execute(params)
    
    def to_openai_schema(self) -> list[dict[str, Any]]:
        """转换为 OpenAI/DeepSeek 兼容的工具定义格式。"""
        schemas: list[dict[str, Any]] = []

        for tool in self._tools.values():
            properties: dict[str, dict[str, Any]] = {}
            required: list[str] = []

            for param in tool.get_params():
                property_schema: dict[str, Any] = {
                    "type": param.type,
                    "description": param.description,
                }
                if "default" in param.model_fields_set:
                    property_schema["default"] = param.default

                properties[param.name] = property_schema
                if param.required:
                    required.append(param.name)

            parameters: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                    },
                }
            )

        return schemas
