import json
import unittest

from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message


class TestSproutLLMIntegration(unittest.TestCase):
    """使用 .env 中的 DeepSeek 配置验证真实模型调用。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.llm = SproutLLM()

    def test_basic_chat(self) -> None:
        messages = [
            Message(role="system", content="你是一个简洁、准确的智能助手。"),
            Message(role="user", content="请只回复：SproutLLM 调用成功"),
        ]

        response = self.llm.chat(messages)
        content = response.choices[0].message.content

        self.assertIsNotNone(content)
        self.assertIn("SproutLLM 调用成功", content)

    def test_tool_call(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        messages = [
            Message(
                role="system",
                content=(
                    "回答天气问题时必须先调用一次 get_weather。收到工具结果后，"
                    "直接根据结果回答，不要再次调用工具。"
                ),
            ),
            Message(role="user", content="北京今天天气怎么样？"),
        ]

        # 第一轮由模型选择工具并生成调用参数。
        first_response = self.llm.chat(messages, tools=tools)
        assistant = first_response.choices[0].message
        self.assertTrue(assistant.tool_calls, "模型没有发起工具调用")

        tool_call = assistant.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        self.assertIsInstance(arguments, dict)

        messages.append(
            Message(
                role="assistant",
                content=assistant.content or "",
                tool_calls=[
                    call.model_dump(exclude_none=True) for call in assistant.tool_calls
                ],
            )
        )

        # Agent 层执行工具，并通过 tool_call_id 回传固定的模拟结果。
        messages.append(
            Message(
                role="tool",
                content="北京今天晴，气温 25℃。",
                tool_call_id=tool_call.id,
            )
        )

        second_response = self.llm.chat(messages, tools=tools)
        final_content = second_response.choices[0].message.content

        self.assertIsNotNone(final_content)
        self.assertIn("25", final_content)


if __name__ == "__main__":
    unittest.main()
