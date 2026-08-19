import unittest
from types import SimpleNamespace
from typing import Any

from sprout_agent.agents.react_agent import (
    CurrentTask,
    DEFAULT_SYSTEM_PROMPT,
    FINAL_SYSTEM_PROMPT,
    ReActAgent,
    ReActStep,
)
from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message
from sprout_agent.tools.base import Tool, ToolParams, ToolResult


class FakeToolCall:
    """模拟 OpenAI SDK 返回的工具调用对象。"""

    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeLLM:
    """按顺序返回预设响应，并记录每次模型输入。"""

    def __init__(self, replies: list[tuple[str, list[FakeToolCall]]]) -> None:
        self.replies = iter(replies)
        self.calls: list[tuple[list[Message], list[dict] | None]] = []

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
    ) -> SimpleNamespace:
        self.calls.append((messages, tools))
        content, tool_calls = next(self.replies)
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class EchoTool(Tool):
    """返回输入文本的测试工具。"""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, description="回显输入文本")
        self.calls: list[dict] = []

    def execute(self, params: dict) -> str:
        self.calls.append(params)
        return str(params["text"])

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="text",
                type="string",
                description="需要回显的文本",
                required=True,
            )
        ]


class FailingTool(Tool):
    """始终失败的测试工具。"""

    def __init__(self, name: str) -> None:
        super().__init__(name=name, description="模拟业务工具失败")

    def execute(self, params: dict) -> str:
        raise RuntimeError("模拟业务工具失败")

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="id",
                type="integer",
                description="测试 ID",
                required=True,
            )
        ]


class RecordingReActAgent(ReActAgent):
    """记录真实运行中的工具调用批次。"""

    def __init__(self, llm: SproutLLM) -> None:
        super().__init__(llm)
        self.tool_call_batches: list[list[str]] = []

    def execute_tool_calls(self, tool_calls: list[Any]) -> list[ToolResult]:
        self.tool_call_batches.append(
            [tool_call.function.name for tool_call in tool_calls]
        )
        return super().execute_tool_calls(tool_calls)


class ReActAgentTestCase(unittest.TestCase):
    """提供 ReActAgent 上下文的精确预期文本。"""

    @staticmethod
    def expected_default_history() -> str:
        return (
            "--- 历史对话 ---\n\n"
            "（无）\n\n"
            "------"
        )

    @classmethod
    def expected_empty_context(cls, user_query: str) -> str:
        return (
            f"{cls.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            f"user_query: {user_query}\n"
            "------"
        )


class TestReActAgentContext(ReActAgentTestCase):
    """验证发送给模型的上下文结构。"""

    def test_build_messages_formats_all_history_and_current_task(self) -> None:
        agent = ReActAgent(FakeLLM([]))
        history = [
            Message(role="system", content="历史系统提示词"),
            Message(role="user", content="查询北京天气"),
            Message(
                role="assistant",
                content="正在查询。",
                tool_calls=[
                    {
                        "id": "history_call_1",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"北京"}',
                        },
                    }
                ],
            ),
            Message(
                role="tool",
                content="北京晴天",
                tool_call_id="history_call_1",
            ),
            Message(role="summary", content="用户正在规划北京行程"),
        ]
        current_task = CurrentTask(
            user_query="查询天气和门票",
            steps=[
                ReActStep(
                    content="天气和门票可以并行查询。",
                    actions=[
                        ToolResult(
                            name="weather",
                            tool_call_id="current_call_1",
                            content="明天多云",
                            arguments={"city": "北京"},
                            status="success",
                        ),
                        ToolResult(
                            name="ticket",
                            tool_call_id="current_call_2",
                            content="门票接口超时",
                            arguments={"place": "北海公园"},
                            status="failed",
                        ),
                    ],
                )
            ],
        )

        messages = agent.build_messages(history, current_task)

        expected_prompt = (
            "--- 历史对话 ---\n\n"
            "[消息 1]\n"
            "Role: user\n"
            "Content:\n"
            "查询北京天气\n\n"
            "[消息 2]\n"
            "Role: assistant\n"
            "Content:\n"
            "正在查询。\n"
            "Tool Calls:\n"
            "  - Call ID: history_call_1\n"
            "    Tool: weather\n"
            '    Arguments: {"city":"北京"}\n\n'
            "[消息 3]\n"
            "Role: tool\n"
            "Tool Call ID: history_call_1\n"
            "Content:\n"
            "北京晴天\n\n"
            "[消息 4]\n"
            "Role: summary\n"
            "Content:\n"
            "用户正在规划北京行程\n\n"
            "------\n\n"
            "--- 当前任务 ---\n"
            "user_query: 查询天气和门票\n\n"
            "步骤 1：\n"
            "Analysis:\n"
            "天气和门票可以并行查询。\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: weather\n"
            '  Input: {"city": "北京"}\n'
            "  Status: success\n"
            "  Observation: 明天多云\n"
            "  [Action 2]\n"
            "  Tool: ticket\n"
            '  Input: {"place": "北海公园"}\n'
            "  Status: failed\n"
            "  Observation: 门票接口超时\n"
            "------"
        )
        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("user", expected_prompt),
            ],
        )

    def test_current_task_omits_analysis_when_content_is_empty(self) -> None:
        current_task = CurrentTask(
            user_query="查询天气",
            steps=[
                ReActStep(
                    actions=[
                        ToolResult(
                            name="weather",
                            tool_call_id="call_1",
                            content="晴天",
                            arguments={"city": "北京"},
                        )
                    ]
                )
            ],
        )

        prompt = ReActAgent._format_current_task(current_task)

        self.assertEqual(
            prompt,
            "--- 当前任务 ---\n"
            "user_query: 查询天气\n\n"
            "步骤 1：\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: weather\n"
            '  Input: {"city": "北京"}\n'
            "  Status: success\n"
            "  Observation: 晴天\n"
            "------",
        )

    def test_custom_system_prompt_keeps_user_prompt_unchanged(self) -> None:
        agent = ReActAgent(FakeLLM([]))
        history, _ = agent.history.get_history()
        current_task = CurrentTask(user_query="总结当前结果")

        normal_messages = agent.build_messages(history, current_task)
        final_messages = agent.build_messages(
            history,
            current_task,
            system_prompt=FINAL_SYSTEM_PROMPT,
        )

        expected_user_prompt = self.expected_empty_context("总结当前结果")
        self.assertEqual(
            [(message.role, message.content) for message in normal_messages],
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("user", expected_user_prompt),
            ],
        )
        self.assertEqual(
            [(message.role, message.content) for message in final_messages],
            [
                ("system", FINAL_SYSTEM_PROMPT),
                ("user", expected_user_prompt),
            ],
        )


class TestReActAgentRun(ReActAgentTestCase):
    """验证工具循环、异常处理和状态流转。"""

    def test_max_steps_rejects_invalid_values(self) -> None:
        for invalid_value in (0, -1, 1.5, "1", True, None):
            with self.subTest(max_steps=invalid_value):
                agent = ReActAgent(FakeLLM([]))
                with self.assertRaisesRegex(ValueError, "max_steps 必须是正整数"):
                    agent.run("测试任务", max_steps=invalid_value)

    def test_plain_content_is_final_answer(self) -> None:
        llm = FakeLLM([("直接回答", [])])
        agent = ReActAgent(llm)

        answer = agent.run("简单问题")

        self.assertEqual(answer, "直接回答")
        self.assertEqual(len(llm.calls), 1)
        history, _ = agent.history.get_history()
        self.assertEqual(
            [(message.role, message.content) for message in history],
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("user", "简单问题"),
                ("assistant", "直接回答"),
            ],
        )

    def test_tool_call_without_content_is_executed(self) -> None:
        llm = FakeLLM(
            [
                ("", [FakeToolCall("call_1", "echo", '{"text":"业务结果"}')]),
                ("最终答案", []),
            ]
        )
        tool = EchoTool("echo")
        agent = ReActAgent(llm)
        agent.add_tool(tool)

        answer = agent.run("测试无分析内容")

        self.assertEqual(answer, "最终答案")
        self.assertEqual(tool.calls, [{"text": "业务结果"}])
        expected_second_prompt = (
            f"{self.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            "user_query: 测试无分析内容\n\n"
            "步骤 1：\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: echo\n"
            '  Input: {"text": "业务结果"}\n'
            "  Status: success\n"
            "  Observation: 业务结果\n"
            "------"
        )
        self.assertEqual(llm.calls[1][0][1].content, expected_second_prompt)

    def test_content_with_multiple_tools_is_recorded_as_one_step(self) -> None:
        llm = FakeLLM(
            [
                (
                    "天气和门票可以并行查询。",
                    [
                        FakeToolCall("call_1", "weather", '{"text":"明天多云"}'),
                        FakeToolCall("call_2", "ticket", '{"text":"成人票20元"}'),
                    ],
                ),
                ("查询完成", []),
            ]
        )
        agent = ReActAgent(llm)
        agent.add_tool(EchoTool("weather"))
        agent.add_tool(EchoTool("ticket"))

        answer = agent.run("查询天气和门票")

        self.assertEqual(answer, "查询完成")
        self.assertEqual(len(llm.calls), 2)
        expected_second_prompt = (
            f"{self.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            "user_query: 查询天气和门票\n\n"
            "步骤 1：\n"
            "Analysis:\n"
            "天气和门票可以并行查询。\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: weather\n"
            '  Input: {"text": "明天多云"}\n'
            "  Status: success\n"
            "  Observation: 明天多云\n"
            "  [Action 2]\n"
            "  Tool: ticket\n"
            '  Input: {"text": "成人票20元"}\n'
            "  Status: success\n"
            "  Observation: 成人票20元\n"
            "------"
        )
        self.assertEqual(llm.calls[1][0][1].content, expected_second_prompt)

    def test_content_with_tool_calls_is_not_treated_as_final_answer(self) -> None:
        llm = FakeLLM(
            [
                (
                    "先查询业务信息。",
                    [FakeToolCall("call_1", "echo", '{"text":"查询结果"}')],
                ),
                ("基于查询结果的最终答案", []),
            ]
        )
        agent = ReActAgent(llm)
        agent.add_tool(EchoTool("echo"))

        answer = agent.run("需要查询的问题")

        self.assertEqual(answer, "基于查询结果的最终答案")
        self.assertEqual(len(llm.calls), 2)

    def test_empty_response_retries_without_changing_context(self) -> None:
        llm = FakeLLM([("", []), ("重试后的答案", [])])
        agent = ReActAgent(llm)

        answer = agent.run("测试空响应")

        self.assertEqual(answer, "重试后的答案")
        expected_prompt = self.expected_empty_context("测试空响应")
        self.assertEqual(
            [messages[1].content for messages, _ in llm.calls],
            [expected_prompt, expected_prompt],
        )

    def test_failed_business_tool_is_included_in_next_context(self) -> None:
        llm = FakeLLM(
            [
                (
                    "调用可能失败的工具。",
                    [FakeToolCall("call_1", "unstable", '{"id":7}')],
                ),
                ("最终答案", []),
            ]
        )
        agent = ReActAgent(llm)
        agent.add_tool(FailingTool("unstable"))

        answer = agent.run("测试工具失败")

        self.assertEqual(answer, "最终答案")
        expected_second_prompt = (
            f"{self.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            "user_query: 测试工具失败\n\n"
            "步骤 1：\n"
            "Analysis:\n"
            "调用可能失败的工具。\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: unstable\n"
            '  Input: {"id": 7}\n'
            "  Status: failed\n"
            "  Observation: 工具 unstable 调用失败：模拟业务工具失败\n"
            "------"
        )
        self.assertEqual(llm.calls[1][0][1].content, expected_second_prompt)

    def test_unknown_tool_is_included_as_failed_result(self) -> None:
        llm = FakeLLM(
            [
                ("尝试调用工具。", [FakeToolCall("call_1", "missing", "{}")] ),
                ("修正后的答案", []),
            ]
        )
        agent = ReActAgent(llm)

        answer = agent.run("测试未知工具")

        self.assertEqual(answer, "修正后的答案")
        expected_second_prompt = (
            f"{self.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            "user_query: 测试未知工具\n\n"
            "步骤 1：\n"
            "Analysis:\n"
            "尝试调用工具。\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: missing\n"
            "  Input: {}\n"
            "  Status: failed\n"
            "  Observation: 工具 missing 调用失败：工具 missing 不存在\n"
            "------"
        )
        self.assertEqual(llm.calls[1][0][1].content, expected_second_prompt)

    def test_invalid_tool_arguments_are_included_as_failed_result(self) -> None:
        llm = FakeLLM(
            [
                ("参数可能有误。", [FakeToolCall("call_1", "echo", "not-json")]),
                ("修正后的答案", []),
            ]
        )
        agent = ReActAgent(llm)
        agent.add_tool(EchoTool("echo"))

        answer = agent.run("测试参数错误")

        self.assertEqual(answer, "修正后的答案")
        expected_second_prompt = (
            f"{self.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            "user_query: 测试参数错误\n\n"
            "步骤 1：\n"
            "Analysis:\n"
            "参数可能有误。\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: echo\n"
            "  Input: {}\n"
            "  Status: failed\n"
            "  Observation: 工具 echo 调用失败：Expecting value: line 1 column 1 (char 0)\n"
            "------"
        )
        self.assertEqual(llm.calls[1][0][1].content, expected_second_prompt)

    def test_max_steps_uses_final_prompt_with_completed_steps(self) -> None:
        llm = FakeLLM(
            [
                (
                    "先调用业务工具。",
                    [FakeToolCall("call_1", "echo", '{"text":"业务结果"}')],
                ),
                ("上限后的最终回答", []),
            ]
        )
        agent = ReActAgent(llm)
        agent.add_tool(EchoTool("echo"))

        answer = agent.run("测试带步骤的上限", max_steps=1)

        self.assertEqual(answer, "上限后的最终回答")
        normal_messages, normal_tools = llm.calls[0]
        final_messages, final_tools = llm.calls[1]
        expected_normal_prompt = self.expected_empty_context("测试带步骤的上限")
        expected_final_prompt = (
            f"{self.expected_default_history()}\n\n"
            "--- 当前任务 ---\n"
            "user_query: 测试带步骤的上限\n\n"
            "步骤 1：\n"
            "Analysis:\n"
            "先调用业务工具。\n"
            "Actions:\n"
            "  [Action 1]\n"
            "  Tool: echo\n"
            '  Input: {"text": "业务结果"}\n'
            "  Status: success\n"
            "  Observation: 业务结果\n"
            "------"
        )
        self.assertIsNotNone(normal_tools)
        self.assertIsNone(final_tools)
        self.assertEqual(
            [(message.role, message.content) for message in normal_messages],
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("user", expected_normal_prompt),
            ],
        )
        self.assertEqual(
            [(message.role, message.content) for message in final_messages],
            [
                ("system", FINAL_SYSTEM_PROMPT),
                ("user", expected_final_prompt),
            ],
        )

    def test_second_run_context_contains_only_previous_final_dialogue(self) -> None:
        llm = FakeLLM([("第一轮答案", []), ("第二轮答案", [])])
        agent = ReActAgent(llm)

        first_answer = agent.run("第一轮问题")
        second_answer = agent.run("第二轮问题")

        self.assertEqual(first_answer, "第一轮答案")
        self.assertEqual(second_answer, "第二轮答案")
        expected_second_prompt = (
            "--- 历史对话 ---\n\n"
            "[消息 1]\n"
            "Role: user\n"
            "Content:\n"
            "第一轮问题\n\n"
            "[消息 2]\n"
            "Role: assistant\n"
            "Content:\n"
            "第一轮答案\n\n"
            "------\n\n"
            "--- 当前任务 ---\n"
            "user_query: 第二轮问题\n"
            "------"
        )
        self.assertEqual(llm.calls[1][0][1].content, expected_second_prompt)

    def test_only_registered_business_tools_are_sent_to_model(self) -> None:
        llm = FakeLLM([("最终答案", [])])
        agent = ReActAgent(llm)
        agent.add_tool(EchoTool("weather"))

        agent.run("测试工具列表")

        schemas = llm.calls[0][1]
        self.assertEqual(
            [schema["function"]["name"] for schema in schemas or []],
            ["weather"],
        )


class TestReActAgentIntegration(unittest.TestCase):
    """使用 .env 中的配置验证真实模型的原生工具循环。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.llm = SproutLLM()

    def test_agent_real_run_can_finish_with_plain_content(self) -> None:
        agent = RecordingReActAgent(self.llm)

        answer = agent.run("请计算 17 + 25，并给出简洁的最终答案。")

        self.assertIn("42", answer)
        self.assertEqual(agent.tool_call_batches, [])
        history, _ = agent.history.get_history()
        self.assertEqual(
            [(message.role, message.content) for message in history],
            [
                ("system", DEFAULT_SYSTEM_PROMPT),
                ("user", "请计算 17 + 25，并给出简洁的最终答案。"),
                ("assistant", answer),
            ],
        )


if __name__ == "__main__":
    unittest.main()
