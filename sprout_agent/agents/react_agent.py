import json
from dataclasses import dataclass, field
from typing import Any

from sprout_agent.core.agent import Agent
from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message
from sprout_agent.tools.base import ToolResult


DEFAULT_SYSTEM_PROMPT = """你是一个使用 ReAct 模式完成任务的智能助手。
请结合历史对话和当前任务中的已有步骤，自主决定直接回答或调用一个或多个业务工具。

调用业务工具时，请在回复正文中简洁说明当前的分析和调用依据；能够直接回答时，不调用工具并输出完整的最终答案。
请参考已有的工具执行结果，避免重复执行结果仍然有效的工具。"""


FINAL_SYSTEM_PROMPT = """你需要基于用户提供的历史对话和当前任务，直接给出最终答案。
本次不提供任何工具。请充分利用已有分析、工具调用参数、执行状态和 Observation，输出准确、完整、可直接返回给用户的回答。
信息不足时应如实说明已有结论和缺失信息。"""


@dataclass
class ReActStep:
    """一次模型响应的可选分析及其全部工具执行记录。"""

    content: str = ""
    actions: list[ToolResult] = field(default_factory=list)


@dataclass
class CurrentTask:
    """一次 run 调用中仅供模型使用的临时任务轨迹。"""

    user_query: str
    steps: list[ReActStep] = field(default_factory=list)


class ReActAgent(Agent):
    """ReActAgent，支持多轮工具调用。"""

    def __init__(
        self,
        llm: SproutLLM,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """初始化 ReActAgent。

        Args:
            llm: 大语言模型客户端实例。
            system_prompt: 系统提示词，默认引导模型使用 ReAct 模式。
        """
        super().__init__(llm, system_prompt=system_prompt)

    def build_messages(
        self,
        history_messages: list[Message],
        current_task: CurrentTask,
        *,
        system_prompt: str | None = None,
    ) -> list[Message]:
        """根据长期问答历史和当前任务轨迹构造模型上下文。"""
        history_prompt = self._format_history(history_messages)
        current_task_prompt = self._format_current_task(current_task)
        # 历史与当前任务合并为一条 user 消息。
        user_prompt = f"{history_prompt}\n\n{current_task_prompt}"

        messages: list[Message] = []
        effective_system_prompt = system_prompt or self.system_prompt
        if effective_system_prompt:
            messages.append(Message(role="system", content=effective_system_prompt))
        messages.append(Message(role="user", content=user_prompt))
        return messages

    @staticmethod
    def _format_history(history_messages: list[Message]) -> str:
        """将全部历史消息及其工具调用信息格式化为文本。"""
        lines = ["--- 历史对话 ---"]
        # system prompt 已作为独立的首条模型消息发送，历史文本中不再重复携带。
        conversation_messages = history_messages
        if history_messages and history_messages[0].role == "system":
            conversation_messages = history_messages[1:]

        if not conversation_messages:
            return "\n".join([*lines, "", "（无）", "", "------"])

        for index, message in enumerate(conversation_messages, start=1):
            lines.extend(
                [
                    "",
                    f"[消息 {index}]",
                    f"Role: {message.role}",
                ]
            )
            if message.tool_call_id:
                lines.append(f"Tool Call ID: {message.tool_call_id}")
            lines.extend(["Content:", message.content or "（空）"])

            if not message.tool_calls:
                continue

            # 工具调用属于 assistant 消息，使用缩进保留这一层级关系。
            lines.append("Tool Calls:")
            for tool_call in message.tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                lines.extend(
                    [
                        f"  - Call ID: {tool_call.get('id', '')}",
                        f"    Tool: {function.get('name', '')}",
                        f"    Arguments: {arguments}",
                    ]
                )
        lines.extend(["", "------"])
        return "\n".join(lines)

    @staticmethod
    def _format_current_task(current_task: CurrentTask) -> str:
        """将结构化步骤格式化为模型可读的当前任务轨迹。"""
        lines = [
            "--- 当前任务 ---",
            f"user_query: {current_task.user_query}",
        ]

        for index, step in enumerate(current_task.steps, start=1):
            lines.extend(["", f"步骤 {index}："])
            if step.content:
                lines.extend(["Analysis:", step.content])
            lines.append("Actions:")
            for action_index, action in enumerate(step.actions, start=1):
                action_input = json.dumps(
                    action.arguments or {},
                    ensure_ascii=False,
                )
                lines.extend(
                    [
                        f"  [Action {action_index}]",
                        f"  Tool: {action.name}",
                        f"  Input: {action_input}",
                        f"  Status: {action.status}",
                        f"  Observation: {action.content}",
                    ]
                )

        lines.append("------")
        return "\n".join(lines)

    def _record_tool_results(
        self,
        current_task: CurrentTask,
        content: str,
        tool_results: list[ToolResult],
    ) -> None:
        """将一次模型响应的可选文本和全部工具结果记录为同一步骤。"""
        # 同一响应中的多个工具属于一个决策批次，后续构造上下文时保持该关系。
        current_task.steps.append(
            ReActStep(
                content=content,
                actions=tool_results,
            )
        )

    def _save_final_answer(self, user_query: str, answer: str) -> str:
        """任务结束后仅把本轮用户问题和最终答案写入长期历史。"""
        self.add_message(Message(role="user", content=user_query))
        self.add_message(Message(role="assistant", content=answer))
        return answer

    def run(self, input: str, **kwargs: Any) -> str:
        """运行 ReAct 推理循环并返回最终答案。

        Args:
            input: 用户输入的任务或问题。
            **kwargs: 额外参数，支持 ``max_steps`` 指定最大循环次数，默认为 10。

        Returns:
            最终答案。

        Raises:
            ValueError: 当 ``max_steps`` 不是正整数时抛出。
        """
        max_steps = kwargs.get("max_steps", 10)
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError("max_steps 必须是正整数")

        current_task = CurrentTask(user_query=input)
        tool_schemas = self.tools.to_openai_schema()

        for _ in range(max_steps):
            history_messages, _ = self.history.get_history()
            messages = self.build_messages(history_messages, current_task)
            response = self.llm.chat(messages, tools=tool_schemas)
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            tool_calls = assistant_message.tool_calls or []

            if tool_calls:
                tool_results = self.execute_tool_calls(tool_calls)
                self._record_tool_results(current_task, content, tool_results)
                continue

            if content:
                return self._save_final_answer(input, content)

            # 既没有工具调用也没有回复内容时无法推进任务，保留上下文并重试。
            continue

        # 达到循环上限后不再提供工具，要求模型基于现有轨迹直接形成最终答案。
        history_messages, _ = self.history.get_history()
        messages = self.build_messages(
            history_messages,
            current_task,
            system_prompt=FINAL_SYSTEM_PROMPT,
        )
        response = self.llm.chat(messages)
        final_content = response.choices[0].message.content or ""
        return self._save_final_answer(input, final_content)
