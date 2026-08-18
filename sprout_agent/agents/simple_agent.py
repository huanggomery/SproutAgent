from typing import Any

from sprout_agent.core.agent import Agent
from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message


class SimpleAgent(Agent):
    """简单的Agent，支持多轮工具调用。

    该 Agent 采用循环执行模式：在每一步中，模型可以选择直接回复用户，
    或者发起工具调用。工具执行结果会被加入对话历史，供模型在下一轮参考。
    当达到最大步数限制时，会强制模型直接给出最终回答。
    """

    def __init__(self, llm: SproutLLM) -> None:
        """初始化 SimpleAgent。

        Args:
            llm: 大语言模型客户端实例。
        """
        super().__init__(llm)

    def run(self, input: str, **kwargs: Any) -> str:
        """执行 SimpleAgent 的核心逻辑。

        Args:
            input: 用户输入的任务或问题。
            **kwargs: 额外参数，支持：
                - max_steps (int): 最大工具调用轮数，默认为 5。

        Returns:
            Agent 执行后的最终响应结果。

        Raises:
            ValueError: 当 max_steps 不是正整数时抛出。
        """
        max_steps = kwargs.get("max_steps", 5)
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps 必须是正整数")

        self.add_message(Message(role="user", content=input))
        tool_schemas = self.tools.to_openai_schema()
        last_content = ""

        for _ in range(max_steps):
            # 每一轮都传入当前完整历史，使模型能看到此前的回复和工具结果。
            messages, _ = self.history.get_history()
            response = self.llm.chat(messages, tools=tool_schemas)
            assistant_message = response.choices[0].message
            last_content = assistant_message.content or ""
            tool_calls = assistant_message.tool_calls or []

            self.add_message(
                Message(
                    role="assistant",
                    content=last_content,
                    tool_calls=[
                        tool_call.model_dump(exclude_none=True)
                        for tool_call in tool_calls
                    ]
                    or None,
                )
            )

            # 没有工具调用时，说明模型已给出最终回复，直接返回。
            if not tool_calls:
                return last_content

            # 执行工具调用并将结果加入历史。
            for tool_result in self.execute_tool_calls(tool_calls):
                self.add_message(Message(
                    role="tool",
                    content=tool_result.content,
                    tool_call_id=tool_result.tool_call_id,
                ))

        # 达到最大步数后关闭工具调用，让模型根据已有历史直接给出最终回答。
        messages, _ = self.history.get_history()
        response = self.llm.chat(messages)
        final_content = response.choices[0].message.content or ""
        self.add_message(Message(role="assistant", content=final_content))
        return final_content
