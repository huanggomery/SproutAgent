"""
Plan-and-Solve Agent 实现。

该模块实现了 Plan-and-Solve 模式的 Agent，将任务分解为两个阶段：
1. 规划阶段（Planner）：根据历史对话和用户当前问题，制定分步执行计划。
2. 执行阶段（Executor）：按照计划逐步执行，每个步骤使用 ReAct 模式调用工具完成。
"""
import ast
import json
import re
from typing import Any

from sprout_agent.core.agent import Agent
from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message
from sprout_agent.tools.base import ToolManager

PLANNER_PROMPT_TEMPLATE = """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。

--- 可用工具 ---
{tools}
------

--- 历史对话 ---
{history}
------

--- 用户当前问题 ---
{user_question}
------

请输出执行计划，使用 Python 列表格式，例如：
['步骤1描述', '步骤2描述', '步骤3描述']
"""


EXECUTOR_PROMPT_TEMPLATE = """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请执行当前步骤："""


class Planner:
    """任务规划器，负责根据用户问题和历史对话生成分步执行计划。"""

    def __init__(self, llm: SproutLLM, tools: ToolManager | None = None) -> None:
        """初始化规划器。

        Args:
            llm: 大语言模型客户端实例。
            tools: 工具管理器，用于向规划提示词提供工具信息。
        """
        self.llm = llm
        self.tools = tools

    def plan(self, user_question: str, history: list[Message]) -> list[str]:
        """根据历史上下文和用户当前问题，制定一个清晰、可执行的分步计划。

        Args:
            user_question: 用户当前的问题或任务描述。
            history: 历史对话消息列表。

        Returns:
            执行计划，每个元素为一个步骤描述字符串。

        Raises:
            ValueError: 当无法从模型响应中解析出计划列表，或解析出的计划格式不正确时抛出。
        """
        prompt = self._build_prompt(user_question, history)
        messages = [Message(role="system", content=prompt)]
        response = self.llm.chat(messages)
        content = response.choices[0].message.content or ""

        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            raise ValueError(f"无法从模型响应中解析出计划列表：{content}")

        plan = ast.literal_eval(match.group(0))
        if not isinstance(plan, list) or not all(isinstance(step, str) for step in plan):
            raise ValueError(f"解析出的计划格式不正确，期望字符串列表：{plan}")

        return plan

    def _build_prompt(self, user_question: str, history: list[Message]) -> str:
        """构建规划器的提示词。

        Args:
            user_question: 用户当前的问题或任务描述。
            history: 历史对话消息列表。

        Returns:
            格式化后的规划器提示词字符串。
        """
        conversation_history = history
        if history and history[0].role == "system":
            conversation_history = history[1:]

        lines: list[str] = []
        for index, message in enumerate(conversation_history, start=1):
            lines.append(f"[消息{index}]")
            lines.append(message.to_str())
        history_text = "\n".join(lines)
        tool_schemas = self.tools.to_openai_schema() if self.tools else []
        tools_text = json.dumps(tool_schemas, ensure_ascii=False, indent=2)
        return PLANNER_PROMPT_TEMPLATE.format(
            tools=tools_text,
            history=history_text,
            user_question=user_question,
        )


class Executor:
    """任务执行器，按照既定计划逐步执行任务，每个步骤采用 ReAct 模式进行工具调用。"""

    def __init__(self, llm: SproutLLM, tools: ToolManager) -> None:
        """初始化执行器。

        Args:
            llm: 大语言模型客户端实例。
            tools: 工具管理器，提供工具调用能力。
        """
        self.llm = llm
        self.tools = tools

    def execute(self, question: str, plan: list[str]) -> str:
        """按计划顺序执行所有步骤，返回最终结果。

        Args:
            question: 用户的原始问题。
            plan: 完整的执行计划步骤列表。

        Returns:
            最后一个步骤执行完成后的结果字符串。
        """
        history = []
        result = ""

        for step in plan:
            context = self._build_context(question, plan, history, step)
            result = self._execute_step(context)
            history.append({"step": step, "result": result})

        return result

    def _execute_step(self, context: str, max_steps: int = 10) -> str:
        """执行单个步骤，使用 ReAct 模式循环进行工具调用直到获得结果。

        Args:
            context: 当前步骤的执行上下文提示词。
            max_steps: 单步内最大的工具调用轮数，默认为 10。

        Returns:
            当前步骤的执行结果字符串。
        """
        messages: list[Message] = [Message(role="system", content=context)]
        tool_schemas = self.tools.to_openai_schema()

        for _ in range(max_steps):
            response = self.llm.chat(messages, tools=tool_schemas)
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            tool_calls = assistant_message.tool_calls or []

            messages.append(
                Message(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        tool_call.model_dump(exclude_none=True)
                        for tool_call in tool_calls
                    ]
                    or None,
                )
            )

            if not tool_calls:
                return content

            for tool_result in self.tools.execute_tool_calls(tool_calls):
                messages.append(
                    Message(
                        role="tool",
                        content=tool_result.content,
                        tool_call_id=tool_result.tool_call_id,
                    )
                )

        response = self.llm.chat(messages)
        return response.choices[0].message.content or ""

    def _build_context(
        self,
        question: str,
        plan: list[str],
        history: list[dict[str, str]],
        current_step: str,
    ) -> str:
        """根据原始问题、完整计划、已完成步骤历史和当前步骤，构建执行上下文提示词。

        Args:
            question: 用户的原始问题。
            plan: 完整的执行计划步骤列表。
            history: 已完成步骤的历史记录，每项包含 step 和 result。
            current_step: 当前需要执行的步骤描述。

        Returns:
            格式化后的执行器提示词字符串。
        """
        plan_str = ""
        for i, step in enumerate(plan, start=1):
            plan_str += f"{i}. {step}\n"

        history_str = ""
        for i, step_info in enumerate(history, start=1):
            step_desc = step_info.get("step", "")
            step_result = step_info.get("result", "")
            history_str += f"步骤 {i}: {step_desc}\n"
            history_str += f"结果: {step_result}\n\n"

        return EXECUTOR_PROMPT_TEMPLATE.format(
            question=question,
            plan=plan_str,
            history=history_str,
            current_step=current_step,
        )


class PlanAndSolveAgent(Agent):
    """Plan-and-Solve 模式 Agent，先规划再执行。

    该 Agent 将复杂任务分解为"先规划、后执行"两个阶段：
    - 规划阶段：由 Planner 分析用户问题并生成分步执行计划。
    - 执行阶段：由 Executor 按计划逐步执行，每个步骤内使用 ReAct 模式调用工具。
    """

    def __init__(self, llm: SproutLLM, system_prompt: str | None = None) -> None:
        """初始化 PlanAndSolveAgent。

        Args:
            llm: 大语言模型客户端实例。
            system_prompt: 系统提示词，用于设定 Agent 的角色和行为规范。
        """
        super().__init__(llm, system_prompt=system_prompt)
        self.planner = Planner(llm, self.tools)
        self.executor = Executor(llm, self.tools)

    def run(self, input: str, **kwargs: Any) -> str:
        """运行规划器和执行器，返回最终答案。

        Args:
            input: 用户输入的任务或问题。
            **kwargs: 额外参数，预留供扩展使用。

        Returns:
            最终答案。
        """
        history_messages, _ = self.history.get_history()
        plan = self.planner.plan(input, history_messages)
        result = self.executor.execute(input, plan)
        self.add_message(Message(role="user", content=input))
        self.add_message(Message(role="assistant", content=result))
        return result
