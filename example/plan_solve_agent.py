import json
import sys
from typing import Any

from sprout_agent.agents.plan_solve_agent import Executor, PlanAndSolveAgent, Planner
from sprout_agent.core.llm import SproutLLM
from sprout_agent.core.message import Message
from sprout_agent.tools.base import Tool, ToolManager, ToolParams, ToolResult


DEFAULT_USER_QUERY = (
    "请为3个人设计一顿预算不超过150元的周末清淡晚餐，"
    "给出最终菜单、食材采购清单和预计总价。"
)


class TracingSproutLLM(SproutLLM):
    """仅打印模型的工具决策，完整文本由所属阶段统一输出。"""

    def __init__(self) -> None:
        super().__init__()
        self.request_number = 0

    def chat(
        self,
        messages: list[Any],
        *,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        response = super().chat(messages, stream=stream, tools=tools)
        if stream:
            return response

        self.request_number += 1
        tool_calls = response.choices[0].message.tool_calls or []
        print(f"\n---------- 模型请求 {self.request_number} 的工具决策 ----------")
        if not tool_calls:
            print("Tool Calls: （无）")
            return response

        for tool_call in tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
        return response


class SearchRecipesTool(Tool):
    """搜索符合人数和口味偏好的演示菜谱。"""

    def __init__(self) -> None:
        super().__init__(
            name="search_recipes",
            description="根据用餐人数和口味偏好搜索晚餐菜谱候选。",
        )

    def execute(self, params: dict) -> str:
        people = int(params["people"])
        preference = str(params["preference"])
        return json.dumps(
            {
                "people": people,
                "preference": preference,
                "recipes": [
                    {"name": "清蒸鲈鱼", "ingredients": ["鲈鱼", "姜", "葱"]},
                    {"name": "蒜蓉西兰花", "ingredients": ["西兰花", "蒜"]},
                    {"name": "番茄菌菇汤", "ingredients": ["番茄", "白玉菇"]},
                ],
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="people",
                type="integer",
                description="用餐人数",
                required=True,
            ),
            ToolParams(
                name="preference",
                type="string",
                description="口味偏好，例如清淡",
                required=True,
            ),
        ]


class QueryIngredientPricesTool(Tool):
    """查询演示用食材价格。"""

    def __init__(self) -> None:
        super().__init__(
            name="query_ingredient_prices",
            description="查询菜谱所需食材的市场价格，食材应来自菜谱搜索结果。",
        )

    def execute(self, params: dict) -> str:
        ingredients = str(params["ingredients"])
        # 固定价格使示例可以稳定复现，同时保留前后步骤的数据依赖。
        return json.dumps(
            {
                "queried_ingredients": ingredients,
                "prices_cny": {
                    "鲈鱼": 48,
                    "姜": 2,
                    "葱": 2,
                    "西兰花": 12,
                    "蒜": 3,
                    "番茄": 8,
                    "白玉菇": 10,
                },
                "estimated_total": 85,
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="ingredients",
                type="string",
                description="菜谱搜索结果中列出的全部食材",
                required=True,
            )
        ]


class BuildDinnerPlanTool(Tool):
    """根据菜谱和价格生成最终晚餐方案。"""

    def __init__(self) -> None:
        super().__init__(
            name="build_dinner_plan",
            description="结合菜谱候选、食材价格和预算，生成菜单及采购清单。",
        )

    def execute(self, params: dict) -> str:
        recipe_candidates = params["recipe_candidates"]
        price_summary = params["price_summary"]
        # 工具调用通常直接传入 JSON 对象，同时兼容调用方传入序列化后的 JSON 字符串。
        if isinstance(recipe_candidates, str):
            recipe_candidates = json.loads(recipe_candidates)
        if isinstance(price_summary, str):
            price_summary = json.loads(price_summary)
        if not isinstance(recipe_candidates, dict) or not isinstance(
            price_summary, dict
        ):
            raise ValueError("菜谱候选和价格汇总必须是 JSON 对象")

        budget = int(params["budget"])
        recipes = recipe_candidates.get("recipes")
        if not isinstance(recipes, list):
            raise ValueError("菜谱候选对象必须包含 recipes 数组")

        estimated_total_value = price_summary.get(
            "estimated_total", price_summary.get("total")
        )
        if estimated_total_value is None:
            raise ValueError("价格汇总对象必须包含 estimated_total 或 total 字段")

        menu = [recipe["name"] for recipe in recipes]
        estimated_total = int(estimated_total_value)
        return json.dumps(
            {
                "menu": menu,
                "shopping_list": [
                    "鲈鱼1条",
                    "姜1块",
                    "葱1把",
                    "西兰花1颗",
                    "蒜1头",
                    "番茄3个",
                    "白玉菇1盒",
                ],
                "estimated_total": estimated_total,
                "budget": budget,
                "within_budget": estimated_total <= budget,
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="recipe_candidates",
                type="object",
                description="包含 recipes 数组的菜谱候选 JSON 对象",
                required=True,
            ),
            ToolParams(
                name="price_summary",
                type="object",
                description=(
                    "食材价格汇总 JSON 对象，其中总价字段为 estimated_total 或 total"
                ),
                required=True,
            ),
            ToolParams(
                name="budget",
                type="integer",
                description="晚餐总预算，单位为元",
                required=True,
            ),
        ]


class TracingPlanner(Planner):
    """打印 Planner 生成的完整计划。"""

    def plan(self, user_question: str, history: list[Message]) -> list[str]:
        print("\n========== Planner 开始规划 ==========")
        plan = super().plan(user_question, history)
        print("\n========== Planner 生成计划 ==========")
        for index, step in enumerate(plan, start=1):
            print(f"{index}. {step}")
        return plan


class TracingToolManager(ToolManager):
    """打印 Executor 每次工具调用的参数和结果。"""

    def execute_tool_calls(self, tool_calls: list[Any]) -> list[ToolResult]:
        results = super().execute_tool_calls(tool_calls)
        for result in results:
            arguments = json.dumps(result.arguments or {}, ensure_ascii=False)
            print("\n---------- Executor 工具执行 ----------")
            print(f"Action: {result.name}")
            print(f"Action Input: {arguments}")
            print(f"Status: {result.status}")
            print(f"Observation: {result.content}")
        return results


class TracingExecutor(Executor):
    """按顺序打印每个计划步骤及其执行结果。"""

    def execute(self, question: str, plan: list[str]) -> str:
        history: list[dict[str, str]] = []
        result = ""

        for index, step in enumerate(plan, start=1):
            print(f"\n========== Executor 步骤 {index}/{len(plan)} ==========")
            print(f"当前步骤: {step}")

            # 使用父类的上下文构建和 ReAct 单步执行逻辑，确保示例行为与正式实现一致。
            context = self._build_context(question, plan, history, step)
            result = self._execute_step(context)
            history.append({"step": step, "result": result})

            # 最后一步的结果就是 Agent 最终答案，由主流程统一输出，避免重复打印。
            if index < len(plan):
                print(f"\n步骤 {index} 执行结果:")
                print(result)

        return result


class TracingPlanAndSolveAgent(PlanAndSolveAgent):
    """为演示替换带过程输出的 Planner、Executor 和 ToolManager。"""

    def __init__(self, llm: SproutLLM) -> None:
        super().__init__(llm)
        # 三个组件必须共享同一个工具管理器，后续 add_tool 才能被 Executor 使用。
        self.tools = TracingToolManager()
        self.planner = TracingPlanner(llm, self.tools)
        self.executor = TracingExecutor(llm, self.tools)


def main() -> None:
    """运行一个展示完整规划和逐步执行过程的 Plan-and-Solve 示例。"""
    agent = TracingPlanAndSolveAgent(TracingSproutLLM())
    agent.add_tool(SearchRecipesTool())
    agent.add_tool(QueryIngredientPricesTool())
    agent.add_tool(BuildDinnerPlanTool())

    # 命令行参数可覆盖默认任务，方便观察不同问题下的规划和执行过程。
    user_query = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_QUERY
    print("========== 用户任务 ==========")
    print(user_query)

    answer = agent.run(user_query)
    print("\n========== Agent 最终返回 ==========")
    print(answer)


if __name__ == "__main__":
    main()
