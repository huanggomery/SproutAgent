import json
import sys
from typing import Any

from sprout_agent.agents.react_agent import ReActAgent
from sprout_agent.core.llm import SproutLLM
from sprout_agent.tools.base import Tool, ToolParams, ToolResult


DEFAULT_USER_QUERY = (
    "请为两个人制定一个明天在杭州、偏重文化体验的半日活动方案，"
    "并给出天气提示、具体行程、选择理由和预算明细。"
)


class QueryWeatherTool(Tool):
    """查询演示用天气数据。"""

    def __init__(self) -> None:
        super().__init__(
            name="query_weather",
            description="查询指定城市和日期的天气预报，为行程规划提供天气信息。",
        )

    def execute(self, params: dict) -> str:
        city = str(params["city"])
        date = str(params["date"])
        # 示例使用固定数据，便于重复运行并观察稳定的多轮工具调用过程。
        return json.dumps(
            {
                "city": city,
                "date": date,
                "weather": "小雨，12～17℃",
                "suggestion": "适合安排室内活动，并携带雨具",
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="city",
                type="string",
                description="需要查询天气的城市",
                required=True,
            ),
            ToolParams(
                name="date",
                type="string",
                description="需要查询天气的日期，例如明天",
                required=True,
            ),
        ]


class SearchActivitiesTool(Tool):
    """搜索演示用活动候选。"""

    def __init__(self) -> None:
        super().__init__(
            name="search_activities",
            description=(
                "搜索指定城市符合主题的活动候选，返回地点、类型、建议游玩时间和费用。"
                "本工具只负责搜索，不需要天气查询结果。"
            ),
        )

    def execute(self, params: dict) -> str:
        city = str(params["city"])
        theme = str(params["theme"])
        return json.dumps(
            {
                "city": city,
                "theme": theme,
                "candidates": [
                    {
                        "name": "中国丝绸博物馆",
                        "type": "室内",
                        "duration_hours": 2.5,
                        "ticket_per_person": 0,
                    },
                    {
                        "name": "西湖环湖骑行",
                        "type": "室外",
                        "duration_hours": 3,
                        "ticket_per_person": 40,
                    },
                ],
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="city",
                type="string",
                description="需要搜索活动的城市",
                required=True,
            ),
            ToolParams(
                name="theme",
                type="string",
                description="活动主题或偏好，例如文化体验",
                required=True,
            ),
        ]


class BuildActivityPlanTool(Tool):
    """根据查询结果制定演示用行程和预算。"""

    def __init__(self) -> None:
        super().__init__(
            name="build_activity_plan",
            description=(
                "结合已查询到的天气预报、活动候选和参加人数，制定半日行程并估算预算。"
                "天气和候选活动应来自相关查询工具的实际结果。"
            ),
        )

    def execute(self, params: dict) -> str:
        weather_summary = str(params["weather_summary"])
        activity_candidates = str(params["activity_candidates"])
        people = int(params["people"])
        if people <= 0:
            raise ValueError("人数必须大于 0")

        # 规划工具消费前序查询的自然语言结果，而不是依赖人为生成的凭证。
        indoor_activity = (
            "中国丝绸博物馆"
            if "中国丝绸博物馆" in activity_candidates
            else "室内文化场馆"
        )
        ticket_per_person = 0 if indoor_activity == "中国丝绸博物馆" else 40
        transport_per_person = 30
        meal_per_person = 50
        contingency_per_person = 30
        total = (
            ticket_per_person
            + transport_per_person
            + meal_per_person
            + contingency_per_person
        ) * people
        return json.dumps(
            {
                "weather_basis": weather_summary,
                "selected_activity": indoor_activity,
                "schedule": "09:30 到达，09:30～12:00 参观，12:00 附近简餐",
                "reason": "优先选择不受降雨影响且适合半日游览的室内文化活动",
                "budget": {
                    "people": people,
                    "transport": transport_per_person * people,
                    "ticket": ticket_per_person * people,
                    "meal": meal_per_person * people,
                    "contingency": contingency_per_person * people,
                    "total": total,
                    "currency": "CNY",
                },
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="weather_summary",
                type="string",
                description="天气查询工具返回的天气及出行建议",
                required=True,
            ),
            ToolParams(
                name="activity_candidates",
                type="string",
                description="活动搜索工具返回的候选活动信息",
                required=True,
            ),
            ToolParams(
                name="people",
                type="integer",
                description="参加活动的人数",
                required=True,
            ),
        ]


class TracingSproutLLM(SproutLLM):
    """打印模型的原始工具选择，包含被 Agent 丢弃的无效响应。"""

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
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        print(f"\n---------- 模型请求 {self.request_number} 的原始决策 ----------")
        if not tool_calls:
            print("Tool Calls: （无）")
            print(f"Content: {assistant_message.content or '（空）'}")
            return response

        for tool_call in tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
        if assistant_message.content:
            print(f"Content: {assistant_message.content}")
        return response


class TracingReActAgent(ReActAgent):
    """在终端打印每轮 ReAct 工具执行轨迹。"""

    def __init__(self, llm: SproutLLM) -> None:
        super().__init__(llm)
        self.round_number = 0

    def execute_tool_calls(self, tool_calls: list[Any]) -> list[ToolResult]:
        results = super().execute_tool_calls(tool_calls)
        self.round_number += 1
        print(f"\n========== ReAct 第 {self.round_number} 轮 ==========")

        for result in results:
            action_input = json.dumps(result.arguments or {}, ensure_ascii=False)
            print(f"Action: {result.name}")
            print(f"Action Input: {action_input}")
            print(f"Status: {result.status}")
            print(f"Observation: {result.content}")

        return results


def main() -> None:
    """运行一个包含前后依赖的多轮 ReAct 示例。"""
    agent = TracingReActAgent(TracingSproutLLM())
    agent.add_tool(QueryWeatherTool())
    agent.add_tool(SearchActivitiesTool())
    agent.add_tool(BuildActivityPlanTool())

    # 支持直接在命令行中输入自然语言问题；空参数保持示例可直接运行。
    user_query = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_QUERY

    print("========== 用户任务 ==========")
    print(user_query)
    answer = agent.run(user_query, max_steps=8)
    print("\n========== Agent 最终返回 ==========")
    print(answer)


if __name__ == "__main__":
    main()
