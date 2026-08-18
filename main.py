import json

from sprout_agent.agents.simple_agent import SimpleAgent
from sprout_agent.core.llm import SproutLLM
from sprout_agent.tools.base import Tool, ToolParams


class CalculateSubtotalTool(Tool):
    """根据单价和数量计算商品小计。"""

    def __init__(self) -> None:
        super().__init__(name="calculate_subtotal", description="根据商品单价和数量计算小计")

    def execute(self, params: dict) -> str:
        return str(round(params["unit_price"] * params["quantity"], 2))

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="unit_price",
                type="number",
                description="商品单价，单位为元",
                required=True,
            ),
            ToolParams(
                name="quantity",
                type="integer",
                description="购买数量",
                required=True,
            ),
        ]


class ApplyDiscountTool(Tool):
    """根据折扣比例计算折后金额。"""

    def __init__(self) -> None:
        super().__init__(name="apply_discount", description="根据原金额和折扣比例计算折后金额")

    def execute(self, params: dict) -> str:
        discounted_amount = params["amount"] * (1 - params["discount_rate"])
        return str(round(discounted_amount, 2))

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="amount",
                type="number",
                description="应用折扣前的金额，单位为元",
                required=True,
            ),
            ToolParams(
                name="discount_rate",
                type="number",
                description="折扣比例，例如 0.15 表示优惠 15%",
                required=True,
            ),
        ]


class CalculateCheckoutTool(Tool):
    """根据折后金额和运费规则计算最终应付金额。"""

    def __init__(self) -> None:
        super().__init__(name="calculate_checkout", description="计算运费和最终应付总额")

    def execute(self, params: dict) -> str:
        amount = params["amount"]
        shipping_fee = 0 if amount >= params["free_shipping_threshold"] else params["shipping_fee"]
        return json.dumps(
            {
                "discounted_amount": round(amount, 2),
                "shipping_fee": round(shipping_fee, 2),
                "total": round(amount + shipping_fee, 2),
            },
            ensure_ascii=False,
        )

    def get_params(self) -> list[ToolParams]:
        return [
            ToolParams(
                name="amount",
                type="number",
                description="折后金额，单位为元",
                required=True,
            ),
            ToolParams(
                name="free_shipping_threshold",
                type="number",
                description="包邮门槛，单位为元",
                required=True,
            ),
            ToolParams(
                name="shipping_fee",
                type="number",
                description="未达到包邮门槛时的运费，单位为元",
                required=True,
            ),
        ]


def main() -> None:
    llm = SproutLLM()
    agent = SimpleAgent(llm)
    agent.add_tool(CalculateSubtotalTool())
    agent.add_tool(ApplyDiscountTool())
    agent.add_tool(CalculateCheckoutTool())

    task = (
        "帮我计算一笔订单：商品单价 17.5 元，购买 23 件，会员优惠 15%，"
        "折后满 350 元包邮，否则运费 12 元。请依次调用工具计算商品小计、"
        "折后金额、运费和最终应付金额，并简要列出计算结果。"
    )
    result = agent.run(task)
    print(f"任务：{task}")
    print(f"结果：{result}")

    _, archive_history = agent.history.get_history()
    print(f"历史记录：{archive_history}")

if __name__ == "__main__":
    main()
