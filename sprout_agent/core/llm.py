import os
from pathlib import Path
from typing import Any, Literal, overload

from dotenv import load_dotenv
from openai import OpenAI, Stream, omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from sprout_agent.core.message import Message


class SproutLLM:
    """基于 OpenAI Chat Completions 格式的大模型客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        """初始化 OpenAI 客户端，空参数从 .env 读取。

        Args:
            api_key: 大模型 API Key，为空时读取 ``LLM_API_KEY``。
            model: 模型名称，为空时读取 ``LLM_MODEL``。
            base_url: API 地址，为空时读取 ``LLM_BASE_URL``。
            env_file: 自定义 .env 文件路径。默认读取项目根目录下的 .env。

        Raises:
            ValueError: 必需的环境变量缺失时抛出。
        """
        api_key = (api_key or "").strip()
        model = (model or "").strip()
        base_url = (base_url or "").strip()

        if not api_key or not model or not base_url:
            # 默认使用项目根目录，避免调用方从其他工作目录启动时加载不到配置。
            dotenv_path = (
                Path(env_file)
                if env_file is not None
                else Path(__file__).resolve().parents[2] / ".env"
            )
            # 已存在的系统环境变量优先级更高，便于部署环境安全注入配置。
            load_dotenv(dotenv_path=dotenv_path, override=False)

        # 初始化参数优先；只有对应参数为空时才回退到环境配置。
        self.api_key = api_key or os.getenv("LLM_API_KEY", "").strip()
        self.model = model or os.getenv("LLM_MODEL", "").strip()
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "").strip()

        required_config = {
            "LLM_API_KEY": self.api_key,
            "LLM_MODEL": self.model,
            "LLM_BASE_URL": self.base_url,
        }
        missing_config = [name for name, value in required_config.items() if not value]
        if missing_config:
            missing_names = ", ".join(missing_config)
            raise ValueError(f"缺少大模型配置：{missing_names}")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @overload
    def chat(
        self,
        messages: list[Message],
        *,
        stream: Literal[False] = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion: ...

    @overload
    def chat(
        self,
        messages: list[Message],
        *,
        stream: Literal[True],
        tools: list[dict[str, Any]] | None = None,
    ) -> Stream[ChatCompletionChunk]: ...

    def chat(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """调用大模型。

        Args:
            messages: 项目自定义的消息列表。
            stream: 为 ``True`` 时返回流式响应迭代器。
            tools: OpenAI Function Calling 格式的工具定义。

        Returns:
            非流式调用返回 ``ChatCompletion``；流式调用返回
            ``Stream[ChatCompletionChunk]``。
        """
        openai_messages = [message.to_dict() for message in messages]

        return self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            stream=stream,
            tools=tools if tools else omit,
            extra_body={"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    llm = SproutLLM()
    example_messages = [
        Message(role="system", content="你是一个简洁、准确的智能助手。"),
        Message(role="user", content="介绍一下 LLM 的定义和应用场景。"),
    ]

    response = llm.chat(example_messages)
    print("非流式响应：")
    print(response.choices[0].message.content)

    print("\n流式响应：")
    stream = llm.chat(example_messages, stream=True)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()
