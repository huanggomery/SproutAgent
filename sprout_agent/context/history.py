from sprout_agent.core.message import Message


class HistoryManager:
    """对话历史管理器，负责维护对话消息并支持历史压缩。

    维护两个消息列表：
    - ``_history``：当前上下文中使用的消息列表，压缩后会被截断。
    - ``_archive_history``：完整的消息归档，始终保留所有消息用于记录和审计。
    """

    def __init__(self) -> None:
        """初始化历史管理器。"""
        self._history: list[Message] = []
        self._archive_history: list[Message] = []

    def append(self, message: Message) -> None:
        """追加一条消息到历史中。
        消息会同时被添加到当前历史和归档历史中。

        Args:
            message: 要追加的消息对象。
        """
        self._history.append(message)
        self._archive_history.append(message)

    def clear(self) -> None:
        """清空当前历史和归档历史。"""
        self._history.clear()
        self._archive_history.clear()

    def compress(self, keep_count: int, summary_text: str) -> None:
        """压缩对话历史，保留系统提示词和最近的若干条消息。

        压缩规则：
        - 如果第一条消息是系统提示词（role="system"），则始终保留它。
        - 用一条摘要消息（role="summary"）替代被压缩的中间消息。
        - 保留最后 ``keep_count`` 条非系统消息。

        Args:
            keep_count: 要保留的最近消息数量（不包含系统提示词）。
            summary_text: 用于替换被压缩消息的摘要文本。
        """
        summary_message = Message(role="summary", content=summary_text)
        prefix: list[Message] = []
        messages_to_compress = self._history

        if self._history and self._history[0].role == "system":
            prefix = [self._history[0]]
            messages_to_compress = self._history[1:]

        if len(messages_to_compress) <= keep_count:
            return

        if keep_count > 0:
            self._history = prefix + [summary_message] + messages_to_compress[-keep_count:]
        else:
            self._history = prefix + [summary_message]
        
    def get_history(self) -> tuple[list[Message], list[Message]]:
        """获取当前对话历史和归档历史。"""
        return self._history, self._archive_history
