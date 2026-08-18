import unittest

from sprout_agent.context.history import HistoryManager
from sprout_agent.core.message import Message


class TestHistoryManager(unittest.TestCase):

    def setUp(self) -> None:
        self.history = HistoryManager()

    def _create_messages(self, count: int) -> list[Message]:
        messages = []
        for i in range(count):
            messages.append(Message(role="user", content=f"message {i}"))
        return messages

    def test_append(self) -> None:
        msg1 = Message(role="user", content="hello")
        msg2 = Message(role="assistant", content="hi there")
        
        self.history.append(msg1)
        self.history.append(msg2)
        
        self.assertEqual(len(self.history._history), 2)
        self.assertEqual(len(self.history._archive_history), 2)
        self.assertIs(self.history._history[0], msg1)
        self.assertIs(self.history._history[1], msg2)
        self.assertIs(self.history._archive_history[0], msg1)
        self.assertIs(self.history._archive_history[1], msg2)

    def test_clear(self) -> None:
        messages = self._create_messages(5)
        for msg in messages:
            self.history.append(msg)
        
        self.assertEqual(len(self.history._history), 5)
        self.assertEqual(len(self.history._archive_history), 5)
        
        self.history.clear()
        
        self.assertEqual(len(self.history._history), 0)
        self.assertEqual(len(self.history._archive_history), 0)

    def test_compress_keep_all_when_fewer_messages(self) -> None:
        messages = self._create_messages(3)
        for msg in messages:
            self.history.append(msg)
        
        original_history = list(self.history._history)
        self.history.compress(keep_count=5, summary_text="summary text")
        
        self.assertEqual(len(self.history._history), 3)
        self.assertEqual(self.history._history, original_history)
        self.assertEqual(len(self.history._archive_history), 3)

    def test_compress_keep_all_when_equal_messages(self) -> None:
        messages = self._create_messages(5)
        for msg in messages:
            self.history.append(msg)
        
        original_history = list(self.history._history)
        self.history.compress(keep_count=5, summary_text="summary text")
        
        self.assertEqual(len(self.history._history), 5)
        self.assertEqual(self.history._history, original_history)

    def test_compress_with_fewer_keep_count(self) -> None:
        messages = self._create_messages(10)
        for msg in messages:
            self.history.append(msg)
        
        summary_text = "这是前 6 条消息的总结"
        self.history.compress(keep_count=4, summary_text=summary_text)
        
        self.assertEqual(len(self.history._history), 5)
        
        summary_msg = self.history._history[0]
        self.assertEqual(summary_msg.role, "summary")
        self.assertEqual(summary_msg.content, summary_text)
        
        recent_messages = self.history._history[1:]
        self.assertEqual(len(recent_messages), 4)
        self.assertIs(recent_messages[0], messages[6])
        self.assertIs(recent_messages[1], messages[7])
        self.assertIs(recent_messages[2], messages[8])
        self.assertIs(recent_messages[3], messages[9])
        
        self.assertEqual(len(self.history._archive_history), 10)
        for i, msg in enumerate(messages):
            self.assertIs(self.history._archive_history[i], msg)

    def test_compress_keep_zero(self) -> None:
        messages = self._create_messages(5)
        for msg in messages:
            self.history.append(msg)
        
        summary_text = "全部消息的总结"
        self.history.compress(keep_count=0, summary_text=summary_text)
        
        self.assertEqual(len(self.history._history), 1)
        self.assertEqual(self.history._history[0].role, "summary")
        self.assertEqual(self.history._history[0].content, summary_text)
        
        self.assertEqual(len(self.history._archive_history), 5)

    def test_compress_preserves_message_order(self) -> None:
        roles = ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
        messages = []
        for i, role in enumerate(roles):
            messages.append(Message(role=role, content=f"turn {i}"))
        for msg in messages:
            self.history.append(msg)
        
        self.history.compress(keep_count=2, summary_text="对话总结")
        
        self.assertEqual(self.history._history[0].role, "summary")
        self.assertEqual(self.history._history[1].role, "user")
        self.assertEqual(self.history._history[1].content, "turn 6")
        self.assertEqual(self.history._history[2].role, "assistant")
        self.assertEqual(self.history._history[2].content, "turn 7")

    def test_compress_with_system_prompt_keeps_system_first(self) -> None:
        system_msg = Message(role="system", content="你是一个有用的助手")
        self.history.append(system_msg)
        messages = self._create_messages(5)
        for msg in messages:
            self.history.append(msg)
        
        self.history.compress(keep_count=2, summary_text="对话总结")
        
        self.assertEqual(len(self.history._history), 4)
        self.assertIs(self.history._history[0], system_msg)
        self.assertEqual(self.history._history[0].role, "system")
        self.assertEqual(self.history._history[1].role, "summary")
        self.assertIs(self.history._history[2], messages[3])
        self.assertIs(self.history._history[3], messages[4])

    def test_compress_with_system_prompt_keep_zero(self) -> None:
        system_msg = Message(role="system", content="系统提示词")
        self.history.append(system_msg)
        messages = self._create_messages(3)
        for msg in messages:
            self.history.append(msg)
        
        self.history.compress(keep_count=0, summary_text="全部总结")
        
        self.assertEqual(len(self.history._history), 2)
        self.assertIs(self.history._history[0], system_msg)
        self.assertEqual(self.history._history[1].role, "summary")

    def test_compress_with_system_prompt_no_compress_needed(self) -> None:
        system_msg = Message(role="system", content="系统提示词")
        self.history.append(system_msg)
        messages = self._create_messages(2)
        for msg in messages:
            self.history.append(msg)
        
        original_history = list(self.history._history)
        self.history.compress(keep_count=2, summary_text="总结")
        
        self.assertEqual(len(self.history._history), 3)
        self.assertEqual(self.history._history, original_history)

    def test_compress_with_system_prompt_fewer_messages_than_keep(self) -> None:
        system_msg = Message(role="system", content="系统提示词")
        self.history.append(system_msg)
        messages = self._create_messages(2)
        for msg in messages:
            self.history.append(msg)
        
        original_history = list(self.history._history)
        self.history.compress(keep_count=5, summary_text="总结")
        
        self.assertEqual(self.history._history, original_history)


if __name__ == "__main__":
    unittest.main()
