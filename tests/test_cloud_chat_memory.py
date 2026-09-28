import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.api.sentence_generation.sentence_service import (
    CHAT_SYSTEM_PROMPT,
    ChatMemory,
    CloudSentenceService,
)


def make_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class CloudSentenceServiceChatMemoryTests(unittest.TestCase):
    def make_service(self) -> CloudSentenceService:
        service = object.__new__(CloudSentenceService)
        service.chat_memory = ChatMemory()
        service.MODEL_NAME = "test-model"
        service.client = MagicMock()
        return service

    def test_successful_turn_persists_user_and_assistant_messages(self):
        service = self.make_service()
        service.client.chat.completions.create.return_value = make_response("Hello back")

        result = service.chat("Hello", session_id="session-1")

        self.assertEqual(result, "Hello back")
        self.assertEqual(
            service.chat_memory.get_history("session-1"),
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello back"},
            ],
        )

    def test_second_turn_sends_prior_history_before_current_user_message(self):
        service = self.make_service()
        service.client.chat.completions.create.side_effect = [
            make_response("First reply"),
            make_response("Second reply"),
        ]

        service.chat("First message", session_id="session-1")
        service.chat("Second message", session_id="session-1")

        second_messages = service.client.chat.completions.create.call_args_list[1].kwargs[
            "messages"
        ]
        self.assertEqual(
            second_messages,
            [
                {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First reply"},
                {"role": "user", "content": "Second message"},
            ],
        )

    def test_session_id_none_remains_stateless(self):
        service = self.make_service()
        service.client.chat.completions.create.return_value = make_response("No memory")

        service.chat("Hello", session_id=None)

        self.assertEqual(service.chat_memory.get_session_count(), 0)

    def test_provider_failure_does_not_append_partial_turn(self):
        service = self.make_service()
        service.client.chat.completions.create.side_effect = RuntimeError("provider down")

        with self.assertRaisesRegex(RuntimeError, "provider down"):
            service.chat("Hello", session_id="session-1")

        self.assertEqual(service.chat_memory.get_history("session-1"), [])

    def test_later_provider_failure_preserves_existing_history_exactly(self):
        service = self.make_service()
        service.client.chat.completions.create.side_effect = [
            make_response("First reply"),
            RuntimeError("provider down"),
        ]

        service.chat("First message", session_id="session-1")
        history_before_failure = service.chat_memory.get_history("session-1")

        with self.assertRaisesRegex(RuntimeError, "provider down"):
            service.chat("Second message", session_id="session-1")

        self.assertEqual(
            service.chat_memory.get_history("session-1"),
            history_before_failure,
        )


if __name__ == "__main__":
    unittest.main()
