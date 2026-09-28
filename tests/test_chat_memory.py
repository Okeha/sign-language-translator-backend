from src.api.sentence_generation.chat_memory import ChatMemory
from src.api.sentence_generation.factory import _with_bounded_chat_memory


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def add_turn(memory: ChatMemory, session_id: str, user: str, assistant: str) -> None:
    memory.add_message(session_id, "user", user)
    memory.add_message(session_id, "assistant", assistant)


def test_history_is_bounded_by_complete_turns() -> None:
    memory = ChatMemory(max_messages=4, ttl_seconds=60)

    add_turn(memory, "s", "u1", "a1")
    add_turn(memory, "s", "u2", "a2")
    add_turn(memory, "s", "u3", "a3")

    assert memory.get_history("s") == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]


def test_pending_user_message_does_not_orphan_the_previous_turn() -> None:
    memory = ChatMemory(max_messages=4, ttl_seconds=60)

    add_turn(memory, "s", "u1", "a1")
    add_turn(memory, "s", "u2", "a2")
    memory.add_message("s", "user", "u3")

    assert memory.get_history("s") == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]


def test_inactive_session_expires() -> None:
    clock = FakeClock()
    memory = ChatMemory(max_messages=4, ttl_seconds=10, clock=clock)

    add_turn(memory, "stale", "u", "a")
    clock.advance(10)

    assert memory.get_history("stale") == []
    assert not memory.has_session("stale")


def test_access_refreshes_session_activity() -> None:
    clock = FakeClock()
    memory = ChatMemory(max_messages=4, ttl_seconds=10, clock=clock)

    add_turn(memory, "active", "u", "a")
    clock.advance(6)
    assert memory.get_history("active")

    clock.advance(6)
    assert memory.has_session("active")


def test_cleanup_expires_only_inactive_sessions() -> None:
    clock = FakeClock()
    memory = ChatMemory(max_messages=4, ttl_seconds=10, clock=clock)

    add_turn(memory, "old", "u", "a")
    clock.advance(6)
    add_turn(memory, "new", "u", "a")
    clock.advance(5)

    assert memory.cleanup_expired() == 1
    assert not memory.has_session("old")
    assert memory.has_session("new")


def test_clear_session_removes_history_and_activity_metadata() -> None:
    memory = ChatMemory(max_messages=4, ttl_seconds=60)
    add_turn(memory, "s", "u", "a")

    memory.clear_session("s")

    assert memory.get_history("s") == []
    assert memory.get_session_count() == 0


def test_returned_history_is_a_copy() -> None:
    memory = ChatMemory(max_messages=4, ttl_seconds=60)
    add_turn(memory, "s", "u", "a")

    history = memory.get_history("s")
    history[0]["content"] = "mutated"

    assert memory.get_history("s")[0]["content"] == "u"


def test_factory_replaces_backend_specific_memory_with_shared_bounded_memory() -> None:
    class DummyService:
        chat_memory = object()

    service = _with_bounded_chat_memory(DummyService())

    assert isinstance(service.chat_memory, ChatMemory)


def test_invalid_limits_are_rejected() -> None:
    for max_messages, ttl_seconds in [(1, 60), (4, 0), (4, -1)]:
        try:
            ChatMemory(max_messages=max_messages, ttl_seconds=ttl_seconds)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ChatMemory limits must raise ValueError")
