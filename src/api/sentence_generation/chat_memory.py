"""Bounded in-memory conversation history for sentence-generation services."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Dict, List, Optional

from src.api.config import config

logger = logging.getLogger(__name__)


class ChatMemory:
    """Thread-safe per-session chat history with size and inactivity bounds."""

    def __init__(
        self,
        max_messages: Optional[int] = None,
        ttl_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_messages = (
            config.CHAT_MEMORY_MAX_MESSAGES if max_messages is None else max_messages
        )
        self.ttl_seconds = (
            config.CHAT_MEMORY_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        )
        if self.max_messages < 2:
            raise ValueError("CHAT_MEMORY_MAX_MESSAGES must be at least 2")
        if self.ttl_seconds <= 0:
            raise ValueError("CHAT_MEMORY_TTL_SECONDS must be greater than 0")

        self._clock = clock
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._last_activity: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _cleanup_expired_locked(self, now: float) -> int:
        expired = [
            session_id
            for session_id, last_activity in self._last_activity.items()
            if now - last_activity >= self.ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
            self._last_activity.pop(session_id, None)
        return len(expired)

    def _trim_history_locked(self, history: List[Dict[str, str]]) -> None:
        while len(history) > self.max_messages:
            if (
                len(history) >= 2
                and history[0].get("role") == "user"
                and history[1].get("role") == "assistant"
            ):
                del history[:2]
            else:
                del history[0]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        now = self._clock()
        with self._lock:
            self._cleanup_expired_locked(now)
            history = self._sessions.get(session_id)
            if history is None:
                return []
            self._last_activity[session_id] = now
            return [message.copy() for message in history]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = self._clock()
        with self._lock:
            self._cleanup_expired_locked(now)
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            self._trim_history_locked(history)
            self._last_activity[session_id] = now

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_activity.pop(session_id, None)
        logger.info("Chat memory cleared for session: %s", session_id)

    def cleanup_expired(self) -> int:
        """Remove inactive sessions and return the number removed."""
        with self._lock:
            return self._cleanup_expired_locked(self._clock())

    def has_session(self, session_id: str) -> bool:
        now = self._clock()
        with self._lock:
            self._cleanup_expired_locked(now)
            exists = session_id in self._sessions
            if exists:
                self._last_activity[session_id] = now
            return exists

    def get_session_count(self) -> int:
        with self._lock:
            self._cleanup_expired_locked(self._clock())
            return len(self._sessions)

    def get_message_count(self, session_id: str) -> int:
        now = self._clock()
        with self._lock:
            self._cleanup_expired_locked(now)
            history = self._sessions.get(session_id)
            if history is None:
                return 0
            self._last_activity[session_id] = now
            return len(history)
