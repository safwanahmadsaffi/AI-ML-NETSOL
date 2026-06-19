from __future__ import annotations


class SessionMemoryStore:
    def __init__(self) -> None:
        # Maps session_id (str) -> list of tuples (user_message, bot_message)
        self._history: dict[str, list[tuple[str, str]]] = {}

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        if session_id not in self._history:
            self._history[session_id] = []
        return self._history[session_id]

    def add_message(self, session_id: str, user_message: str, bot_message: str) -> None:
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append((user_message, bot_message))

    def reset_session(self, session_id: str) -> None:
        if session_id in self._history:
            self._history[session_id] = []


# Singleton instance
session_store = SessionMemoryStore()
