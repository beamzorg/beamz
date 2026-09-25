"""
Whole-scene-snapshot undo/redo. Checkpoints capture the complete scene and
are restored through the connector's snapshot/restore API.
"""
from __future__ import annotations

from typing import Any


class UndoManager:
    def __init__(self, connector, max_depth: int = 50) -> None:
        self.connector = connector
        self._undo_stack: list[list[dict[str, Any]]] = []
        self._redo_stack: list[list[dict[str, Any]]] = []
        self._max_depth = max_depth

    def checkpoint(self) -> None:
        """Save the current scene before a discrete mutating action.

        Call this BEFORE the action runs, not after — the whole point is
        to capture the state undo should return TO. Calling it after the
        mutation instead pushes the post-action state, which makes undo()
        restore the scene to itself (a no-op) rather than back it out.
        """
        self.checkpoint_with(self.connector.snapshot())

    def checkpoint_with(self, snapshot: list[dict[str, Any]]) -> None:
        """Push an ALREADY-CAPTURED snapshot rather than the connector's
        current state. For actions that might fail validation partway
        through (e.g. the Property Editor's Apply, or a script console
        line) — capture the snapshot before attempting the mutation, then
        call this only once success is confirmed, so a rejected/failed
        edit never wastes an undo slot on a no-op and clears the redo
        stack for nothing.
        """
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.connector.snapshot())
        self.connector.restore(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.connector.snapshot())
        self.connector.restore(self._redo_stack.pop())
        return True

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()


class UndoStack(UndoManager):
    """Compatibility name for the main window's existing call sites."""

    def push_snapshot(self) -> None:
        self.checkpoint()