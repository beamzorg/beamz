"""
Whole-scene-snapshot undo/redo. Coarse-grained by design: a "checkpoint"
captures the ENTIRE scene state (every structure/source/monitor/region),
not a per-field diff. Simpler and more robust than a fine-grained command
pattern, and cheap enough to be practical — recipes are small plain
dicts, not full beamz objects or large arrays (aside from the occasional
monitor `freqs` array or source `signal`, still tiny next to actual
simulation field data).

GUI code is responsible for calling `checkpoint()` right BEFORE a
discrete, user-visible mutating action (an Apply click, a Delete, an
Add-from-toolbar) — not on every keystroke while a value is still being
typed. The Property Editor's own staged working-copy already makes this
natural: nothing reaches the connector until Apply, so there's exactly
one checkpoint-worthy moment per edit regardless.
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
        """Call BEFORE a mutating action. Pushes the CURRENT (pre-change)
        state onto the undo stack and clears the redo stack — making a
        fresh change invalidates any previously-undone redo history, the
        usual rule in every app that has both.
        """
        self._undo_stack.append(self.connector.snapshot())
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
        state = self._undo_stack.pop()
        self.connector.restore(state)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.connector.snapshot())
        state = self._redo_stack.pop()
        self.connector.restore(state)
        return True

    def clear(self) -> None:
        """Called after a full scene replacement that shouldn't itself be
        undoable back to whatever was open before it — loading a
        workspace or importing a script is "open a different file", not
        an editable action within the current one.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()