from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QVBoxLayout, QWidget

from .. import scripting
from ..model import BeamzConnector
from ..serialization import _pyrepr

PROMPT = ">>> "

WELCOME = (
    "BEAMZ script console — this runs against the SAME live objects the GUI\n"
    "edits, via the shared namespace. `bz` is the beamz package itself.\n"
    "GUI actions (toolbar Add, Properties Apply, Delete) are also echoed\n"
    "here as the equivalent Python, so this doubles as a running record of\n"
    "everything you've done — this app is meant to be nothing more than an\n"
    "interface onto beamz itself.\n"
    "\n"
    "  wg = bz.Rectangle(position=(0,0,0), width=2e-6, height=0.5e-6, depth=0.22e-6)\n"
    "    -> creates a structure AND adds it to the Object Tree / canvas\n"
    "  wg = wg.updated_copy(width=3e-6)\n"
    "    -> structures are immutable, so an edit is a rebind, not a mutation;\n"
    "       the GUI updates as soon as the line runs\n"
    "  del wg\n"
    "    -> removes it from the scene\n"
)


class ScriptConsole(QWidget):
    """Bottom dock: a REPL-style console executing Python against
    `connector.namespace`. After every command, `connector.sync_from_script`
    reconciles whatever the script did (created/edited/deleted a structure)
    back into the tracked scene, so the Object Tree / canvas / property
    editor update exactly as if the same change had been made through the
    GUI.

    Also ECHOES every GUI-driven change (toolbar Add, Properties Apply,
    Delete, ...) as the equivalent Python line — this app is meant to be
    "nothing more than an interface to beamz", so every action should have
    a visible, copy-pasteable Python equivalent. Implemented by listening
    to the connector's own structure_added/changed/removed signals rather
    than touching every call site in main_window.py/property_editor.py:
    those signals fire identically regardless of which UI path triggered
    them, so this is the one place that needs to know about it.
    """

    def __init__(self, connector: BeamzConnector, parent=None) -> None:
        super().__init__(parent)
        self.connector = connector
        self._history: list[str] = []
        self._history_index = 0
        self._sid_names: dict[str, str] = {}  # survives past removal, for "del <name>" echo
        self._suppress_echo = False  # True while a console command's OWN
        # execution is in flight, so its resulting signals don't echo a
        # duplicate of what the user just typed themselves.

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(mono)
        self.output.setPlainText(WELCOME)
        layout.addWidget(self.output, stretch=1)

        self.input = QLineEdit()
        self.input.setFont(mono)
        self.input.setPlaceholderText("Python — same live session as the GUI")
        self.input.installEventFilter(self)
        layout.addWidget(self.input)

        self.input.returnPressed.connect(self._run_current_line)

        connector.structure_added.connect(self._echo_added)
        connector.structure_changed.connect(self._echo_changed)
        connector.structure_removed.connect(self._echo_removed)

    # ------------------------------------------------------------------ #
    def _append(self, text: str) -> None:
        if text:
            self.output.appendPlainText(text)
            self.output.moveCursor(QTextCursor.MoveOperation.End)

    def _run_current_line(self) -> None:
        code = self.input.text()
        self.input.clear()
        if not code.strip():
            return

        self._history.append(code)
        self._history_index = len(self._history)
        self._append(PROMPT + code)

        self._suppress_echo = True
        try:
            output, is_error = scripting.run_snippet(code, self.connector.namespace)
            if is_error:
                self._append(output.rstrip("\n"))
                return  # don't attempt to sync state after a failed command
            if output:
                self._append(output.rstrip("\n"))
            affected = self.connector.sync_from_script()
        finally:
            self._suppress_echo = False

        if affected:
            self._append(f"# synced {len(affected)} structure(s) with GUI: {', '.join(affected)}")

    # ------------------------------------------------------------------ #
    # GUI-action echoing — see class docstring.
    # ------------------------------------------------------------------ #
    def _echo_added(self, sid: str) -> None:
        so = self.connector.get(sid)
        self._sid_names[sid] = so.name
        if self._suppress_echo:
            return
        self._append(self._format_call(so))

    def _echo_changed(self, sid: str) -> None:
        so = self.connector.get(sid)
        self._sid_names[sid] = so.name
        if self._suppress_echo:
            return
        self._append(self._format_call(so, updated=True))

    def _echo_removed(self, sid: str) -> None:
        name = self._sid_names.pop(sid, sid)
        if self._suppress_echo:
            return
        self._append(f"del {name}")

    def _format_call(self, so, updated: bool = False) -> str:
        if so.category == "region":
            return f"# {so.name}: FDTD Simulation Region updated (GUI-only concept, not a beamz call)"
        kwargs = ", ".join(f"{k}={_pyrepr(v)}" for k, v in so.recipe.items())
        suffix = "  # updated via GUI" if updated else "  # added via GUI"
        return f"{so.name} = bz.{so.class_name}({kwargs}){suffix}"

    # ------------------------------------------------------------------ #
    # Command history (Up/Down), installed on the QLineEdit via event
    # filter so plain QLineEdit key handling doesn't swallow the arrows.
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override naming)
        if obj is self.input and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Up:
                self._navigate_history(-1)
                return True
            if event.key() == Qt.Key.Key_Down:
                self._navigate_history(1)
                return True
        return super().eventFilter(obj, event)

    def _navigate_history(self, delta: int) -> None:
        if not self._history:
            return
        self._history_index = max(0, min(len(self._history), self._history_index + delta))
        if self._history_index == len(self._history):
            self.input.clear()
        else:
            self.input.setText(self._history[self._history_index])