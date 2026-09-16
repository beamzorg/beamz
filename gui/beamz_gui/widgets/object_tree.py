from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QMenu, QTreeWidget, QTreeWidgetItem

from ..model import BeamzConnector

SID_ROLE = Qt.ItemDataRole.UserRole

# Fixed group order/labels, mirroring the Structures/Sources/Monitors
# grouping used throughout the design discussion (and Lumerical itself).
CATEGORY_GROUPS = (
    ("structure", "Structures"),
    ("source", "Sources"),
    ("monitor", "Monitors"),
    ("region", "Simulation Region"),
)


class ObjectTree(QTreeWidget):
    """Lumerical-style object tree: one row per placeable object, grouped
    under fixed Structures/Sources/Monitors headers. Selecting a row
    selects the object everywhere else in the app.
    """

    selected = Signal(object)  # sid: str | None
    view_results_requested = Signal(str)  # sid — "View Results" clicked on a monitor

    def __init__(self, connector: BeamzConnector, parent=None) -> None:
        super().__init__(parent)
        self.connector = connector
        self.setHeaderLabels(["Scene"])
        self._items: dict[str, QTreeWidgetItem] = {}
        self._group_items: dict[str, QTreeWidgetItem] = {}
        self._suppress_selection_signal = False

        for category, label in CATEGORY_GROUPS:
            group = QTreeWidgetItem([label])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)  # header row, not selectable
            self.addTopLevelItem(group)
            group.setExpanded(True)
            self._group_items[category] = group

        self.itemSelectionChanged.connect(self._on_selection_changed)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        connector.structure_added.connect(self._on_added)
        connector.structure_removed.connect(self._on_removed)
        connector.structure_changed.connect(self._on_changed)

    # ------------------------------------------------------------------ #
    def _on_added(self, sid: str) -> None:
        so = self.connector.get(sid)
        group = self._group_items[so.category]
        item = QTreeWidgetItem([f"{so.name}  ({so.class_name})"])
        item.setData(0, SID_ROLE, sid)
        group.addChild(item)
        self._items[sid] = item

    def _on_removed(self, sid: str) -> None:
        item = self._items.pop(sid, None)
        if item is not None:
            parent = item.parent()
            if parent is not None:
                parent.removeChild(item)

    def _on_changed(self, sid: str) -> None:
        so = self.connector.get(sid)
        item = self._items.get(sid)
        if item is None:
            return
        item.setText(0, f"{so.name}  ({so.class_name})")
        # A script rebind can change category (rare, but e.g. reassigning
        # a tracked name to a completely different kind of object) — move
        # the row to the right group if so.
        correct_group = self._group_items[so.category]
        if item.parent() is not correct_group:
            old_parent = item.parent()
            if old_parent is not None:
                old_parent.removeChild(item)
            correct_group.addChild(item)

    def _on_selection_changed(self) -> None:
        if self._suppress_selection_signal:
            return
        items = self.selectedItems()
        sid = items[0].data(0, SID_ROLE) if items else None
        self.selected.emit(sid)

    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        sid = item.data(0, SID_ROLE) if item is not None else None
        if sid is None:
            return  # clicked a group header or empty space — nothing to do
        so = self.connector.get(sid)
        menu = QMenu(self)
        if so.category == "monitor":
            menu.addAction("View Results...", lambda: self.view_results_requested.emit(sid))
            menu.addSeparator()
        menu.addAction("Delete", lambda: self.connector.remove_structure(sid))
        menu.exec(self.viewport().mapToGlobal(pos))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            items = self.selectedItems()
            if items:
                sid = items[0].data(0, SID_ROLE)
                if sid is not None:
                    self.connector.remove_structure(sid)
                    return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    def select_external(self, sid: Optional[str]) -> None:
        """Programmatically select `sid` (e.g. because the user clicked the
        2D canvas) without re-emitting `selected` and causing a signal loop.
        """
        self._suppress_selection_signal = True
        try:
            self.clearSelection()
            item = self._items.get(sid) if sid else None
            if item is not None:
                item.setSelected(True)
                self.scrollToItem(item)
        finally:
            self._suppress_selection_signal = False