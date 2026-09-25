from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem

from .. import materials
from ..model import BeamzConnector

SID_ROLE = Qt.ItemDataRole.UserRole
DISABLED_TEXT_COLOR = QColor("#888888")

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
        # Default QTreeWidget selection mode is single-item — Ctrl/Shift
        # click (and Shift-click range-select) for bulk Delete/Enable-
        # Disable needs Extended explicitly.
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._items: dict[str, QTreeWidgetItem] = {}
        self._group_items: dict[str, QTreeWidgetItem] = {}
        self._suppress_selection_signal = False
        self._suppress_check_signal = False

        for category, label in CATEGORY_GROUPS:
            group = QTreeWidgetItem([label])
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)  # header row, not selectable
            self.addTopLevelItem(group)
            group.setExpanded(True)
            self._group_items[category] = group

        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemChanged.connect(self._on_item_changed)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        connector.structure_added.connect(self._on_added)
        connector.structure_removed.connect(self._on_removed)
        connector.structure_changed.connect(self._on_changed)

    # ------------------------------------------------------------------ #
    def _label_for(self, so) -> str:
        label = f"{so.name}  ({so.class_name})"
        if so.category == "structure":
            # Shows the material's refractive index directly in the tree
            # — a quick, unambiguous way to confirm two structures really
            # DO have different materials (vs. just looking similar in
            # the canvas at a glance/at a given zoom/opacity), since the
            # canvas/3D preview color-code by this same value (see
            # materials.color_for_material).
            n = materials.index_from_material(so.recipe.get("material"))
            label += f"  \u2014 n={n:.3g}"
        return label

    def _on_added(self, sid: str) -> None:
        so = self.connector.get(sid)
        group = self._group_items[so.category]
        item = QTreeWidgetItem([self._label_for(so)])
        item.setData(0, SID_ROLE, sid)
        # Confirmed directly (a standalone Qt test) that setCheckState()
        # on an item NOT YET added to the tree does not fire itemChanged
        # — so this ordering (configure fully, then addChild) is actually
        # safe as-is. Suppressed anyway, cheaply, so it stays safe even
        # if that ordering ever changes (see _on_changed's comment for
        # why an unsuppressed itemChanged here would be a real bug, not
        # just a theoretical one).
        self._suppress_check_signal = True
        try:
            if so.category != "region":
                # The region has no enabled/disabled concept (see
                # connector.set_enabled) — no checkbox for it at all,
                # rather than one that would do nothing.
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if so.enabled else Qt.CheckState.Unchecked)
            group.addChild(item)
        finally:
            self._suppress_check_signal = False
        self._items[sid] = item
        self._apply_enabled_style(item, so.enabled)

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
        # The WHOLE body below runs under suppression, not just the
        # setCheckState() call — confirmed directly (a standalone Qt
        # test) that `item.setText()` on an item ALREADY in the tree
        # fires `itemChanged` too, same as setCheckState() does. Since
        # `_on_changed` fires on EVERY property edit, rename, rotation,
        # or enable/disable toggle (anything that emits structure_changed
        # — not just actual checkbox clicks), leaving setText()
        # unguarded meant _on_item_changed ran for real after every one
        # of those, pushing a bogus extra undo checkpoint each time. That
        # checkpoint captured the ALREADY-CURRENT state (nothing had
        # changed since — the click already happened), so the very next
        # Undo landed on that no-op duplicate first: visibly "delete the
        # object, recreate an identical one" instead of actually undoing
        # anything, with the real undo target one press further back.
        self._suppress_check_signal = True
        try:
            item.setText(0, self._label_for(so))
            # A script rebind can change category (rare, but e.g.
            # reassigning a tracked name to a completely different kind
            # of object) — move the row to the right group if so.
            correct_group = self._group_items[so.category]
            if item.parent() is not correct_group:
                old_parent = item.parent()
                if old_parent is not None:
                    old_parent.removeChild(item)
                correct_group.addChild(item)

            if so.category != "region":
                item.setCheckState(0, Qt.CheckState.Checked if so.enabled else Qt.CheckState.Unchecked)
        finally:
            self._suppress_check_signal = False
        self._apply_enabled_style(item, so.enabled)

    def _apply_enabled_style(self, item: QTreeWidgetItem, enabled: bool) -> None:
        item.setForeground(0, QBrush(DISABLED_TEXT_COLOR) if not enabled else QBrush())

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._suppress_check_signal or column != 0:
            return
        sid = item.data(0, SID_ROLE)
        if sid is None or not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return
        self.window()._checkpoint()
        self.connector.set_enabled(sid, item.checkState(0) == Qt.CheckState.Checked)

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

        sids = self.selected_sids()
        if len(sids) > 1 and sid in sids:
            self._multi_context_menu(pos, sids)
            return

        so = self.connector.get(sid)
        menu = QMenu(self)
        if so.category == "monitor":
            menu.addAction("View Results...", lambda: self.view_results_requested.emit(sid))
            menu.addSeparator()

        menu.addAction("Rename...", lambda: self.window()._rename_sid(sid))
        if so.category != "region":
            menu.addAction("Duplicate", lambda: self.window()._duplicate_sid(sid))
            menu.addAction("Copy", lambda: self.window()._copy_sids([sid]))
            menu.addAction(
                "Disable" if so.enabled else "Enable",
                lambda: self._set_enabled_checkpointed([sid], not so.enabled),
            )
        menu.addSeparator()

        def _delete_action():
            # Checkpoint BEFORE the mutation — see undo.py's checkpoint()
            # docstring for why the ordering matters.
            self.window()._checkpoint()
            self.connector.remove_structure(sid)

        menu.addAction("Delete", _delete_action)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _multi_context_menu(self, pos, sids: list[str]) -> None:
        # Rename/Duplicate/View Results don't generalize cleanly to a
        # batch (which name would a batch rename even use?) — multi-
        # select gets exactly the two operations that DO: bulk
        # enable/disable and bulk delete.
        non_region_sids = [sid for sid in sids if self.connector.get(sid).category != "region"]
        menu = QMenu(self)
        if non_region_sids:
            menu.addAction("Copy", lambda: self.window()._copy_sids(non_region_sids))
            menu.addAction("Enable", lambda: self._set_enabled_checkpointed(non_region_sids, True))
            menu.addAction("Disable", lambda: self._set_enabled_checkpointed(non_region_sids, False))
            menu.addSeparator()

        def _delete_all():
            self.window()._checkpoint()
            for sid in sids:
                self.connector.remove_structure(sid)

        menu.addAction(f"Delete {len(sids)} objects", _delete_all)
        menu.exec(self.viewport().mapToGlobal(pos))

    def _set_enabled_checkpointed(self, sids: list[str], enabled: bool) -> None:
        self.window()._checkpoint()
        for sid in sids:
            self.connector.set_enabled(sid, enabled)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sids = self.selected_sids()
            if sids:
                # One checkpoint for the whole batch — a single Delete
                # press (even across several selected rows) is one
                # discrete user action, undo-able as one.
                self.window()._checkpoint()
                for sid in sids:
                    self.connector.remove_structure(sid)
                return
        super().keyPressEvent(event)

    def selected_sids(self) -> list[str]:
        return [item.data(0, SID_ROLE) for item in self.selectedItems() if item.data(0, SID_ROLE) is not None]

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