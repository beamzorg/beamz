from __future__ import annotations
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QTabWidget,
    QToolBar,
    QToolButton,
)
from PySide6.QtGui import QAction, QKeySequence
from . import defaults
from . import icons
from . import serialization as ser
from . import simulation_runner as sr
from .model import BeamzConnector, MONITOR_KINDS, SOURCE_KINDS, STRUCTURE_KINDS
from .widgets.canvas_2d import Canvas2D
from .widgets.console import ScriptConsole
from .widgets.figure_dialog import FigureDialog
from .widgets.gds_import_dialog import GdsImportDialog
from .widgets.object_tree import ObjectTree
from .widgets.preview_3d import Preview3D
from .widgets.property_editor import PropertyEditor
from .workspace import WorkspaceManager, entries_to_clipboard_text, entries_from_clipboard_text
from .undo import UndoStack

class MainWindow(QMainWindow):
    """Connector + object tree + 3D/2D views (tabbed, 3D first/default) +
    property editor + script console, wired for round-trip selection and
    live editing across every view. GDS import and simulation run are
    still ahead.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BEAMZ Studio")
        self.resize(1440, 900)

        self.connector = BeamzConnector()

        # Shared state a few call sites below need to exist before
        # _build_toolbar()/_build_menu() reference them (toolbar/menu
        # actions call back into undo_stack/current selection at CLICK
        # time, not at build time, so this ordering is mostly for
        # clarity — but _on_selected below does touch self.rename_action/
        # self.duplicate_action, which _build_menu() creates).
        self.workspace = WorkspaceManager(self.connector)
        self.undo_stack = UndoStack(self.connector)
        self.current_results = {}
        self._current_sid: str | None = None

        # 3D is the default/primary view (Lumerical-style: precise editing
        # happens in the Properties panel, the 3D view is where you
        # actually look at the result) with 2D available as a secondary
        # tab, rather than two permanently-visible docked panes.
        self.preview3d = Preview3D(self.connector)
        self.canvas = Canvas2D(self.connector)
        self.canvas.status_message.connect(self._on_status_message)

        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self.preview3d, "3D View")
        self.view_tabs.addTab(self.canvas, "2D View")
        self.view_tabs.setCurrentIndex(0)
        self.setCentralWidget(self.view_tabs)

        self.tree = ObjectTree(self.connector)
        tree_dock = QDockWidget("Object Tree", self)
        tree_dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, tree_dock)

        self.props = PropertyEditor(self.connector)
        props_dock = QDockWidget("Properties", self)
        props_dock.setWidget(self.props)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, props_dock)

        self.console = ScriptConsole(self.connector)
        console_dock = QDockWidget("Script Console", self)
        console_dock.setWidget(self.console)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, console_dock)

        # Selection round-trips across all four views: whichever one was
        # clicked drives the rest, using `select_external` everywhere else
        # to avoid feedback loops.
        self.tree.selected.connect(self._on_selected)
        self.canvas.selected.connect(self._on_selected)
        self.preview3d.selected.connect(self._on_selected)
        self.tree.view_results_requested.connect(self._on_view_results)

        self._build_toolbar()
        self._build_menu()
        self.statusBar()

        # Populated after a successful Run; holds the (thread, worker)
        # pair while a run is in flight (must be kept alive — see
        # simulation_runner.run_simulation_async's docstring) and the
        # last SimulationResults once done, so monitor "View Results"
        # has something to plot.
        self._sim_thread = None
        self._sim_worker = None
        self._task_thread = None
        self._task_worker = None
        self._task_kind = None
        self._progress_dialog = None
        self.last_results = None
        self._open_dialogs = []  # keeps non-modal FigureDialogs alive —
        # without a kept reference, Python can garbage-collect a `.show()`n
        # (non-modal) QDialog as soon as the handler function returns,
        # since nothing else holds it, closing the window immediately.

    # ------------------------------------------------------------------ #
    def _on_selected(self, sid) -> None:
        self._current_sid = sid
        self.props.set_selection(sid)
        self.tree.select_external(sid)
        self.canvas.select_external(sid)
        self.preview3d.select_external(sid)

        so = self.connector.get(sid) if sid is not None else None
        self.rename_action.setEnabled(sid is not None)
        self.duplicate_action.setEnabled(sid is not None and so.category != "region")

    def _on_status_message(self, text: str) -> None:
        if text:
            self.statusBar().showMessage(text)
        else:
            self.statusBar().clearMessage()

    # ------------------------------------------------------------------ #
    # Toolbar: Add-Structure / Add-Source / Add-Monitor menus are built
    # DYNAMICALLY from model.STRUCTURE_KINDS / SOURCE_KINDS / MONITOR_KINDS
    # — the same registries `classify_category` uses for script-namespace
    # syncing — rather than one hardcoded action per class. Adding a new
    # class to those tuples in model.py (plus, if it has required
    # constructor fields, a fallback default in defaults.py) is then the
    # ONLY change needed to get a working toolbar entry for it; nothing
    # here needs to know class names ahead of time.
    # ------------------------------------------------------------------ #
    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        # Create Actions and connect to the MainWindow methods
        open_ws_action = QAction("Open Workspace...", self)
        open_ws_action.setShortcut(QKeySequence("Ctrl+O"))
        open_ws_action.triggered.connect(self._open_workspace)
        file_menu.addAction(open_ws_action)

        save_ws_action = QAction("Save Workspace...", self)
        save_ws_action.setShortcut(QKeySequence("Ctrl+S"))
        save_ws_action.triggered.connect(self._save_workspace)
        file_menu.addAction(save_ws_action)

        file_menu.addSeparator()

        import_script_action = QAction("Import Setup from Python...", self)
        import_script_action.triggered.connect(self._import_script)
        file_menu.addAction(import_script_action)

        export_script_action = QAction("Export Setup as Python...", self)
        export_script_action.triggered.connect(self._export_script)
        file_menu.addAction(export_script_action)

        file_menu.addSeparator()

        import_gds_action = QAction("Import GDS Layout...", self)
        import_gds_action.triggered.connect(self._import_gds)
        file_menu.addAction(import_gds_action)

        edit_menu = menu_bar.addMenu("Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)  # Ctrl+Z
        self.undo_action.triggered.connect(self._on_undo)
        self.undo_action.setEnabled(False)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        # StandardKey.Redo already resolves to Ctrl+Y on Windows and
        # Ctrl+Shift+Z elsewhere; adding Ctrl+Y explicitly on top makes it
        # work everywhere, matching the TODO's explicit ask for Ctrl+Y.
        self.redo_action.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        self.redo_action.triggered.connect(self._on_redo)
        self.redo_action.setEnabled(False)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        self.rename_action = QAction("Rename...", self)
        self.rename_action.setShortcut(QKeySequence("F2"))
        self.rename_action.triggered.connect(self._rename_current)
        self.rename_action.setEnabled(False)
        edit_menu.addAction(self.rename_action)

        self.duplicate_action = QAction("Duplicate", self)
        self.duplicate_action.setShortcut(QKeySequence("Ctrl+D"))
        self.duplicate_action.triggered.connect(self._duplicate_current)
        self.duplicate_action.setEnabled(False)
        edit_menu.addAction(self.duplicate_action)

        edit_menu.addSeparator()

        self.copy_action = QAction("Copy", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)  # Ctrl+C
        self.copy_action.triggered.connect(self._on_copy)
        edit_menu.addAction(self.copy_action)

        self.paste_action = QAction("Paste", self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)  # Ctrl+V
        self.paste_action.triggered.connect(self._on_paste)
        edit_menu.addAction(self.paste_action)

    # ------------------------------------------------------------------ #
    # Undo/Redo — whole-scene-snapshot, see undo.py. `_checkpoint()` is
    # for actions guaranteed to succeed (Add/Delete/Duplicate/Rename):
    # call it BEFORE the mutation. `_checkpoint_with_snapshot()` is for
    # actions that might fail validation partway through (Property
    # Editor Apply, script console lines): capture the snapshot before
    # attempting the mutation, but only push it once success is
    # confirmed — see undo.py's checkpoint_with() docstring.
    # ------------------------------------------------------------------ #
    def _checkpoint(self) -> None:
        self.undo_stack.push_snapshot()
        self._refresh_undo_actions()

    def _checkpoint_with_snapshot(self, snapshot) -> None:
        self.undo_stack.checkpoint_with(snapshot)
        self._refresh_undo_actions()

    def _refresh_undo_actions(self) -> None:
        self.undo_action.setEnabled(self.undo_stack.can_undo())
        self.redo_action.setEnabled(self.undo_stack.can_redo())

    def _on_undo(self) -> None:
        if self.undo_stack.undo():
            self._resync_selection_after_restore()
        self._refresh_undo_actions()

    def _on_redo(self) -> None:
        if self.undo_stack.redo():
            self._resync_selection_after_restore()
        self._refresh_undo_actions()

    def _resync_selection_after_restore(self) -> None:
        """After undo/redo, restore()'s diff (model.py) keeps the same
        sid for anything that persisted with the same name and was
        merely updated in place — so, unlike an earlier version where
        EVERY undo/redo regenerated every sid unconditionally, the
        current selection is usually still valid and worth keeping
        (re-selecting it also refreshes the Properties panel to the
        restored values). Only actually clears selection when the
        selected object didn't survive the restore at all (removed, or
        replaced via a rename/class-change edge case — see restore()'s
        docstring).
        """
        if self._current_sid is not None and self.connector.has(self._current_sid):
            self._on_selected(self._current_sid)
        else:
            self._on_selected(None)

    # ------------------------------------------------------------------ #
    # Rename / Duplicate — connector.rename()/duplicate() (model.py) did
    # all the real work already; this is just the UI on top of it. Reused
    # both from the Edit menu (acting on whatever's currently selected)
    # and from the Object Tree's per-item context menu (acting on
    # whichever row was right-clicked, which may not be the current
    # selection).
    # ------------------------------------------------------------------ #
    def _rename_current(self) -> None:
        if self._current_sid is not None:
            self._rename_sid(self._current_sid)

    def _duplicate_current(self) -> None:
        if self._current_sid is not None:
            self._duplicate_sid(self._current_sid)

    def _rename_sid(self, sid: str) -> None:
        so = self.connector.get(sid)
        new_name, ok = QInputDialog.getText(self, "Rename", "New name:", QLineEdit.EchoMode.Normal, so.name)
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == so.name:
            return
        if self.connector.name_exists(new_name):
            # Two SceneObjects sharing one namespace key silently breaks
            # sync_from_script (one name can only ever point to one
            # tracked sid) and workspace save/load — refuse rather than
            # let it happen.
            QMessageBox.warning(self, "Name already in use", f"'{new_name}' is already used by another object.")
            return
        self._checkpoint()
        self.connector.rename(sid, new_name)

    def _duplicate_sid(self, sid: str) -> None:
        so = self.connector.get(sid)
        if so.category == "region":
            self.statusBar().showMessage("The simulation region can't be duplicated.", 5000)
            return
        self._checkpoint()
        new_sid = self.connector.duplicate(sid)
        self._on_selected(new_sid)

    # ------------------------------------------------------------------ #
    # Copy/Paste — via the OS clipboard (see workspace.py's
    # entries_to/from_clipboard_text), which is what makes this work
    # BETWEEN two separate running windows of the app, not just within
    # one. Copy acts on whichever view actually holds a multi-selection
    # (tree or canvas — see _active_selection_sids), falling back to
    # the single "primary" selection for the ordinary single-click case.
    # ------------------------------------------------------------------ #
    def _active_selection_sids(self) -> list[str]:
        tree_sids = self.tree.selected_sids()
        if len(tree_sids) > 1:
            return tree_sids
        canvas_sids = self.canvas.selected_sids()
        if len(canvas_sids) > 1:
            return canvas_sids
        return [self._current_sid] if self._current_sid is not None else []

    def _copy_sids(self, sids: list[str]) -> None:
        if not sids:
            return
        text = entries_to_clipboard_text(self.connector, sids)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Copied {len(sids)} object(s).", 3000)

    def _on_copy(self) -> None:
        self._copy_sids(self._active_selection_sids())

    def _on_paste(self) -> None:
        text = QApplication.clipboard().text()
        try:
            entries = entries_from_clipboard_text(text)
        except (ValueError, TypeError) as exc:  # ValueError: no marker /
            # bad JSON shape; TypeError: e.g. a beamz class that's since
            # dropped/renamed a kwarg the copy was made with — either way
            # this is untrusted clipboard content, not a crash.
            self.statusBar().showMessage(f"Can't paste: {exc}", 5000)
            return
        if not entries:
            return
        pre_snapshot = self.connector.snapshot()
        try:
            new_sids = self.connector.import_entries(entries)
        except Exception as exc:  # noqa: BLE001 — clipboard content could
            # in principle be a copy from an incompatible beamz version;
            # same reasoning as _import_script's catch-all.
            QMessageBox.warning(self, "Can't paste", str(exc))
            return
        self._checkpoint_with_snapshot(pre_snapshot)
        if new_sids:
            self._on_selected(new_sids[-1])

    def _export_script(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export Setup", "setup.py", "Python files (*.py)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(ser.export_script(self.connector))
        except OSError as exc:
            QMessageBox.warning(self, "Can't save script", str(exc))

    def _import_script(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Import Setup", "", "Python files (*.py)")
        if not path:
            return
        try:
            with open(path) as f:
                code = f.read()
        except OSError as exc:
            QMessageBox.warning(self, "Can't read script", str(exc))
            return
        # Pre-snapshot, checkpoint only pushed on success — same reasoning
        # as Property Editor's Apply (see undo.py's checkpoint_with()):
        # an import can fail partway through arbitrary user code, and a
        # failed import shouldn't waste an undo slot on a no-op.
        pre_snapshot = self.connector.snapshot()
        try:
            ser.import_script(self.connector, code)
        except Exception as exc:  # noqa: BLE001 — loaded scripts are
            # arbitrary Python; anything from a syntax error to beamz
            # rejecting a bad value can happen here, and showing the
            # message beats crashing on a bad file.
            QMessageBox.warning(self, "Can't load script", str(exc))
            self.props.set_selection(None)
            return
        self._checkpoint_with_snapshot(pre_snapshot)
        self.props.set_selection(None)

    def _import_gds(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Import GDS Layout", "", "GDS files (*.gds *.gds2 *.gdsii)")
        if not path:
            return
        dialog = GdsImportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_kwargs is None:
            return

        try:
            from beamz.design import import_gds
        except ImportError as exc:
            QMessageBox.warning(self, "GDS import unavailable", str(exc))
            return
        try:
            imported = import_gds(path, **dialog.result_kwargs)
        except ImportError as exc:
            # beamz's own import_gds() raises this INSIDE the call too —
            # beamz.design itself has no import-time dependency on
            # gdsfactory (per gds.py's own module docstring), so the
            # module import above can succeed even without it installed;
            # the real check only happens once a GDS function actually
            # runs.
            QMessageBox.warning(self, "GDS import unavailable", f"{exc}\n\nInstall it with: pip install beamz[gds]")
            return
        except Exception as exc:  # noqa: BLE001 — bad file, requested
            # layer absent from it, unresolvable component, etc.
            QMessageBox.warning(self, "Can't import GDS file", str(exc))
            return

        # One checkpoint for the whole import (potentially many
        # structures) — see undo.py's checkpoint_with() docstring.
        pre_snapshot = self.connector.snapshot()
        new_sids = [self.connector.add_from_object("structure", structure) for structure in imported.design.structures]
        self._checkpoint_with_snapshot(pre_snapshot)

        n_ports = len(imported.ports)
        message = f"Imported {len(new_sids)} structure(s) from GDS."
        if n_ports:
            # Ports (modal port planes at the component's original layout
            # boundary) aren't auto-converted into ModeSource/ModeMonitor
            # placeables yet — flagged rather than silently dropped, so
            # it's clear this import isn't a fully wired-up simulation on
            # its own.
            message += f" ({n_ports} port(s) available but not yet auto-added as sources/monitors.)"
        self.statusBar().showMessage(message, 8000)

        answer = QMessageBox.question(
            self,
            "Match Simulation Domain?",
            "Also resize the Design domain (width/height/depth/background) to match "
            f"the imported layout ({imported.design.width * 1e6:.3g} \u00d7 {imported.design.height * 1e6:.3g} "
            f"\u00d7 {imported.design.depth * 1e6:.3g} \u00b5m)?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.connector.design_width = imported.design.width
            self.connector.design_height = imported.design.height
            self.connector.design_depth = imported.design.depth
            self.connector.background_material = imported.design.background
            self.connector.design_changed.emit()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Add", self)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        tb.setIconSize(QSize(icons.ICON_SIZE, icons.ICON_SIZE))
        self.addToolBar(tb)

        self._add_category_menu(tb, "Structure", "structure", STRUCTURE_KINDS, icons.structure_icon())
        self._add_category_menu(tb, "Source", "source", SOURCE_KINDS, icons.source_icon())
        self._add_category_menu(tb, "Monitor", "monitor", MONITOR_KINDS, icons.monitor_icon())

        region_action = tb.addAction(icons.region_icon(), "Region")
        region_action.triggered.connect(self._add_region)

        tb.addSeparator()
        fit_action = tb.addAction(icons.fit_icon(), "Fit View")
        fit_action.triggered.connect(self._fit_current_view)

        solver_tb = QToolBar("Solver", self)
        solver_tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        solver_tb.setIconSize(QSize(icons.ICON_SIZE, icons.ICON_SIZE))
        self.addToolBar(solver_tb)

        mesh_action = solver_tb.addAction(icons.mesh_icon(), "View Mesh")
        mesh_action.triggered.connect(self._on_view_mesh)
        memory_action = solver_tb.addAction(icons.memory_icon(), "Memory")
        memory_action.triggered.connect(self._on_memory_estimate)

        solver_tb.addSeparator()
        run_action = solver_tb.addAction(icons.run_icon(), "Run Solver")
        run_action.triggered.connect(self._on_run_solver)

    def _add_category_menu(
        self, toolbar: QToolBar, label: str, category: str, kinds: tuple[str, ...], icon
    ) -> None:
        menu = QMenu(f"Add {label}", self)
        for class_name in kinds:
            menu.addAction(class_name, lambda cn=class_name: self._add_dynamic(category, cn))

        action = toolbar.addAction(icon, label)
        action.setMenu(menu)
        # Force the toolbar button to pop the menu on a plain click rather
        # than requiring a separate dropdown-arrow click (Qt's default for
        # an action-with-menu varies by style/platform) — deterministic
        # across platforms, and reads better for an icon+label button than
        # a tiny arrow-only trigger area.
        button = toolbar.widgetForAction(action)
        if isinstance(button, QToolButton):
            button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

    def _add_dynamic(self, category: str, class_name: str) -> None:
        try:
            recipe = defaults.build_default_recipe(class_name)
        except TypeError as exc:
            self.statusBar().showMessage(f"Can't add {class_name}: {exc}", 5000)
            return
        # Checkpoint BEFORE the mutation — see undo.py's checkpoint()
        # docstring for why the ordering matters. Deliberately after the
        # recipe-build try/except above: a TypeError there means nothing
        # is actually about to change, so nothing needs to be undoable.
        self._checkpoint()
        # Extract name to prevent kwarg collision
        obj_name = recipe.pop("name", None)
        if obj_name:
            self.connector.add_object(category, class_name, name=obj_name, **recipe)
        else:
            self.connector.add_object(category, class_name, **recipe)

    def _add_region(self) -> None:
        if self.connector.by_category("region"):
            self.statusBar().showMessage("A Simulation Region already exists — select it to edit.", 5000)
            return
        self._checkpoint()
        self.connector.add_region()

    def _fit_current_view(self) -> None:
        current = self.view_tabs.currentWidget()
        if hasattr(current, "fit_to_content"):
            current.fit_to_content()

    # ------------------------------------------------------------------ #
    # Solver: View Mesh / Memory Estimate / Run, all built on the same
    # `simulation_runner.build_simulation()` — which validates materials
    # are assigned first and raises a clear ValueError naming the actual
    # structures if not (see simulation_runner.py's docstring for why
    # that check exists: beamz's own rasterizer crashes with a completely
    # unhelpful AttributeError otherwise). All three of these show that
    # error in a dialog rather than letting it propagate.
    # ------------------------------------------------------------------ #
    def _build_simulation_or_warn(self):
        try:
            return sr.build_simulation(self.connector)
        except Exception as exc:  # noqa: BLE001 — deliberately broad, see
            # simulation_runner.build_simulation's docstring: beamz can
            # raise several different exception types during compilation
            # depending on what's wrong (missing material -> our own
            # ValueError; invalid source geometry -> beamz's own
            # ValueError; etc.) and the point here is to catch ALL of
            # them and show a message instead of crashing the GUI.
            QMessageBox.warning(self, "Can't build simulation", str(exc))
            return None

    def _apply_coordinate_offset_labels(self, axes) -> None:
        """Relabels a beamz-drawn plot's tick values from beamz's own
        shifted/origin-anchored frame back into the user's world
        coordinates (the ones actually typed into the Properties panel).
        The two frames are legitimately different whenever a
        SimulationRegion isn't centered at the origin — Design has no
        offset field, so build_simulation() re-anchors everything onto
        it (see simulation_runner.py) — which is correct for the actual
        physics, but confusing to look at if the numbers on screen don't
        match what you typed.

        Two DIFFERENT beamz plot conventions have to be handled here,
        confirmed by checking each directly rather than assuming one:
        `Simulation.plot()`'s raw tick values are in METRES with a
        display-only formatter (mislabeled axis "(um)");
        `SimulationResults.plot_field()`'s raw tick values are ALREADY in
        MICRONS under the same-looking "(um)" label. An earlier version
        assumed metres for both, which for plot_field() multiplied an
        already-micron value by 1e6 a SECOND time — labels like "999999"
        instead of order-1 micron values. Detected here via the actual
        tick magnitude rather than the label text, since both cases
        render an identical-looking axis label.
        """
        import matplotlib.ticker as mticker

        offset_m = sr.compute_region_offset(self.connector)
        dim_index = {"x": 0, "y": 1, "z": 2}
        ax_list = axes if hasattr(axes, "__iter__") else [axes]
        for ax in ax_list:
            for axis_obj, label in ((ax.xaxis, ax.get_xlabel()), (ax.yaxis, ax.get_ylabel())):
                dim = label.strip()[0].lower() if label else None
                if dim not in dim_index:
                    continue
                shift_m = offset_m[dim_index[dim]]
                if not shift_m:
                    continue
                ticks = axis_obj.get_majorticklocs()
                max_abs = max((abs(t) for t in ticks), default=0.0)
                # On-chip photonic domains are always << 1 in metres
                # (sub-millimetre); the same domain already-in-microns is
                # typically order 1-1000 — a wide enough gap to reliably
                # tell the two conventions apart from tick magnitude alone.
                if max_abs > 1e-3:  # already microns
                    shift = shift_m * 1e6
                    axis_obj.set_major_formatter(mticker.FuncFormatter(lambda v, _p, s=shift: f"{v + s:g}"))
                else:  # raw metres
                    axis_obj.set_major_formatter(
                        mticker.FuncFormatter(lambda v, _p, s=shift_m: f"{(v + s) * 1e6:g}")
                    )

    def _on_view_mesh(self) -> None:
        sim = self._build_simulation_or_warn()
        if sim is None:
            return
        self._run_task_with_progress(sim.plot, "Rendering mesh view\u2026", "mesh")

    def _on_memory_estimate(self) -> None:
        sim = self._build_simulation_or_warn()
        if sim is None:
            return
        self._run_task_with_progress(
            lambda: sim.memory_estimate(num_steps=sim.num_steps), "Estimating memory\u2026", "memory"
        )

    def _run_task_with_progress(self, fn, message: str, kind: str) -> None:
        """Shared plumbing for View Mesh / Memory Estimate — both used to
        call straight into beamz on the GUI thread, which could freeze
        the whole app for a large grid (or worse: OOM-crash it outright,
        reproduced directly against a 25M-cell grid) with zero feedback.
        Same fix as Run Solver: off the GUI thread, via a bound-method
        pair (`_on_task_finished`/`_on_task_failed`) rather than local
        closures — cross-thread Qt signal delivery to a QObject's bound
        method is safely queued onto the receiver's own thread
        automatically; a plain closure has no thread affinity for Qt to
        use for that, and constructing GUI objects from the wrong thread
        (which the result handlers do — QDialog/QMessageBox) is exactly
        the kind of thing that crashes intermittently rather than
        reliably, matching "sometimes crashes entire UI".
        """
        self._task_kind = kind
        self._progress_dialog = QProgressDialog(message, None, 0, 0, self)
        self._progress_dialog.setWindowTitle("BEAMZ Studio")
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setRange(0, 0)
        self._progress_dialog.show()
        self._task_thread, self._task_worker = sr.run_off_thread(fn, self._on_task_finished, self._on_task_failed)

    def _on_task_finished(self, result: Any) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        if self._task_kind == "mesh":
            fig, axes = result
            self._apply_coordinate_offset_labels(axes)
            dialog = FigureDialog(fig, title="Simulation Layout / Mesh", parent=self)
            dialog.show()
            self._open_dialogs.append(dialog)
        elif self._task_kind == "memory":
            QMessageBox.information(self, "Memory Estimate", sr.format_memory_report(result))

    def _on_task_failed(self, message: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        title = "Can't render mesh view" if self._task_kind == "mesh" else "Can't estimate memory"
        QMessageBox.warning(self, title, message)

    def _on_run_solver(self) -> None:
        sim = self._build_simulation_or_warn()
        if sim is None:
            return

        self._progress_dialog = QProgressDialog("Running simulation\u2026", None, 0, 0, self)
        self._progress_dialog.setWindowTitle("BEAMZ Studio")
        self._progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        # Indeterminate (busy) bar, not a percentage: Simulation.run() is
        # a single blocking call with no incremental progress callback —
        # confirmed by reading its docstring — so a real progress
        # percentage isn't something this can honestly show.
        self._progress_dialog.setRange(0, 0)
        self._progress_dialog.show()

        self._sim_thread, self._sim_worker = sr.run_simulation_async(
            sim, on_finished=self._on_run_finished, on_failed=self._on_run_failed
        )

    def _on_run_finished(self, results) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        self.last_results = results
        self.current_results = sr.build_results_datasets(results, self.connector)
        self.statusBar().showMessage("Simulation finished. Right-click a monitor to view results.", 8000)

    def _on_run_failed(self, message: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        QMessageBox.warning(self, "Simulation failed", message)

    def _on_view_results(self, sid: str) -> None:
        if self.last_results is None:
            QMessageBox.information(
                self, "No results yet", "Run the simulation first (Solver \u2192 Run Solver)."
            )
            return
        so = self.connector.get(sid)
        # FluxMonitor/ModeMonitor have no spatial field frame at all
        # (plot_field() is a FieldMonitor-only concept — a flux/mode
        # monitor only has scalar-per-frequency data), so each gets its
        # own spectrum plot instead. Only the FieldMonitor path draws
        # actual space-domain axes, so only it gets the coordinate-offset
        # relabeling below.
        try:
            if so.class_name == "FluxMonitor":
                fig, axes = sr.plot_flux_spectrum(self.last_results, so)
            elif so.class_name == "ModeMonitor":
                fig, axes = sr.plot_mode_spectrum(self.last_results, so)
            else:
                fig, axes = self.last_results.plot_field(monitor_name=so.name, show=False)
        except Exception as exc:  # noqa: BLE001 — e.g. the monitor wasn't
            # part of the simulation that produced `last_results` (deleted
            # and re-added since, or the run predates it existing), or
            # (Flux/Mode) simply has no recorded frequency-domain data yet.
            QMessageBox.warning(self, "Can't plot monitor results", str(exc))
            return
        if so.class_name not in ("FluxMonitor", "ModeMonitor"):
            self._apply_coordinate_offset_labels(axes)
        dialog = FigureDialog(fig, title=f"Results \u2014 {so.name}", parent=self)
        dialog.show()
        self._open_dialogs.append(dialog)

    # --- Slot Implementations ---
    def _save_workspace(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Workspace", "", "BEAMZ Workspace (*.h5)")
        if path:
            try:
                self.workspace.save_workspace(path, self.current_results)
                self.statusBar().showMessage(f"Saved workspace to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _open_workspace(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Workspace", "", "BEAMZ Workspace (*.h5)")
        if path:
            try:
                self.current_results = self.workspace.load_workspace(path)
                self.statusBar().showMessage(f"Loaded workspace from {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))