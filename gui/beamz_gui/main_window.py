from __future__ import annotations
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QTabWidget,
    QToolBar,
    QToolButton,
)

from . import defaults
from . import icons
from . import serialization as ser
from . import simulation_runner as sr
from .model import BeamzConnector, MONITOR_KINDS, SOURCE_KINDS, STRUCTURE_KINDS
from .widgets.canvas_2d import Canvas2D
from .widgets.console import ScriptConsole
from .widgets.figure_dialog import FigureDialog
from .widgets.object_tree import ObjectTree
from .widgets.preview_3d import Preview3D
from .widgets.property_editor import PropertyEditor


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
        self.props.set_selection(sid)
        self.tree.select_external(sid)
        self.canvas.select_external(sid)
        self.preview3d.select_external(sid)

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
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Save Script...", self._on_save_script)
        file_menu.addAction("Load Script...", self._on_load_script)

    def _on_save_script(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Save Script", "scene.py", "Python files (*.py)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(ser.export_script(self.connector))
        except OSError as exc:
            QMessageBox.warning(self, "Can't save script", str(exc))

    def _on_load_script(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Load Script", "", "Python files (*.py)")
        if not path:
            return
        try:
            with open(path) as f:
                code = f.read()
        except OSError as exc:
            QMessageBox.warning(self, "Can't read script", str(exc))
            return
        try:
            ser.import_script(self.connector, code)
        except Exception as exc:  # noqa: BLE001 — loaded scripts are
            # arbitrary Python; anything from a syntax error to beamz
            # rejecting a bad value can happen here, and showing the
            # message beats crashing on a bad file.
            QMessageBox.warning(self, "Can't load script", str(exc))
        self.props.set_selection(None)

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
        self.connector.add_object(category, class_name, **recipe)

    def _add_region(self) -> None:
        if self.connector.by_category("region"):
            self.statusBar().showMessage("A Simulation Region already exists — select it to edit.", 5000)
            return
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
        try:
            fig, axes = self.last_results.plot_field(monitor_name=so.name, show=False)
        except Exception as exc:  # noqa: BLE001 — e.g. the monitor wasn't
            # part of the simulation that produced `last_results` (deleted
            # and re-added since, or the run predates it existing).
            QMessageBox.warning(self, "Can't plot monitor results", str(exc))
            return
        self._apply_coordinate_offset_labels(axes)
        dialog = FigureDialog(fig, title=f"Results \u2014 {so.name}", parent=self)
        dialog.show()
        self._open_dialogs.append(dialog)