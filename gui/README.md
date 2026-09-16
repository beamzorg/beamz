# BEAMZ Studio — v1 skeleton

An open-source, GUI for the [beamz](https://github.com/beamzorg/beamz)
FDTD engine: define structures by editing parametric primitives, with a
script console that operates on the exact same live objects the GUI edits.

This is the **v1 minimal test bed** — connector + object tree + 2D canvas +
property editor + script console, wired end-to-end and covered by an
executable smoke test. Materials database, GDS import, sources/monitors/
boundaries panels, 3D preview, and simulation execution are not built yet.

## Run the app

```bash
python3 main.py
```

Use the toolbar (**+ Rectangle / + Circle / + Ring**) to place structures,
click a structure in the Object Tree or on the canvas to select it, and
edit its parameters in the Properties panel on the right. The Script
Console at the bottom runs Python against the same live objects — try:

```python
wg = bz.Rectangle(position=(0,0,0), width=2e-6, height=0.5e-6, depth=0.22e-6)
wg = wg.updated_copy(width=3.5e-6)   # structures are frozen; edit = rebind
del wg
```

## File hierarchy

```
beamz-gui/
├── main.py                        entry point: creates QApplication + MainWindow
├── requirements.txt                beamz, PySide6
├── smoke_test.py                   headless automated test suite (run this first)
└── beamz_gui/
    ├── __init__.py
    ├── model.py                    BeamzConnector — the single source of truth.
    │                                 Wraps beamz's frozen dataclass structures in
    │                                 stable-id SceneObjects; add_structure /
    │                                 update_param / remove_structure /
    │                                 sync_from_script(); build_design() assembles
    │                                 a live beamz.Design from current state.
    ├── introspection.py            Reads a beamz dataclass's fields and classifies
    │                                 each as float/int/bool/str/tuple2/tuple3/
    │                                 readonly, so the property editor can build
    │                                 forms for new beamz primitives automatically.
    ├── scripting.py                run_snippet(code, namespace): REPL-style
    │                                 execution (echoes bare expressions), used by
    │                                 the console. No Qt dependency — unit-testable
    │                                 on its own.
    └── widgets/
        ├── __init__.py
        ├── object_tree.py          Left dock: QTreeWidget listing every structure.
        ├── property_editor.py      Right dock: auto-generated QFormLayout from
        │                             introspection.py; edits round-trip through
        │                             connector.update_param().
        ├── canvas_2d.py            Center: top-down QGraphicsView. Renders
        │                             Rectangle/Circle/Ring as proxy shapes (Ring
        │                             as a true annulus via QPainterPath). Other
        │                             primitives (Taper, Polygon, Box, Sphere, ...)
        │                             exist fully in the model/tree/properties but
        │                             have no 2D glyph yet.
        └── console.py               Bottom dock: the script console widget —
                                      input line + scrollback + history, executes
                                      against connector.namespace, then calls
                                      connector.sync_from_script() to reconcile.
```

## Design notes worth knowing before extending this

- **beamz structures are frozen dataclasses.** There is no in-place
  mutation anywhere in this app — every edit (GUI or script) produces a
  new instance via `obj.updated_copy(**changes)`. The connector is the
  only place object identity is allowed to change; everything else
  addresses structures by a GUI-owned stable `sid` string.
- **GUI and console share one namespace.** `connector.namespace` is the
  literal dict the console executes against. A GUI edit calls
  `update_param()` directly; a console edit runs arbitrary code and then
  calls `sync_from_script()`, which diffs the namespace against tracked
  structures (new structure-typed variable → added, tracked variable now
  points to a different object → edited, tracked variable gone → removed).
- **Everything is signal-driven.** `BeamzConnector` emits Qt signals
  (`structure_added`, `structure_changed`, `structure_removed`,
  `design_changed`); every view subscribes independently and has zero
  knowledge of the others. Adding a new view (e.g. a future 3D preview)
  means subscribing to these same four signals — no other wiring needed.
