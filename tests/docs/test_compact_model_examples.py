from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_tiny_crossing_example_uses_current_simulation_modules():
    path = ROOT / "examples" / "compact_models" / "tiny_beamz_crossing.py"
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert "beamz.simulation.core" not in source
    assert "use_fixed_micromode_y_projection_convention" not in source
