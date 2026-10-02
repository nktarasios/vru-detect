import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_error_report.py"
    spec = importlib.util.spec_from_file_location("build_error_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_error_report_builds(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    out = tmp_path / "ERROR_REPORT.md"
    monkeypatch.setattr(mod, "OUT", out)
    mod.main()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Error / residual-risk report")
    assert "night pedestrians" in text.lower() or "Night pedestrians" in text
    assert "small" in text.lower()
