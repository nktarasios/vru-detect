import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_portfolio_figures.py"
    spec = importlib.util.spec_from_file_location("build_portfolio_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portfolio_figure_builders(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    outs = [
        mod.plot_baseline_vs_finetuned(),
        mod.plot_failure_modes(),
        mod.plot_threshold_tradeoff(),
        mod.write_findings_markdown(),
    ]
    for path in outs:
        assert path.exists()
        assert path.stat().st_size > 0
    assert (tmp_path / "FINDINGS.md").read_text(encoding="utf-8").startswith("# VRU-Detect Findings")
