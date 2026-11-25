import json
import os

from eagmodel.cli import main


def test_simulate_then_check(tmp_path):
    d = tmp_path / "data"
    assert main(["simulate", "--out", str(d), "--n", "120", "--rho", "-2", "--seed", "3"]) == 0
    out = tmp_path / "report"
    assert main(["check", "--analytic", str(d / "analytic.csv"), "--holistic", str(d / "holistic.csv"),
                 "--students", str(d / "students.csv"), "--out", str(out), "--threshold", "0.6"]) == 0
    for name in ("report.html", "report.md", "summary.json", "students.csv"):
        assert os.path.getsize(out / name) > 0
    summary = json.loads((out / "summary.json").read_text())
    assert 0 < summary["kappa"] < 1
