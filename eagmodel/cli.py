"""Command line interface:  eag simulate | check | power"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="eag", description="Evidence-Aggregation-Growth model of competence")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a synthetic data set in the input format")
    s.add_argument("--out", required=True, help="folder for analytic.csv, holistic.csv, students.csv")
    s.add_argument("--n", type=int, default=200, help="number of students")
    s.add_argument("--rho", type=float, default=-2.0, help="substitution parameter (1 = sum score)")
    s.add_argument("--seed", type=int, default=1)

    c = sub.add_parser("check", help="fit the model to your data and test whether the data support it")
    c.add_argument("--analytic", required=True, help="CSV/XLSX: student, occasion, component, indicator, score")
    c.add_argument("--holistic", required=True, help="CSV/XLSX: student, occasion, rater, score")
    c.add_argument("--students", help="optional CSV/XLSX: student, group")
    c.add_argument("--out", required=True, help="folder for the report")
    c.add_argument("--threshold", type=float, help="readiness threshold on the competence scale (0 to 1)")
    c.add_argument("--max-score", type=int, help="highest rubric level if not all levels occur in the data")
    c.add_argument("--seed", type=int, default=0)

    w = sub.add_parser("power", help="plan the sample size by simulation")
    w.add_argument("--n", type=int, nargs="+", default=[100, 200, 400])
    w.add_argument("--rho", type=float, default=-2.0)
    w.add_argument("--reps", type=int, default=50)
    w.add_argument("--seed", type=int, default=1)

    a = p.parse_args(argv)
    if a.cmd == "simulate":
        from .simulate import SimConfig, simulate

        data, truth = simulate(SimConfig(n_students=a.n, rho=a.rho), seed=a.seed)
        data.to_files(a.out)
        with open(f"{a.out}/truth.json", "w") as fh:
            json.dump(truth["config"], fh, indent=2)
        print(f"Wrote analytic.csv, holistic.csv, students.csv and truth.json to {a.out}")
    elif a.cmd == "check":
        from .data import EAGData
        from .estimate import fit
        from .report import write_report
        from .validate import validate

        data = EAGData.from_files(a.analytic, a.holistic, a.students, a.max_score)
        print(f"Read {data.n_students} students, {data.n_components} components, {data.n_occasions} occasion(s), "
              f"{data.n_raters} rater(s). Fitting ...")
        f = fit(data, seed=a.seed, threshold=a.threshold, extra_models=True, verbose=True)
        rep = validate(f, threshold=a.threshold, seed=a.seed)
        path = write_report(f, rep, a.out)
        print("\n" + rep.headline)
        print(rep.checks_frame()[["check", "verdict"]].to_string(index=False))
        print(f"\nFull report: {path}")
    elif a.cmd == "power":
        from .power import power_analysis
        from .simulate import SimConfig

        df = power_analysis(SimConfig(rho=a.rho), a.n, a.reps, seed=a.seed,
                            progress=lambda f: print(f"\r{100 * f:5.1f}%", end="", file=sys.stderr))
        print("\n" + df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
