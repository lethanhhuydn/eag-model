"""Data container and loaders for the EAG model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _component_codes(components):
    """Short codes for default indicator labels: initials of the component name ("Problem framing" gives PF),
    or its first three letters when the name is one word or the initials are not unique."""
    codes = []
    for c in components:
        words = [w for w in str(c).replace("-", " ").replace("_", " ").split() if w]
        codes.append("".join(w[0].upper() for w in words) if len(words) > 1 else str(c)[:3])
    if len(set(codes)) < len(codes):
        codes = [f"C{k + 1}" for k in range(len(components))]
    return [c + "_" if c[-1:].isdigit() else c for c in codes]


@dataclass
class EAGData:
    """Scores for N students, T occasions, K components, J indicators and R raters.

    X      analytic rubric scores, int array (N, T, K, J), -1 where missing
    Y      holistic ratings, float array (N, T, R), NaN where missing
    times  numeric occasion codes (design cycles), length T
    group  optional 0/1 indicator per student (for example comparison/intervention)
    """

    X: np.ndarray
    Y: np.ndarray
    times: np.ndarray
    max_score: int
    group: Optional[np.ndarray] = None
    students: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    indicators: List[List[str]] = field(default_factory=list)
    raters: List[str] = field(default_factory=list)
    occasions: List[str] = field(default_factory=list)
    group_labels: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=int)
        self.Y = np.asarray(self.Y, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        n, t, k, j = self.X.shape
        if self.Y.shape[:2] != (n, t):
            raise ValueError("X and Y must agree on students and occasions")
        if not self.students:
            self.students = [f"S{i + 1:03d}" for i in range(n)]
        if not self.components:
            self.components = [f"Component {i + 1}" for i in range(k)]
        if not self.indicators:
            codes = _component_codes(self.components)
            self.indicators = [[f"{code}{i + 1}" for i in range(j)] for code in codes]
        if not self.raters:
            self.raters = [f"Rater {i + 1}" for i in range(self.Y.shape[2])]
        if not self.occasions:
            self.occasions = [f"{v:g}" for v in self.times]
        if self.group is not None:
            self.group = np.asarray(self.group, dtype=int)
            if not self.group_labels:
                self.group_labels = ["0", "1"]

    @property
    def n_students(self):
        return self.X.shape[0]

    @property
    def n_occasions(self):
        return self.X.shape[1]

    @property
    def n_components(self):
        return self.X.shape[2]

    @property
    def n_indicators(self):
        return self.X.shape[3]

    @property
    def n_raters(self):
        return self.Y.shape[2]

    def subset(self, idx):
        """Return the data for a subset of students."""
        idx = np.asarray(idx)
        return EAGData(
            X=self.X[idx], Y=self.Y[idx], times=self.times, max_score=self.max_score,
            group=None if self.group is None else self.group[idx],
            students=[self.students[i] for i in idx], components=self.components,
            indicators=self.indicators, raters=self.raters, occasions=self.occasions,
            group_labels=self.group_labels,
        )

    # ------------------------------------------------------------------ export
    def to_frames(self):
        """Return (analytic, holistic, students) long-format pandas data frames."""
        import pandas as pd

        n, t, k, j = self.X.shape
        times = np.asarray(self.times, float)
        whole = bool(np.all(times == np.round(times)))            # write design cycles as 0, 1, 2 when they are whole
        occ = [int(v) if whole else float(v) for v in times]
        rows = []
        for i in range(n):
            for o in range(t):
                for c in range(k):
                    for q in range(j):
                        v = self.X[i, o, c, q]
                        if v >= 0:
                            rows.append((self.students[i], occ[o], self.components[c],
                                         self.indicators[c][q], int(v)))
        analytic = pd.DataFrame(rows, columns=["student", "occasion", "component", "indicator", "score"])
        rows = []
        for i in range(n):
            for o in range(t):
                for r in range(self.Y.shape[2]):
                    v = self.Y[i, o, r]
                    if np.isfinite(v):
                        rows.append((self.students[i], occ[o], self.raters[r], float(v)))
        holistic = pd.DataFrame(rows, columns=["student", "occasion", "rater", "score"])
        students = pd.DataFrame({"student": self.students})
        if self.group is not None:
            students["group"] = [self.group_labels[g] for g in self.group]
        return analytic, holistic, students

    # ------------------------------------------------------------------ import
    @classmethod
    def from_frames(cls, analytic, holistic, students=None, max_score=None):
        """Build the container from long-format data frames.

        analytic  columns student, occasion, component, indicator, score (integers from 0)
        holistic  columns student, occasion, rater, score (any numeric scale)
        students  optional, columns student, group (two levels)
        """
        import pandas as pd

        a = analytic.rename(columns=str.lower).copy()
        h = holistic.rename(columns=str.lower).copy()
        for name, frame, cols in (("analytic", a, ["student", "occasion", "component", "indicator", "score"]),
                                  ("holistic", h, ["student", "occasion", "rater", "score"])):
            missing = [c for c in cols if c not in frame.columns]
            if missing:
                raise ValueError(f"The {name} table lacks the column(s): {', '.join(missing)}")
        a = a.dropna(subset=["score"])
        h = h.dropna(subset=["score"])
        a["student"] = a["student"].astype(str)
        h["student"] = h["student"].astype(str)
        a["occasion"] = pd.to_numeric(a["occasion"], errors="raise")
        h["occasion"] = pd.to_numeric(h["occasion"], errors="raise")
        if not np.allclose(a["score"], np.round(a["score"])):
            raise ValueError("Analytic rubric scores must be whole numbers starting at 0")
        if a["score"].min() < 0:
            raise ValueError("Analytic rubric scores must not be negative")
        studs = sorted(a["student"].unique())
        extra = sorted(set(h["student"]) - set(studs))
        if extra:
            raise ValueError(f"{len(extra)} student(s) have holistic ratings but no rubric scores, e.g. {extra[:3]}")
        times = np.array(sorted(set(a["occasion"]).union(h["occasion"])), dtype=float)
        comps = list(pd.unique(a["component"]))
        inds = [list(pd.unique(a.loc[a["component"] == c, "indicator"])) for c in comps]
        jmax = max(len(v) for v in inds)
        raters = sorted(h["rater"].astype(str).unique())
        s_ix = {s: i for i, s in enumerate(studs)}
        t_ix = {v: i for i, v in enumerate(times)}
        c_ix = {c: i for i, c in enumerate(comps)}
        r_ix = {r: i for i, r in enumerate(raters)}
        X = -np.ones((len(studs), len(times), len(comps), jmax), dtype=int)
        for c, items in zip(comps, inds):
            sub = a[a["component"] == c]
            q_ix = {q: i for i, q in enumerate(items)}
            X[sub["student"].map(s_ix).to_numpy(), sub["occasion"].map(t_ix).to_numpy(), c_ix[c],
              sub["indicator"].map(q_ix).to_numpy()] = sub["score"].astype(int).to_numpy()
        Y = np.full((len(studs), len(times), len(raters)), np.nan)
        Y[h["student"].map(s_ix).to_numpy(), h["occasion"].map(t_ix).to_numpy(),
          h["rater"].astype(str).map(r_ix).to_numpy()] = h["score"].astype(float).to_numpy()
        group, labels = None, []
        if students is not None:
            s = students.rename(columns=str.lower).copy()
            if "group" in s.columns:
                s["student"] = s["student"].astype(str)
                s = s.drop_duplicates("student").set_index("student")
                g = s.reindex(studs)["group"]
                if g.isna().any():
                    raise ValueError("Every student needs a group label when a group column is supplied")
                labels = sorted(g.astype(str).unique())
                if len(labels) == 2:
                    group = (g.astype(str) == labels[1]).astype(int).to_numpy()
                elif len(labels) > 2:
                    raise ValueError("The group column must have exactly two levels")
                else:
                    labels = []
        m = int(a["score"].max()) if max_score is None else int(max_score)
        return cls(X=X, Y=Y, times=times, max_score=m, group=group, students=studs, components=comps,
                   indicators=[v + [f"(unused {i})" for i in range(jmax - len(v))] for v in inds],
                   raters=raters, occasions=[f"{v:g}" for v in times], group_labels=labels)

    @classmethod
    def from_files(cls, analytic_path, holistic_path, students_path=None, max_score=None):
        """Read CSV or Excel files in the long format described in :meth:`from_frames`."""
        import pandas as pd

        def read(p):
            p = str(p)
            return pd.read_excel(p) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p)

        return cls.from_frames(read(analytic_path), read(holistic_path),
                               None if students_path is None else read(students_path), max_score)

    def to_files(self, folder):
        """Write analytic.csv, holistic.csv and students.csv into a folder."""
        import os

        os.makedirs(folder, exist_ok=True)
        a, h, s = self.to_frames()
        a.to_csv(os.path.join(folder, "analytic.csv"), index=False)
        h.to_csv(os.path.join(folder, "holistic.csv"), index=False)
        s.to_csv(os.path.join(folder, "students.csv"), index=False)
