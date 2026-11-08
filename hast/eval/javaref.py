"""Instance-by-instance comparison against the Java reference (java_ref/RefRunner)."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_REF = os.path.join(ROOT, "java_ref", "run_ref.sh")
REF_DIR = os.path.join(ROOT, "java_ref", "out")


@dataclass
class RefOutput:
    preds: np.ndarray
    votes: list
    measurements: dict


def run_java_reference(learner_cli: str, arff: str, n: int, tag: str, force: bool = False) -> RefOutput:
    os.makedirs(REF_DIR, exist_ok=True)
    out = os.path.join(REF_DIR, f"{tag}.txt")
    if force or not os.path.exists(out):
        subprocess.run([RUN_REF, learner_cli, arff, str(n), out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return read_reference(out)


def read_reference(path: str) -> RefOutput:
    preds, votes, meas = [], [], {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                k, v = line[1:].rsplit("=", 1)
                meas[k] = float(v)
                continue
            p, v = line.split(";")
            preds.append(int(p))
            votes.append(np.array([float(t) for t in v.split(",")]) if v else np.zeros(0))
    return RefOutput(np.array(preds), votes, meas)


@dataclass
class Comparison:
    n: int
    pred_mismatches: int
    first_pred_mismatch: int
    vote_mismatches: int
    first_vote_mismatch: int
    max_rel_err: float

    def ok(self, exact: bool = True) -> bool:
        return self.pred_mismatches == 0 and (not exact or self.vote_mismatches == 0)

    def __str__(self) -> str:
        return (f"n={self.n} pred_mismatch={self.pred_mismatches} (first {self.first_pred_mismatch}) "
                f"vote_mismatch={self.vote_mismatches} (first {self.first_vote_mismatch}) "
                f"max_rel_err={self.max_rel_err:.3g}")


def compare_votes(ref: RefOutput, py_votes: list) -> Comparison:
    n = min(len(ref.votes), len(py_votes))
    pm = vm = 0
    fp = fv = -1
    max_rel = 0.0
    from hast.trees.kernels import utils_max_index
    for i in range(n):
        a, b = ref.votes[i], np.asarray(py_votes[i], dtype=np.float64)
        pa = ref.preds[i]
        pb = int(utils_max_index(b, b.shape[0])) if b.shape[0] else 0
        if pa != pb:
            pm += 1
            if fp < 0:
                fp = i
        la, lb = a.shape[0], b.shape[0]
        m = max(la, lb)
        aa = np.zeros(m); aa[:la] = a
        bb = np.zeros(m); bb[:lb] = b
        same = np.array_equal(aa, bb, equal_nan=True)
        if not same:
            vm += 1
            if fv < 0:
                fv = i
            with np.errstate(invalid="ignore", divide="ignore"):
                den = np.maximum(np.abs(aa), np.abs(bb))
                rel = np.where(den > 0, np.abs(aa - bb) / den, 0.0)
            rel = rel[np.isfinite(rel)]
            if rel.size:
                max_rel = max(max_rel, float(rel.max()))
    return Comparison(n, pm, fp, vm, fv, max_rel)
