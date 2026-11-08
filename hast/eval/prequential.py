"""Prequential (test-then-train) evaluation replicating the paper's protocol.

Protocol (validated against the paper's ARF+HT/Outdoor numbers, see docs):
MOA ``EvaluatePrequential`` with the cumulative
``BasicClassificationPerformanceEvaluator``; metrics sampled at 20 evenly
spaced points; a NaN metric (e.g. F1 while some class was never predicted)
counts as 0; the reported value is mean ± (population) std over the points.

The cumulative confusion matrix is stored at every sample point, so further
metrics (G-mean, kappa, per-class recall) can be derived afterwards.
"""
from __future__ import annotations

import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from hast.ensembles.base import predicted_class


# --------------------------------------------------------------------------- metrics (MOA semantics)
def moa_metrics(cm: np.ndarray) -> dict:
    """Metrics of ``BasicClassificationPerformanceEvaluator`` from a confusion
    matrix ``cm[true, pred]`` (unit weights)."""
    n = cm.sum()
    K = cm.shape[0]
    correct = np.trace(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = correct / n
        row = cm.sum(axis=0) / n  # predicted fraction
        col = cm.sum(axis=1) / n  # true fraction
        pc = 0.0
        for i in range(K):
            pc += row[i] * col[i]
        kappa = (acc - pc) / (1.0 - pc)
        diag = np.diag(cm).astype(float)
        prec_i = diag / cm.sum(axis=0)  # NaN if class never predicted
        rec_i = diag / cm.sum(axis=1)  # NaN if class never seen
        P = 0.0
        R = 0.0
        for i in range(K):
            P += prec_i[i]
            R += rec_i[i]
        P /= K
        R /= K
        f1 = 2 * ((P * R) / (P + R))
        # extra metrics for the imbalanced study (not MOA)
        seen = cm.sum(axis=1) > 0
        rec_seen = rec_i[seen]
        gmean = float(np.prod(rec_seen) ** (1.0 / rec_seen.size)) if rec_seen.size else math.nan
    return {
        "accuracy": 100.0 * acc,
        "kappa": 100.0 * kappa,
        "f1": 100.0 * f1,
        "precision": 100.0 * P,
        "recall": 100.0 * R,
        "gmean": 100.0 * gmean,
        "recall_per_class": (100.0 * rec_i).tolist(),
    }


def summarize(values) -> tuple[float, float]:
    """Mean and population std over sample points, NaN counted as 0 (paper protocol)."""
    v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    return float(v.mean()), float(v.std())


# --------------------------------------------------------------------------- runner
@dataclass
class RunResult:
    dataset: str
    ensemble: str
    tree: str
    n_instances: int
    sample_points: list
    metrics: dict  # metric -> list over sample points
    tree_size_mean: list  # mean #nodes of the ensemble trees at each sample point
    tree_size_std: list
    splits: dict
    cpu_seconds: float
    wall_seconds: float
    peak_rss_mb: float
    confusion: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        out = {"dataset": self.dataset, "ensemble": self.ensemble, "tree": self.tree,
               "n": self.n_instances, "cpu_s": self.cpu_seconds, "peak_rss_mb": self.peak_rss_mb}
        for k in ("accuracy", "f1", "kappa", "gmean"):
            m, s = summarize(self.metrics[k])
            out[k] = m
            out[k + "_std"] = s
        out["tree_size"] = float(np.mean(self.tree_size_mean)) if self.tree_size_mean else math.nan
        out.update(self.splits)
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh)


def sample_points(n: int, n_points: int = 20, moa_final_point: bool = False) -> list[int]:
    """Indices (1-based instance counts) where metrics are recorded.

    MOA records every ``f = n // n_points`` instances and, additionally, at the
    end of the stream when ``n`` is not a multiple of ``f``. With
    ``moa_final_point=False`` exactly ``n_points`` points are kept.
    """
    f = max(1, n // n_points)
    pts = list(range(f, n + 1, f))[:n_points] if not moa_final_point else list(range(f, n + 1, f))
    if moa_final_point and pts[-1] != n:
        pts.append(n)
    return pts


def run_prequential(model, stream, n_max: int | None = None, n_points: int = 20,
                    moa_final_point: bool = False, ensemble_name: str = "", tree_name: str = "",
                    progress: bool = False) -> RunResult:
    X, y = stream.X, stream.y
    n = len(y) if n_max is None else min(n_max, len(y))
    K = stream.header.n_classes
    pts = set(sample_points(n, n_points, moa_final_point))
    cm = np.zeros((K, K), dtype=np.int64)
    metrics: dict[str, list] = {k: [] for k in ("accuracy", "kappa", "f1", "precision", "recall", "gmean")}
    recall_pc, confusions, ts_mean, ts_std, recorded = [], [], [], [], []
    has_members = hasattr(model, "members")
    t_cpu, t_wall = time.process_time(), time.perf_counter()
    for i in range(n):
        x = X[i]
        yi = int(y[i])
        votes = model.predict_votes(x)
        cm[yi, predicted_class(votes)] += 1
        if has_members:
            model.train(x, yi)
        else:
            model.train(x, yi, 1.0)
        if (i + 1) in pts:
            m = moa_metrics(cm)
            for k in metrics:
                metrics[k].append(m[k])
            recall_pc.append(m["recall_per_class"])
            confusions.append(cm.tolist())
            sizes = model.tree_sizes() if has_members else np.array([model.n_nodes], float)
            ts_mean.append(float(sizes.mean()))
            ts_std.append(float(sizes.std()))
            recorded.append(i + 1)
            if progress:
                el = time.perf_counter() - t_wall
                print(f"  [{i + 1}/{n}] acc={m['accuracy']:.2f} f1={m['f1']:.2f} "
                      f"size={sizes.mean():.1f} {el:.0f}s", flush=True)
    cpu = time.process_time() - t_cpu
    wall = time.perf_counter() - t_wall
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    splits = model.split_counts() if has_members else {
        "splits_periodic": model.n_splits_periodic, "splits_adaptive": model.n_splits_adaptive}
    res = RunResult(stream.name, ensemble_name, tree_name, n, recorded, metrics, ts_mean, ts_std, splits,
                    cpu, wall, peak, confusions)
    res.extra["recall_per_class"] = recall_pc
    return res
