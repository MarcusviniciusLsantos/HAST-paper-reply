"""Shared pieces of the ensemble ports (vote arithmetic identical to MOA's DoubleVector)."""
from __future__ import annotations

import math

import numpy as np

from hast.core.fastrandom import rng_new
from hast.detectors.fast import adwin_cd_input, adwin_estimation, adwin_new
from hast.trees.kernels import utils_max_index


class ADWINCD:
    """``ADWINChangeDetector`` backed by the numba ADWIN (bit-identical to MOA)."""

    __slots__ = ("delta", "ai", "af")

    def __init__(self, delta: float):
        self.delta = delta
        self.ai, self.af = adwin_new(delta)

    def input(self, value: float) -> bool:
        """Feeds the detector; returns ``getChange()``."""
        return adwin_cd_input(self.ai, self.af, value)

    def estimation(self) -> float:
        return adwin_estimation(self.ai, self.af)

    def copy(self) -> "ADWINCD":
        return ADWINCD(self.delta)


def java_random(seed: int):
    """State array of a ``java.util.Random`` (see :mod:`hast.core.fastrandom`)."""
    return rng_new(seed)


def java_round(x: float) -> int:
    """``Math.round(double)``: floor(x + 0.5)."""
    return int(math.floor(x + 0.5))


def normalize_vote(v: np.ndarray) -> np.ndarray:
    """``DoubleVector.normalize``: scale by ``1.0 / sumOfAbsoluteValues``."""
    s = 0.0
    for e in v:
        s += e if e > 0.0 else -e
    return v * (1.0 / s)


def vote_sum(v: np.ndarray) -> float:
    s = 0.0
    for e in v:
        s += e
    return s


class VoteAccumulator:
    """``DoubleVector`` used as ``combinedVote`` (auto-expanding ``addValues``)."""

    __slots__ = ("arr",)

    def __init__(self):
        self.arr = np.zeros(0)

    def add(self, v: np.ndarray) -> None:
        if v.shape[0] > self.arr.shape[0]:
            a = np.zeros(v.shape[0])
            a[: self.arr.shape[0]] = self.arr
            self.arr = a
        self.arr[: v.shape[0]] += v


def predicted_class(votes: np.ndarray) -> int:
    return int(utils_max_index(votes, votes.shape[0]))


def correctly_classifies(tree, x, y) -> bool:
    v = tree.predict_votes(x)
    return predicted_class(v) == y


class MemberAccuracy:
    """Accuracy (in %) of ``BasicClassificationPerformanceEvaluator`` as used by
    ARF/SRP to weight votes: ``getPerformanceMeasurements()[1]``."""

    __slots__ = ("n", "correct")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.n = 0.0
        self.correct = 0.0

    def add(self, votes: np.ndarray, y: int, weight: float = 1.0) -> None:
        self.add_pred(predicted_class(votes), y, weight)

    def add_pred(self, pred: int, y: int, weight: float = 1.0) -> None:
        if weight > 0.0:
            self.n += 1.0
            self.correct += weight if pred == y else 0.0

    def accuracy_percent(self) -> float:
        return (self.correct / self.n) * 100.0 if self.n > 0 else math.nan


class Ensemble:
    """Interface: ``predict_votes(x)`` then ``train(x, y)`` (test-then-train)."""

    name = "ensemble"

    def set_schema(self, schema) -> None:
        self.schema = schema
        self.n_classes = schema.n_classes

    def predict_votes(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def train(self, x: np.ndarray, y: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def members(self) -> list:
        """Current base trees (for tree-size statistics)."""
        raise NotImplementedError

    def tree_sizes(self) -> np.ndarray:
        return np.array([t.n_nodes for t in self.members()], dtype=np.float64)

    def split_counts(self) -> dict:
        ms = self.members()
        return {
            "splits_periodic": int(sum(getattr(t, "n_splits_periodic", 0) for t in ms)),
            "splits_adaptive": int(sum(getattr(t, "n_splits_adaptive", 0) for t in ms)),
        }
