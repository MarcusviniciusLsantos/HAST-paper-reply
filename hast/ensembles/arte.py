"""Adaptive Random Tree Ensemble (Paim & Enembreck, KBS 2024), port of the
authors' ``meta/ARTE.java``.

* each member gets a random subspace size in [2, n] (drawn once at start with
  ``Random(n + numClasses)``, and again with a per-instance seed after a reset);
* online bagging with Poisson(6); ADWIN (1e-3) drift detector per member;
* only members whose accuracy over the last 400 trained instances is at least
  the ensemble average vote (unweighted).
"""
from __future__ import annotations

import numpy as np

from hast.core.fastrandom import rng_next_int, rng_poisson, rng_set_seed
from hast.core.jrandom import java_double_to_string_integral, java_string_hash
from hast.ensembles.base import ADWINCD, Ensemble, java_random


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


class _ARTEMember:
    __slots__ = ("tree", "drift", "acc_window", "W", "arr", "last_index", "local_random", "owner", "n_drifts")

    def __init__(self, owner, tree, window):
        self.owner = owner
        self.tree = tree
        self.drift = ADWINCD(owner.drift_delta)
        self.acc_window = 0.0
        self.W = window
        self.arr = None
        self.last_index = -1
        self.local_random = java_random(0)
        self.n_drifts = 0

    def reset(self) -> None:
        self.tree.reset()
        self.drift = self.drift.copy()
        self.arr = None
        o = self.owner
        self.tree.subspace_size = rng_next_int(self.local_random, o.max_value + 1 - o.min_value) + o.min_value

    def train(self, x, y, k):
        correct = self.tree.train_check(x, y, float(k))
        if self.drift.input(0.0 if correct else 1.0):
            self.n_drifts += 1
            self.reset()
        self._update_window(correct)

    def _update_window(self, correct: bool) -> None:
        W = self.W
        if self.arr is None:
            self.arr = [0] * (W + 2)
            self.last_index = -1
        a = self.arr
        if self.last_index == W - 1:
            self.last_index = -1
        if a[W] < W:
            a[W] += 1
        self.last_index += 1
        a[W + 1] -= a[self.last_index]
        a[self.last_index] = 1 if correct else 0
        a[W + 1] += a[self.last_index]
        self.acc_window = a[W + 1] / a[W]


class ARTE(Ensemble):
    name = "ARTE"

    def __init__(self, tree, ensemble_size: int = 100, lam: float = 6.0, drift_delta: float = 1e-3,
                 window: int = 400, random_seed: int = 1):
        self.tree_proto = tree
        self.ensemble_size = ensemble_size
        self.lam = lam
        self.drift_delta = drift_delta
        self.window = window
        self.random_seed = random_seed
        self.ensemble = None

    def set_schema(self, schema) -> None:
        super().set_schema(schema)
        self.rnd = java_random(self.random_seed)
        self.instances_seen = 0
        self.avg_acc = 0.0
        n = schema.n_attributes
        subspace_random = java_random(n + schema.n_classes)
        self.min_value = 2
        self.max_value = n
        self.tree_proto.set_schema(schema)
        self.ensemble = []
        for _ in range(self.ensemble_size):
            self.tree_proto.subspace_size = rng_next_int(subspace_random, self.max_value + 1 - self.min_value) + self.min_value
            self.ensemble.append(_ARTEMember(self, self.tree_proto.clone(), self.window))

    def members(self):
        return [m.tree for m in self.ensemble]

    def predict_votes(self, x):
        acc = np.zeros(self.n_classes)
        avg = self.avg_acc
        for m in self.ensemble:
            m.tree.vote_into(x, acc, -1.0, m.acc_window >= avg)
        return acc

    def train(self, x, y):
        self.instances_seen += 1
        base = java_string_hash(java_double_to_string_integral(float(self.instances_seen)))
        total = 0.0
        for i, m in enumerate(self.ensemble):
            rng_set_seed(m.local_random, _i32(base + i))
            k = rng_poisson(self.rnd, self.lam)
            if k > 0:
                m.train(x, y, k)
            total += m.acc_window
        if total > 0.0:
            self.avg_acc = total / len(self.ensemble)
