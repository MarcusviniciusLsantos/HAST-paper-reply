"""``meta.AdaptiveRandomForest`` (Gomes et al., 2017) as modified by the authors
(base tree typed as ``ARFBaseTree`` so that ARFHLAST can be plugged in)."""
from __future__ import annotations

import math

import numpy as np

from hast.core.fastrandom import rng_poisson
from hast.ensembles.base import ADWINCD, Ensemble, MemberAccuracy, java_random, java_round


class _ARFMember:
    __slots__ = ("tree", "evaluator", "drift", "warning", "bkg", "is_bkg", "n_drifts", "n_warnings")

    def __init__(self, tree, drift_delta, warning_delta, is_bkg=False):
        self.tree = tree
        self.evaluator = MemberAccuracy()
        self.drift = ADWINCD(drift_delta)
        self.warning = ADWINCD(warning_delta)
        self.bkg = None
        self.is_bkg = is_bkg
        self.n_drifts = 0
        self.n_warnings = 0

    def reset(self):
        if self.bkg is not None:
            self.tree = self.bkg.tree
            self.drift = self.bkg.drift
            self.warning = self.bkg.warning
            self.evaluator = self.bkg.evaluator
            self.bkg = None
        else:
            self.tree.reset()
            self.drift = self.drift.copy()
        self.evaluator.reset()

    def train(self, x, y, k):
        correct = self.tree.train_check(x, y, float(k))
        if self.bkg is not None:
            self.bkg.tree.train(x, y, 1.0)  # the background tree sees weight 1 (as in MOA)
        if not self.is_bkg:
            err = 0.0 if correct else 1.0
            if self.warning.input(err):
                self.n_warnings += 1
                self.bkg = _ARFMember(self.tree.clone(), self.drift.delta, self.warning.delta, is_bkg=True)
                self.warning = self.warning.copy()
            if self.drift.input(err):
                self.n_drifts += 1
                self.reset()


class AdaptiveRandomForest(Ensemble):
    name = "ARF"

    def __init__(self, tree, ensemble_size: int = 100, m_features_mode: str = "percentage",
                 m_features: int = 60, lam: float = 6.0, drift_delta: float = 1e-3,
                 warning_delta: float = 1e-2, weighted_vote: bool = True, random_seed: int = 1):
        self.tree_proto = tree
        self.ensemble_size = ensemble_size
        self.m_features_mode = m_features_mode
        self.m_features = m_features
        self.lam = lam
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.weighted_vote = weighted_vote
        self.random_seed = random_seed
        self.ensemble = None

    def _subspace_size(self, n: int) -> int:
        m = self.m_features
        mode = self.m_features_mode
        if mode == "sqrt":
            m = java_round(math.sqrt(n)) + 1
        elif mode == "msqrt":
            m = n - java_round(math.sqrt(n) + 1)
        elif mode == "percentage":
            percent = (100 + m) / 100.0 if m < 0 else m / 100.0
            m = java_round(n * percent)
        if m < 0:
            m = n + m
        if m <= 0:
            m = 1
        if m > n:
            m = n
        return m

    def set_schema(self, schema) -> None:
        super().set_schema(schema)
        self.rnd = java_random(self.random_seed)
        self.subspace = self._subspace_size(schema.n_attributes)
        self.tree_proto.subspace_size = self.subspace
        self.tree_proto.set_schema(schema)
        self.ensemble = [_ARFMember(self.tree_proto.clone(), self.drift_delta, self.warning_delta)
                         for _ in range(self.ensemble_size)]
        self._last_x = None
        self._last_preds = None

    def members(self):
        return [m.tree for m in self.ensemble]

    def _vote_weight(self, m) -> float:
        a = m.evaluator.accuracy_percent()
        return a if (self.weighted_vote and a > 0.0) else -1.0

    def predict_votes(self, x):
        acc = np.zeros(self.n_classes)
        preds = []
        for m in self.ensemble:
            pred, _ = m.tree.vote_into(x, acc, self._vote_weight(m), True)
            preds.append(pred)
        # the per-member predictions are reused by train() (same instance, unchanged trees)
        self._last_x, self._last_preds = x, preds
        return acc

    def _member_preds(self, x):
        if self._last_x is x:
            preds = self._last_preds
        else:
            dummy = np.zeros(self.n_classes)
            preds = [m.tree.vote_into(x, dummy, -1.0, False)[0] for m in self.ensemble]
        self._last_x = self._last_preds = None
        return preds

    def train(self, x, y):
        for m, pred in zip(self.ensemble, self._member_preds(x)):
            m.evaluator.add_pred(pred, y)
            k = rng_poisson(self.rnd, self.lam)
            if k > 0:
                m.train(x, y, k)
