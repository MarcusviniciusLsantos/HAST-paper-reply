"""MOA ``meta.StreamingRandomPatches`` (Gomes et al., 2019), Random Patches mode."""
from __future__ import annotations

import numpy as np

from hast.core.fastrandom import rng_next_int, rng_poisson
from hast.ensembles.base import ADWINCD, Ensemble, MemberAccuracy, java_random, java_round

TRAIN_RANDOM_SUBSPACES = 0
TRAIN_RESAMPLING = 1
TRAIN_RANDOM_PATCHES = 2


def local_random_k_combinations(k, length, n_comb, rnd):
    combs = []
    for _ in range(n_comb):
        comb = list(range(length))
        for _ in range(length - k):
            comb.pop(rng_next_int(rnd, len(comb)))
        combs.append(comb)
    return combs


def all_k_combinations(k, length):
    out = []

    def inner(offset, kk, comb):
        if kk == 0:
            out.append(list(comb))
            return
        for i in range(offset, length - kk + 1):
            comb.append(i)
            inner(i + 1, kk - 1, comb)
            comb.pop()

    inner(0, k, [])
    return out


class _SRPMember:
    __slots__ = ("tree", "features", "evaluator", "drift", "warning", "bkg", "is_bkg", "n_drifts",
                 "n_warnings", "owner")

    def __init__(self, owner, tree, features, is_bkg=False):
        self.owner = owner
        self.tree = tree
        self.features = features  # np.int64 indices (class excluded) or None
        if features is not None:
            tree.set_schema(owner.schema.subset(features))
        self.evaluator = MemberAccuracy()
        self.drift = ADWINCD(owner.drift_delta)
        self.warning = ADWINCD(owner.warning_delta)
        self.bkg = None
        self.is_bkg = is_bkg
        self.n_drifts = 0
        self.n_warnings = 0

    def sub(self, x):
        return x if self.features is None else np.ascontiguousarray(x[self.features])

    def vote_into(self, x, acc, weight, add):
        return self.tree.vote_into(self.sub(x), acc, weight, add)

    def _reset_subset(self, rnd):
        """``applySubsetResetStrategy``: new random subset of the same size."""
        n = self.owner.schema.n_attributes
        f = list(range(n))
        # MOA removes (numAttributes - featureIndexes.length) entries; both counts include the class
        for _ in range((n + 1) - (len(self.features) + 1)):
            f.pop(rng_next_int(rnd, len(f)))
        return np.array(f, dtype=np.int64)

    def reset(self, rnd):
        if self.bkg is not None:
            # MOA quirk (replicated on purpose): unlike ARF, SRP does not clear
            # ``bkgLearner`` here. The stale background member keeps pointing at
            # the promoted tree, which is therefore trained twice per instance
            # until the next warning creates a new background learner.
            self.tree = self.bkg.tree
            self.drift = self.bkg.drift
            self.warning = self.bkg.warning
            self.evaluator = self.bkg.evaluator
            self.evaluator.reset()
            self.features = self.bkg.features
        else:
            self.tree.reset()
            self.evaluator.reset()
            self.drift = self.drift.copy()
            if self.features is not None:
                self.features = self._reset_subset(rnd)
                self.tree.set_schema(self.owner.schema.subset(self.features))

    def train(self, x, y, w, rnd):
        xs = self.sub(x)
        correct = self.tree.train_check(xs, y, w)
        if self.bkg is not None:
            self.bkg.train(x, y, w, rnd)
        if not self.is_bkg:
            err = 0.0 if correct else 1.0
            if self.warning.input(err):
                self.n_warnings += 1
                self._trigger_warning(rnd)
            if self.drift.input(err):
                self.n_drifts += 1
                self.reset(rnd)

    def _trigger_warning(self, rnd):
        bkg_tree = self.tree.clone()
        bkg_tree.reset()
        feats = self._reset_subset(rnd) if self.features is not None else None
        self.bkg = _SRPMember(self.owner, bkg_tree, feats, is_bkg=True)
        self.warning = self.warning.copy()


class StreamingRandomPatches(Ensemble):
    name = "SRP"

    def __init__(self, base_learner, ensemble_size: int = 100, subspace_mode: str = "percentage",
                 subspace_size: int = 60, training_method: int = TRAIN_RANDOM_PATCHES, lam: float = 6.0,
                 drift_delta: float = 1e-5, warning_delta: float = 1e-4, weighted_vote: bool = True,
                 random_seed: int = 1):
        self.base_learner = base_learner
        self.ensemble_size = ensemble_size
        self.subspace_mode = subspace_mode
        self.subspace_size = subspace_size
        self.training_method = training_method
        self.lam = lam
        self.drift_delta = drift_delta
        self.warning_delta = warning_delta
        self.weighted_vote = weighted_vote
        self.random_seed = random_seed
        self.ensemble = None

    def set_schema(self, schema) -> None:
        super().set_schema(schema)
        self.rnd = java_random(self.random_seed)
        self.base_learner.set_schema(schema)
        self._init_ensemble(schema.n_attributes)
        self._last_x = None
        self._last_votes = None

    def _init_ensemble(self, n):
        method = self.training_method
        k = self.subspace_size
        subspaces = None
        if method != TRAIN_RESAMPLING:
            if self.subspace_mode == "sqrt":
                k = java_round(n ** 0.5) + 1
            elif self.subspace_mode == "msqrt":
                k = n - java_round(n ** 0.5 + 1)
            elif self.subspace_mode == "percentage":
                percent = (100 + k) / 100.0 if k < 0 else k / 100.0
                k = java_round(n * percent)
                if java_round(n * percent) < 2:
                    k = java_round(n * percent) + 1
            if k < 0:
                k = n + k
            if k != 0 and k < n:
                if n <= 20 or k < 2:
                    if k == 1 and n + 1 > 2:
                        k = 2
                    subspaces = all_k_combinations(k, n)
                    i = 0
                    while len(subspaces) < self.ensemble_size:
                        i = 0 if i == len(subspaces) else i
                        subspaces.append(list(subspaces[i]))
                        i += 1
                else:
                    subspaces = local_random_k_combinations(k, n, self.ensemble_size, self.rnd)
            else:
                method = TRAIN_RESAMPLING
        self.method = method
        self.k = k
        self.ensemble = []
        for i in range(self.ensemble_size):
            tree = self.base_learner.clone()
            if method == TRAIN_RESAMPLING:
                self.ensemble.append(_SRPMember(self, tree, None))
            else:
                sel = rng_next_int(self.rnd, len(subspaces))
                feats = np.array(subspaces.pop(sel), dtype=np.int64)
                self.ensemble.append(_SRPMember(self, tree, feats))

    def members(self):
        return [m.tree for m in self.ensemble]

    def predict_votes(self, x):
        acc = np.zeros(self.n_classes)
        preds = []
        for m in self.ensemble:
            a = m.evaluator.accuracy_percent()
            pred, _ = m.vote_into(x, acc, a if (self.weighted_vote and a > 0.0) else -1.0, True)
            preds.append(pred)
        self._last_x, self._last_votes = x, preds
        return acc

    def train(self, x, y):
        if self._last_x is x:
            preds = self._last_votes
        else:
            dummy = np.zeros(self.n_classes)
            preds = [m.vote_into(x, dummy, -1.0, False)[0] for m in self.ensemble]
        self._last_x = self._last_votes = None
        for m, pred in zip(self.ensemble, preds):
            m.evaluator.add_pred(pred, y)
            if self.method == TRAIN_RANDOM_SUBSPACES:
                m.train(x, y, 1.0, self.rnd)
            else:
                kk = rng_poisson(self.rnd, self.lam)
                if kk > 0:
                    m.train(x, y, float(kk), self.rnd)
