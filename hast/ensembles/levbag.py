"""MOA ``meta.LeveragingBag`` (Bifet et al., 2010), default variant (no output codes)."""
from __future__ import annotations

import numpy as np

from hast.core.fastrandom import rng_poisson
from hast.ensembles.base import ADWINCD, Ensemble, java_random


class LeveragingBag(Ensemble):
    name = "LevBag"

    def __init__(self, base_learner, ensemble_size: int = 100, weight_shrink: float = 6.0,
                 delta_adwin: float = 0.002, random_seed: int = 1):
        self.base_learner = base_learner
        self.ensemble_size = ensemble_size
        self.w = weight_shrink
        self.delta_adwin = delta_adwin
        self.random_seed = random_seed
        self.ensemble = None
        self.n_changes = 0

    def set_schema(self, schema) -> None:
        super().set_schema(schema)
        self.base_learner.set_schema(schema)
        self.rnd = java_random(self.random_seed)
        self.ensemble = [self.base_learner.clone() for _ in range(self.ensemble_size)]
        self.adwin = [ADWINCD(self.delta_adwin) for _ in range(self.ensemble_size)]

    def members(self):
        return self.ensemble

    def predict_votes(self, x):
        acc = np.zeros(self.n_classes)
        for t in self.ensemble:
            t.vote_into(x, acc, -1.0, True)
        return acc

    def train(self, x, y):
        change = False
        for i, t in enumerate(self.ensemble):
            k = rng_poisson(self.rnd, self.w)
            correct = t.train_check(x, y, float(k)) if k > 0 else t.correctly_classifies(x, y)
            # ADError[i].setInput(...) && estimation increased  ==  ADWINChangeDetector semantics
            if self.adwin[i].input(0.0 if correct else 1.0):
                change = True
        if change:
            self.n_changes += 1
            mx = 0.0
            imax = -1
            for i, ad in enumerate(self.adwin):
                e = ad.estimation()
                if mx < e:
                    mx = e
                    imax = i
            if imax != -1:
                self.ensemble[imax].reset()
                self.adwin[imax] = ADWINCD(self.delta_adwin)
