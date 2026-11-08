"""Adaptive Regularized Ensemble (Paim & Enembreck, Pattern Recognition Letters
2024) -- RECONSTRUCTION.

The original code is not public. This version follows the description given
in the HAST paper (Sec. 3.4 and 6.2):

* ARF ensemble (random-subspace trees, Poisson(6), ADWIN warning/drift with
  background learners) with **instance selection**: a member trains on an
  instance it misclassifies; an instance it classifies correctly is rejected,
  except that after ``zeta`` rejected instances of the same class the next
  correctly classified instance of that class is used;
* the members' change detectors are fed only with the instances used for
  training ("receive mostly instances incorrectly classified");
* **classifier selection**: only members whose accuracy is at least the
  ensemble average vote (accuracy-weighted, as in ARF).

``zeta`` is not reported in the HAST paper; ``zeta=10`` is our default.
"""
from __future__ import annotations

import numpy as np

from hast.core.fastrandom import rng_poisson
from hast.ensembles.arf import AdaptiveRandomForest


class AdaptiveRegularizedEnsemble(AdaptiveRandomForest):
    name = "ARE"

    def __init__(self, tree, ensemble_size: int = 100, zeta: int = 10, **kw):
        super().__init__(tree, ensemble_size=ensemble_size, **kw)
        self.zeta = zeta

    def set_schema(self, schema) -> None:
        super().set_schema(schema)
        self.rejected = np.zeros((self.ensemble_size, schema.n_classes), dtype=np.int64)
        self.n_trained = 0
        self.n_rejected = 0

    def predict_votes(self, x):
        accs = [m.evaluator.accuracy_percent() for m in self.ensemble]
        valid = [a for a in accs if a == a]
        avg = float(np.mean(valid)) if valid else 0.0
        acc = np.zeros(self.n_classes)
        preds = []
        for m, a in zip(self.ensemble, accs):
            selected = not (a == a and a < avg)
            w = a if (self.weighted_vote and a > 0.0) else -1.0
            pred, _ = m.tree.vote_into(x, acc, w, selected)
            preds.append(pred)
        self._last_x, self._last_preds = x, preds
        return acc

    def train(self, x, y):
        for i, (m, pred) in enumerate(zip(self.ensemble, self._member_preds(x))):
            m.evaluator.add_pred(pred, y)
            k = rng_poisson(self.rnd, self.lam)
            if pred == y:
                if self.rejected[i, y] < self.zeta:
                    self.rejected[i, y] += 1
                    self.n_rejected += 1
                    continue
                self.rejected[i, y] = 0
            if k > 0:
                self.n_trained += 1
                m.train(x, y, k)  # detectors are updated inside, only for trained instances
