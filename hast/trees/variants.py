"""Base-tree variants used inside ensembles.

* ARF trees (``ARFHoeffdingTree`` / ``ARFHLAST``): every new leaf draws a random
  subset of ``subspace_size`` attributes with the tree's ``java.util.Random``.
* ARTE trees (``ARTEHoeffdingTree`` / ``ARTEHLAST``): random subset per leaf (by
  rejection sampling), numeric splits evaluated at random cut points, and
  nominal multiway splits that keep each branch with probability 1/2.

Faithfulness note: the authors' ``ARFHLAST`` observes only the first
``subspace_size - 1`` drawn attributes (``for j < numAttributes - 1``), whereas
MOA's ``ARFHoeffdingTree`` observes all of them. ``fix_arf_subspace_bug=True``
makes ARFHLAST observe the full subset.
"""
from __future__ import annotations

import numpy as np

from hast.trees import kernels as K
from hast.trees.hoeffding import HLAST, ActiveLeaf, HoeffdingTree


class _ARFMixin:
    randomizable = True

    def _arf_init(self, subspace_size: int, drop_last: bool):
        self.subspace_size = subspace_size
        self._drop_last = drop_last

    def _leaf_subspace_size(self) -> int:
        return self.subspace_size

    def _init_leaf_attributes(self, leaf: ActiveLeaf) -> np.ndarray:
        total = self.schema.n_attributes
        k = leaf.subspace
        if k >= total or k < 0:
            chosen = list(range(total))
        else:
            pool = list(range(total))
            chosen = []
            rnd = self.classifier_random
            for _ in range(k):
                idx = rnd.next_int(len(pool))
                chosen.append(pool.pop(idx))
        if self._drop_last:
            chosen = chosen[:-1]
        return np.array(chosen, dtype=np.int64)


class ARFHoeffdingTree(_ARFMixin, HoeffdingTree):
    """MOA ``ARFHoeffdingTree`` (defaults used by ARF: ``-e 2000000 -g 50 -c 0.01``)."""

    name = "HT"

    def __init__(self, subspace_size: int = 2, **kw):
        self._arf_init(subspace_size, drop_last=False)
        kw.setdefault("grace_period", 50)
        kw.setdefault("split_confidence", 0.01)
        kw.setdefault("memory_estimate_period", 2_000_000)
        super().__init__(**kw)

    def params(self):
        p = super().params()
        p["subspace_size"] = self.subspace_size
        return p


class ARFHLAST(_ARFMixin, HLAST):
    """Authors' ``ARFHLAST`` (HLAST with ARF random subspaces)."""

    def __init__(self, subspace_size: int = 2, fix_arf_subspace_bug: bool = False, **kw):
        self._arf_init(subspace_size, drop_last=not fix_arf_subspace_bug)
        self.fix_arf_subspace_bug = fix_arf_subspace_bug
        kw.setdefault("grace_period", 50)
        kw.setdefault("split_confidence", 0.01)
        kw.setdefault("memory_estimate_period", 2_000_000)
        super().__init__(**kw)

    def params(self):
        p = super().params()
        p["subspace_size"] = self.subspace_size
        p["fix_arf_subspace_bug"] = self.fix_arf_subspace_bug
        return p


class _ARTEMixin:
    """Random subspace per leaf + random cut points (reconstructed observer)."""

    randomizable = True

    def _leaf_subspace_size(self) -> int:
        return self.subspace_size

    def _init_leaf_attributes(self, leaf: ActiveLeaf) -> np.ndarray:
        # ARTEHoeffdingTree.RandomLearningNode: draw with rejection until unique
        total = self.schema.n_attributes
        rnd = self.classifier_random
        chosen: list[int] = []
        for _ in range(leaf.subspace):
            while True:
                a = rnd.next_int(total)
                if a not in chosen:
                    break
            chosen.append(a)
        return np.array(chosen, dtype=np.int64)

    def _numeric_split_points(self, leaf: ActiveLeaf, points, npoints) -> None:
        """Reconstructed ``ARTEAttributeClassObserver`` (see docs/port_notes.md):
        one cut point drawn uniformly in [min, max] of the attribute, as in
        Extremely Randomized Trees. Draws use the tree's ``java.util.Random`` in
        attribute order."""
        rnd = self.classifier_random
        NUM = leaf.NUM
        C = self._C
        for j in range(leaf.kind.shape[0]):
            npoints[j] = 0
            if leaf.kind[j] != K.KIND_NUMERIC:
                continue
            lo, hi = np.inf, -np.inf
            for c in range(C):
                if NUM[0, j, c] > 0.0:
                    lo = min(lo, NUM[3, j, c])
                    hi = max(hi, NUM[4, j, c])
            if lo < np.inf and hi > lo:
                points[j, 0] = lo + rnd.next_double() * (hi - lo)
                npoints[j] = 1

    def _skip_multiway_child(self) -> bool:
        return self.classifier_random.next_double() > 0.5


class ARTEHoeffdingTree(_ARTEMixin, HoeffdingTree):
    """``ARTEHoeffdingTree -e 2000000 -g 100 -c 0.01 -n ARTEAttributeClassObserver``."""

    name = "HT"

    def __init__(self, subspace_size: int = 2, **kw):
        self.subspace_size = subspace_size
        kw.setdefault("grace_period", 100)
        kw.setdefault("split_confidence", 0.01)
        kw.setdefault("memory_estimate_period", 2_000_000)
        super().__init__(**kw)

    def params(self):
        p = super().params()
        p["subspace_size"] = self.subspace_size
        return p


class ARTEHLAST(_ARTEMixin, HLAST):
    """Authors' ``ARTEHLAST``."""

    def __init__(self, subspace_size: int = 2, **kw):
        self.subspace_size = subspace_size
        kw.setdefault("grace_period", 100)
        kw.setdefault("split_confidence", 0.01)
        kw.setdefault("memory_estimate_period", 2_000_000)
        super().__init__(**kw)

    def params(self):
        p = super().params()
        p["subspace_size"] = self.subspace_size
        return p
