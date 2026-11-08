"""Hoeffding Tree (MOA ``trees.HoeffdingTree``) and the Hoeffding Adaptive
Splitting Trees HLAST / HLAST_D (Nowak Assis et al., DMKD 2026).

The structure follows the authors' ``HLAST.java`` line by line:

* periodic split attempt every ``grace_period`` weight seen by a leaf, accepted
  when ``G(Xa) - G(Xb) > eps`` or ``eps < tie_threshold`` (Hoeffding Tree);
* HLAST only: adaptive split attempt whenever the leaf change detector fires,
  accepted when ``G(Xa) > 0`` (Algorithm 1, line 18). The acceptance rule is
  isolated in :meth:`HLAST.adaptive_split_accepts` so regularized variants
  (HLAST-R) only need to override it.

Subclasses customise which attributes a leaf monitors (ARF / ARTE random
subspaces) through :meth:`HoeffdingTree._init_leaf_attributes`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hast.core.jrandom import JavaRandom
from hast.core.mstats import entropy as _entropy  # noqa: F401  (re-export for users)
from hast.detectors import HDDM_A, ChangeDetector
from hast.trees import kernels as K

_EMPTY_I = np.zeros(0, dtype=np.int64)
_EMPTY_K = np.zeros(0, dtype=np.int8)


@dataclass
class Schema:
    """Input attributes seen by a tree (class excluded)."""

    is_nominal: np.ndarray  # bool[d]
    n_values: np.ndarray  # int[d], 0 for numeric
    n_classes: int

    @property
    def n_attributes(self) -> int:
        return len(self.is_nominal)

    @classmethod
    def from_header(cls, header) -> "Schema":
        return cls(header.is_nominal.copy(), header.n_values.copy(), header.n_classes)

    def subset(self, idx: np.ndarray) -> "Schema":
        return Schema(self.is_nominal[idx].copy(), self.n_values[idx].copy(), self.n_classes)


# --------------------------------------------------------------------------- nodes
class Node:
    __slots__ = ("cls", "clen")

    def __init__(self, cls: np.ndarray, clen: int):
        self.cls = cls  # float64[C] (zero padded)
        self.clen = clen  # MOA DoubleVector length

    def add_class(self, c: int, w: float) -> None:
        self.cls[c] += w
        if c + 1 > self.clen:
            self.clen = c + 1

    def weight_seen(self) -> float:
        return K.dv_sum(self.cls, self.clen)

    def is_pure(self) -> bool:
        return np.count_nonzero(self.cls[: self.clen]) < 2

    def promise(self) -> float:
        total = self.weight_seen()
        if total > 0.0:
            return total - self.cls[K.dv_max_index(self.cls, self.clen)]
        return 0.0


class SplitNode(Node):
    __slots__ = ("kind", "att", "value", "children", "max_branches")

    def __init__(self, kind: int, att: int, value: float, cls, clen, n_splits: int):
        super().__init__(cls, clen)
        self.kind = kind
        self.att = att
        self.value = value
        self.children: list = [None] * n_splits
        self.max_branches = -1 if kind == K.SPLIT_NOMINAL_MULTIWAY else 2

    def branch(self, x) -> int:
        v = x[self.att]
        if v != v:  # missing
            return -1
        if self.kind == K.SPLIT_NUMERIC_BINARY:
            return 0 if v <= self.value else 1  # equalsPassesTest
        if self.kind == K.SPLIT_NOMINAL_MULTIWAY:
            return int(v)
        return 0 if int(v) == self.value else 1

    def get_child(self, i: int):
        return self.children[i] if 0 <= i < len(self.children) else None

    def set_child(self, i: int, node) -> None:
        if self.max_branches >= 0 and i >= self.max_branches:
            raise IndexError(i)
        if i >= len(self.children):
            self.children.extend([None] * (i + 1 - len(self.children)))
        self.children[i] = node


class InactiveLeaf(Node):
    __slots__ = ()


class ActiveLeaf(Node):
    __slots__ = ("att", "kind", "NUM", "NOM", "NLEN", "w_last", "mc_nb", "detector", "subspace")

    def __init__(self, cls, clen, detector: ChangeDetector | None, subspace: int = -1):
        super().__init__(cls, clen)
        self.att = None  # monitored attributes (sorted), None until first instance
        self.kind = _EMPTY_K
        self.NUM = None
        self.NOM = None
        self.NLEN = None
        self.w_last = self.weight_seen()  # weightSeenAtLastSplitEvaluation
        self.mc_nb = np.zeros(2)  # [mcCorrectWeight, nbCorrectWeight]
        self.detector = detector
        self.subspace = subspace  # ARF / ARTE: number of attributes to draw


# --------------------------------------------------------------------------- tree
class HoeffdingTree:
    """MOA ``HoeffdingTree`` with InfoGain, Gaussian numeric estimator (10 bins),
    multiway nominal splits and NBAdaptive leaves (MOA defaults)."""

    name = "HT"
    randomizable = False

    def __init__(self, grace_period: int = 200, split_confidence: float = 1e-7,
                 tie_threshold: float = 0.05, binary_splits: bool = False,
                 no_pre_prune: bool = False, leaf_prediction: str = "NBA", nb_threshold: int = 0,
                 num_bins: int = 10, min_branch_frac: float = 0.01,
                 memory_estimate_period: int = 1_000_000, max_byte_size: int = 33_554_432,
                 random_seed: int = 1):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_splits = binary_splits
        self.no_pre_prune = no_pre_prune
        if leaf_prediction not in ("MC", "NB", "NBA"):
            raise ValueError(leaf_prediction)
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.num_bins = num_bins
        self.min_branch_frac = min_branch_frac
        self.memory_estimate_period = memory_estimate_period
        self.max_byte_size = max_byte_size
        self.random_seed = random_seed
        self.schema: Schema | None = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def params(self) -> dict:
        keys = ("grace_period", "split_confidence", "tie_threshold", "binary_splits", "no_pre_prune",
                "leaf_prediction", "nb_threshold", "num_bins", "min_branch_frac",
                "memory_estimate_period", "max_byte_size", "random_seed")
        return {k: getattr(self, k) for k in keys}

    def clone(self) -> "HoeffdingTree":
        """Fresh, untrained copy with the same options (MOA ``copy()`` before training)."""
        t = self.__class__(**self.params())
        if self.schema is not None:
            t.set_schema(self.schema)
        return t

    def set_schema(self, schema: Schema) -> None:
        self.schema = schema
        C = schema.n_classes
        self._C = C
        nom = schema.n_values[schema.is_nominal]
        self._vmax = int(nom.max()) if nom.size else 0
        self._max_branches = max(2, self._vmax)
        self._scratch = np.zeros(C)
        # placeholders for leaves whose observers are not created yet
        self._empty_num = np.zeros((5, 0, C))
        self._empty_nom = np.zeros((0, 1, C))
        self._empty_nlen = np.zeros((0, C), dtype=np.int64)

    def reset(self) -> None:
        self.root: Node | None = None
        self.decision_node_count = 0
        self.active_leaf_count = 0
        self.inactive_leaf_count = 0
        self.inactive_leaf_byte_size_estimate = 0.0
        self.active_leaf_byte_size_estimate = 0.0
        self.byte_size_estimate_overhead_fraction = 1.0
        self.growth_allowed = True
        self.training_weight_seen = 0.0
        self.classifier_random = JavaRandom(self.random_seed) if self.randomizable else None
        # diagnostics
        self.n_splits_periodic = 0
        self.n_splits_adaptive = 0
        self.n_attempts_periodic = 0
        self.n_attempts_adaptive = 0

    # ------------------------------------------------------------------ hooks
    def _new_detector(self) -> ChangeDetector | None:
        return None

    def _leaf_subspace_size(self) -> int:
        return -1

    def _init_leaf_attributes(self, leaf: ActiveLeaf) -> np.ndarray:
        """Attributes monitored by a new active leaf (all of them in HT)."""
        return np.arange(self.schema.n_attributes, dtype=np.int64)

    def _numeric_split_points(self, leaf: ActiveLeaf, points, npoints) -> None:
        K.gaussian_split_points_all(leaf.NUM, leaf.kind, self._C, self.num_bins, points, npoints)

    def _skip_multiway_child(self) -> bool:
        return False

    # ------------------------------------------------------------------ nodes
    def _new_leaf(self, cls: np.ndarray | None = None, clen: int = 0) -> ActiveLeaf:
        arr = np.zeros(self._C)
        if cls is not None:
            arr[:clen] = cls[:clen]
        leaf = ActiveLeaf(arr, clen, self._new_detector(), self._leaf_subspace_size())
        leaf.NUM = self._empty_num
        leaf.NOM = self._empty_nom
        leaf.NLEN = self._empty_nlen
        return leaf

    def _ensure_observers(self, leaf: ActiveLeaf) -> None:
        att = self._init_leaf_attributes(leaf)
        att = np.sort(att)
        leaf.att = att
        leaf.kind = self.schema.is_nominal[att].astype(np.int8)
        k = att.shape[0]
        leaf.NUM = np.zeros((5, k, self._C))
        if self._vmax:
            leaf.NOM = np.zeros((k, self._vmax, self._C))
            leaf.NLEN = np.zeros((k, self._C), dtype=np.int64)
        else:
            leaf.NOM = np.zeros((k, 1, self._C))
            leaf.NLEN = np.zeros((k, self._C), dtype=np.int64)

    # ------------------------------------------------------------------ learning
    def _filter(self, x):
        """``filterInstanceToLeaf``: returns (node, parent, parent_branch)."""
        node = self.root
        parent = None
        branch = -1
        # SplitNode.branch / get_child inlined: this loop dominates the runtime
        while type(node) is SplitNode:
            v = x[node.att]
            if v != v:  # missing value
                return node, parent, branch
            kd = node.kind
            if kd == 0:  # numeric binary (equalsPassesTest)
                b = 0 if v <= node.value else 1
            elif kd == 1:  # nominal multiway
                b = int(v)
            else:
                b = 0 if int(v) == node.value else 1
            ch = node.children
            child = ch[b] if b < len(ch) else None
            if child is None:
                return None, node, b
            parent, branch, node = node, b, child
        return node, parent, branch

    def _leaf_learn(self, leaf: ActiveLeaf, x, c: int, w: float) -> None:
        if self.leaf_prediction == "NBA":
            att = leaf.att if leaf.att is not None else _EMPTY_I
            val = K.nb_adaptive_learn(x, c, w, leaf.cls, leaf.clen, att, leaf.kind, leaf.NUM, leaf.NOM,
                                      leaf.NLEN, self._scratch, leaf.mc_nb, self._distribution_input())
            if leaf.detector is not None:
                leaf.detector.input(val)
        leaf.add_class(c, w)
        if leaf.att is None:
            self._ensure_observers(leaf)
        K.observe(x, c, w, leaf.att, leaf.kind, leaf.NUM, leaf.NOM, leaf.NLEN)

    def _distribution_input(self) -> bool:
        return False

    def train(self, x: np.ndarray, y: int, weight: float = 1.0) -> None:
        """``trainOnInstance`` (instances with weight <= 0 are ignored, as in MOA)."""
        if weight <= 0.0:
            return
        self.training_weight_seen += weight
        self._train_impl(x, int(y), weight)

    def _train_impl(self, x, c, w) -> None:
        if self.root is None:
            self.root = self._new_leaf()
            self.active_leaf_count = 1
        node, parent, branch = self._filter(x)
        if node is None:
            node = self._new_leaf()
            parent.set_child(branch, node)
            self.active_leaf_count += 1
        if isinstance(node, ActiveLeaf):
            self._leaf_learn(node, x, c, w)
            if self.growth_allowed:
                self._after_learn(node, parent, branch)
        elif isinstance(node, InactiveLeaf):
            node.add_class(c, w)
        if math.fmod(self.training_weight_seen, self.memory_estimate_period) == 0:
            self._estimate_model_byte_sizes()

    def _after_learn(self, leaf: ActiveLeaf, parent, branch) -> None:
        weight_seen = leaf.weight_seen()
        if weight_seen - leaf.w_last >= self.grace_period:
            self._attempt_to_split(leaf, parent, branch)
            leaf.w_last = weight_seen

    # ------------------------------------------------------------------ splitting
    def _best_split_suggestions(self, leaf: ActiveLeaf) -> list:
        """List of (merit, kind, att, value, dists, lens) sorted ascending by merit
        (Java ``Arrays.sort`` is stable; null split first)."""
        C = self._C
        pre = np.zeros(C)
        pre[: leaf.clen] = leaf.cls[: leaf.clen]
        sugg = []
        if not self.no_pre_prune:
            sugg.append((K.null_split_merit(pre, C, self.min_branch_frac), -1, -1, 0.0, None, None))
        if leaf.att is not None and leaf.att.shape[0] > 0:
            k = leaf.att.shape[0]
            points = np.empty((k, max(self.num_bins, 1)))
            npoints = np.zeros(k, dtype=np.int64)
            self._numeric_split_points(leaf, points, npoints)
            mb = self._max_branches
            merit = np.empty(k)
            kind = np.zeros(k, dtype=np.int64)
            value = np.zeros(k)
            nb = np.zeros(k, dtype=np.int64)
            dists = np.zeros((k, mb, C))
            lens = np.zeros((k, mb), dtype=np.int64)
            valid = np.zeros(k, dtype=np.bool_)
            K.best_split_suggestions(pre, C, leaf.att, leaf.kind, leaf.NUM, leaf.NOM, leaf.NLEN, points,
                                     npoints, self.binary_splits, self.min_branch_frac, merit, kind, value,
                                     nb, dists, lens, valid)
            for j in range(k):
                if valid[j]:
                    sugg.append((merit[j], int(kind[j]), int(leaf.att[j]), value[j],
                                 dists[j, : nb[j]], lens[j, : nb[j]]))
        sugg.sort(key=lambda s: _java_double_key(s[0]))
        return sugg

    def _hoeffding_bound(self, leaf: ActiveLeaf) -> float:
        n_classes = leaf.clen if leaf.clen > 2 else 2
        r = math.log(n_classes) / math.log(2.0)
        return math.sqrt(((r * r) * math.log(1.0 / self.split_confidence)) / (2.0 * leaf.weight_seen()))

    def _attempt_to_split(self, leaf: ActiveLeaf, parent, branch) -> None:
        if leaf.is_pure():
            return
        self.n_attempts_periodic += 1
        sugg = self._best_split_suggestions(leaf)
        if len(sugg) < 2:
            should = len(sugg) > 0
        else:
            hb = self._hoeffding_bound(leaf)
            should = (sugg[-1][0] - sugg[-2][0] > hb) or (hb < self.tie_threshold)
        if should:
            if self._do_split(leaf, parent, branch, sugg[-1]):
                self.n_splits_periodic += 1

    def _do_split(self, leaf: ActiveLeaf, parent, branch, decision) -> bool:
        merit, kind, att, value, dists, lens = decision
        if kind < 0:  # pre-pruning: null split wins
            self._deactivate(leaf, parent, branch)
            split = False
        else:
            n_splits = dists.shape[0]
            node = SplitNode(kind, att, value, leaf.cls.copy(), leaf.clen, n_splits)
            for i in range(n_splits):
                if kind == K.SPLIT_NOMINAL_MULTIWAY and self._skip_multiway_child():
                    continue
                node.set_child(i, self._new_leaf(dists[i], int(lens[i])))
            self.active_leaf_count -= 1
            self.decision_node_count += 1
            self.active_leaf_count += n_splits
            if parent is None:
                self.root = node
            else:
                parent.set_child(branch, node)
            split = True
        self._enforce_tracker_limit()
        return split

    # ------------------------------------------------------------------ prediction
    def _leaf_votes(self, node, x) -> np.ndarray:
        if isinstance(node, ActiveLeaf):
            if self.leaf_prediction == "MC" or (self.leaf_prediction == "NBA" and node.mc_nb[0] > node.mc_nb[1]):
                return node.cls[: node.clen].copy()
            if self.leaf_prediction == "NB" and node.weight_seen() < self.nb_threshold:
                return node.cls[: node.clen].copy()
            out = np.empty(node.clen)
            att = node.att if node.att is not None else _EMPTY_I
            K.naive_bayes(x, node.cls, node.clen, att, node.kind, node.NUM, node.NOM, node.NLEN, out)
            return out
        return node.cls[: node.clen].copy()

    def predict_votes(self, x: np.ndarray) -> np.ndarray:
        """``getVotesForInstance``."""
        if self.root is None:
            return np.zeros(self.schema.n_classes)
        node, parent, _ = self._filter(x)
        if node is None:
            node = parent
        return self._leaf_votes(node, x)

    def correctly_classifies(self, x, y) -> bool:
        v = self.predict_votes(x)
        return int(K.utils_max_index(v, v.shape[0])) == int(y)

    def vote_into(self, x, acc, weight=-1.0, add=True):
        """Ensemble voting step (same semantics as ``fast.tree_vote_into``)."""
        v = self.predict_votes(x)
        n = v.shape[0]
        pred = int(K.utils_max_index(v, n))
        if not add:
            return pred, False
        s = 0.0
        for e in v:
            s += e
        if s > 0.0:
            sa = 0.0
            for e in v:
                sa += e if e > 0.0 else -e
            inv = 1.0 / sa
            if weight > 0.0:
                for k in range(n):
                    acc[k] += (v[k] * inv) * weight
            else:
                for k in range(n):
                    acc[k] += v[k] * inv
            return pred, True
        return pred, False

    def train_check(self, x, y, weight) -> bool:
        self.train(x, y, weight)
        return self.correctly_classifies(x, y)

    def predict(self, x: np.ndarray) -> int:
        v = self.predict_votes(x)
        return int(K.utils_max_index(v, v.shape[0]))

    # ------------------------------------------------------------------ memory management
    def _find_learning_nodes(self) -> list:
        found = []

        def rec(node, parent, branch):
            if node is None:
                return
            if isinstance(node, (ActiveLeaf, InactiveLeaf)):
                found.append((node, parent, branch))
            if isinstance(node, SplitNode):
                for i, ch in enumerate(node.children):
                    rec(ch, node, i)

        rec(self.root, None, -1)
        return found

    def _deactivate(self, leaf: ActiveLeaf, parent, branch) -> None:
        new = InactiveLeaf(leaf.cls.copy(), leaf.clen)
        if parent is None:
            self.root = new
        else:
            parent.set_child(branch, new)
        self.active_leaf_count -= 1
        self.inactive_leaf_count += 1

    def _activate(self, leaf: InactiveLeaf, parent, branch) -> None:
        new = self._new_leaf(leaf.cls, leaf.clen)
        if parent is None:
            self.root = new
        else:
            parent.set_child(branch, new)
        self.active_leaf_count += 1
        self.inactive_leaf_count -= 1

    def _enforce_tracker_limit(self) -> None:
        est = ((self.active_leaf_count * self.active_leaf_byte_size_estimate
                + self.inactive_leaf_count * self.inactive_leaf_byte_size_estimate)
               * self.byte_size_estimate_overhead_fraction)
        if self.inactive_leaf_count > 0 or est > self.max_byte_size:
            nodes = self._find_learning_nodes()
            nodes.sort(key=lambda t: t[0].promise())  # stable, as Java's TimSort
            max_active = 0
            while max_active < len(nodes):
                max_active += 1
                if ((max_active * self.active_leaf_byte_size_estimate
                     + (len(nodes) - max_active) * self.inactive_leaf_byte_size_estimate)
                        * self.byte_size_estimate_overhead_fraction > self.max_byte_size):
                    max_active -= 1
                    break
            cutoff = len(nodes) - max_active
            for i in range(cutoff):
                if isinstance(nodes[i][0], ActiveLeaf):
                    self._deactivate(*nodes[i])
            for i in range(cutoff, len(nodes)):
                if isinstance(nodes[i][0], InactiveLeaf):
                    self._activate(*nodes[i])

    def _estimate_model_byte_sizes(self) -> None:
        """Approximation of MOA's SizeOf-based estimate (see docs/port_notes.md)."""
        nodes = self._find_learning_nodes()
        tot_a = tot_i = 0
        for node, _, _ in nodes:
            if isinstance(node, ActiveLeaf):
                tot_a += _approx_leaf_bytes(node, self._C)
            else:
                tot_i += 64 + 8 * self._C
        if tot_a > 0:
            self.active_leaf_byte_size_estimate = tot_a / self.active_leaf_count
        if tot_i > 0:
            self.inactive_leaf_byte_size_estimate = tot_i / self.inactive_leaf_count
        actual = self.measure_byte_size()
        est = (self.active_leaf_count * self.active_leaf_byte_size_estimate
               + self.inactive_leaf_count * self.inactive_leaf_byte_size_estimate)
        self.byte_size_estimate_overhead_fraction = actual / est if est else float("inf")
        if actual > self.max_byte_size:
            self._enforce_tracker_limit()

    def measure_byte_size(self) -> int:
        total = 200
        stack = [self.root] if self.root is not None else []
        while stack:
            n = stack.pop()
            if n is None:
                continue
            if isinstance(n, SplitNode):
                total += 96 + 8 * self._C + 8 * len(n.children)
                stack.extend(n.children)
            elif isinstance(n, ActiveLeaf):
                total += _approx_leaf_bytes(n, self._C)
            else:
                total += 64 + 8 * self._C
        return total

    # ------------------------------------------------------------------ measurements
    @property
    def n_nodes(self) -> int:
        return self.decision_node_count + self.active_leaf_count + self.inactive_leaf_count

    def depth(self) -> int:
        def rec(n):
            if isinstance(n, SplitNode):
                return 1 + max((rec(c) for c in n.children if c is not None), default=0)
            return 0
        return rec(self.root) if self.root is not None else 0


def _approx_leaf_bytes(leaf: ActiveLeaf, C: int) -> int:
    size = 160 + 8 * C
    if leaf.att is not None:
        for kd in leaf.kind:
            size += 400 + 56 * C if kd == K.KIND_NUMERIC else 200 + 32 * C
    return size


def _java_double_key(m: float):
    """Sort key reproducing ``Double.compare`` (-0.0 < 0.0, NaN largest)."""
    if m != m:
        return (1, 0.0, 0)
    return (0, m, 1 if math.copysign(1.0, m) > 0 else 0)


# --------------------------------------------------------------------------- HLAST
class HLAST(HoeffdingTree):
    """Hoeffding Local Adaptive Splitting Tree.

    ``distribution=False`` (HLAST): the leaf detector monitors the error of the
    leaf predictor. ``distribution=True`` (HLAST_D): it monitors the entropy
    (purity) of the leaf class distribution.
    """

    name = "HLAST"

    def __init__(self, detector: ChangeDetector | None = None, distribution: bool = False, **kw):
        self.detector_template = detector if detector is not None else HDDM_A()
        self.distribution = distribution
        super().__init__(**kw)
        if self.leaf_prediction != "NBA":
            raise ValueError("HLAST leaf detectors are only fed by NBAdaptive leaves (as in the Java code)")

    def params(self) -> dict:
        p = super().params()
        p.update(detector=self.detector_template, distribution=self.distribution)
        return p

    def _new_detector(self) -> ChangeDetector:
        return self.detector_template.copy()

    def _distribution_input(self) -> bool:
        return self.distribution

    def _after_learn(self, leaf: ActiveLeaf, parent, branch) -> None:
        weight_seen = leaf.weight_seen()
        if leaf.detector.get_change():
            self._attempt_to_split_adaptive(leaf, parent, branch)
        elif weight_seen - leaf.w_last >= self.grace_period:
            self._attempt_to_split(leaf, parent, branch)
            leaf.w_last = weight_seen

    def adaptive_split_accepts(self, leaf: ActiveLeaf, suggestions: list) -> bool:
        """Acceptance rule of the adaptive path (Algorithm 1, line 18): ``G(Xa) > 0``.

        ``suggestions`` is sorted ascending by merit and holds at least two
        entries. Override in regularized variants (``R(l, Xa)``).
        """
        return suggestions[-1][0] > 0.0

    def _attempt_to_split_adaptive(self, leaf: ActiveLeaf, parent, branch) -> None:
        # attemptToSplitLAST: no purity check, as in the Java code
        self.n_attempts_adaptive += 1
        sugg = self._best_split_suggestions(leaf)
        if len(sugg) < 2:
            should = len(sugg) > 0
        else:
            should = self.adaptive_split_accepts(leaf, sugg)
        if should:
            if self._do_split(leaf, parent, branch, sugg[-1]):
                self.n_splits_adaptive += 1


def HLAST_D(**kw) -> HLAST:
    """HLAST monitoring class-distribution purity (``-D`` flag)."""
    kw.setdefault("distribution", True)
    return HLAST(**kw)
