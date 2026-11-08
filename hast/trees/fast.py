"""Array-backed (numba) implementation of HT / HLAST / HLAST_D and their ARF/ARTE
variants, with the same behaviour as the readable object implementation in
:mod:`hast.trees.hoeffding` (which serves as its test oracle).

The whole per-instance path (routing, leaf statistics, Naive Bayes, the HDDM_A
leaf detector, grace-period bookkeeping) runs inside numba. Only split
attempts -- rare events -- return to Python, where the split *decision* is
taken; HLAST's adaptive acceptance rule therefore remains a plain Python method
(:meth:`FastTree.adaptive_split_accepts`) that regularized variants override.

Nodes are integer ids. Every active leaf owns a *light* slot (leaf predictor
counters, grace-period weight, HDDM_A state). Its attribute observers live in
a *heavy* slot that is only allocated when the leaf receives its first
instance -- exactly when MOA creates them -- so that multiway splits on
high-cardinality attributes (e.g. 293 airports) do not allocate observers for
children that never see data. Observer arrays are indexed with schema-wide
maps: ``num_idx[a]`` for numeric attributes, ``nom_idx[a]``/``nom_off[a]`` for
nominal ones (counts of all values of all nominal attributes are packed).
"""
from __future__ import annotations

import math

import numpy as np
from numba import boolean, float32, float64, int8, int32, int64, njit
from numba.experimental import jitclass

from hast.core.fastrandom import rng_new, rng_next_double, rng_next_int, rng_set_seed
from hast.core.mstats import NORMAL_CONSTANT, entropy, normal_probability
from hast.trees.kernels import dv_max_index, dv_sum, gaussian_pdf, info_gain_merit, utils_max_index

SPLIT, ACTIVE, INACTIVE = 0, 1, 2
POLICY_ALL, POLICY_ARF, POLICY_ARTE = 0, 1, 2
LEAF_MC, LEAF_NB, LEAF_NBA = 0, 1, 2
NUMERIC_BINARY, NOMINAL_MULTIWAY, NOMINAL_BINARY = 0, 1, 2

_spec = [
    # configuration
    ("C", int64), ("d", int64), ("n_num", int64), ("n_nom", int64), ("nom_total", int64), ("kmax", int64),
    ("num_idx", int64[:]), ("nom_idx", int64[:]), ("nom_off", int64[:]),
    ("leaf_mode", int64), ("nb_threshold", float64),
    ("use_detector", boolean), ("dist_input", boolean),
    ("hd_drift", float64), ("hd_warn", float64), ("hd_two_sided", boolean), ("hd_log_drift", float64),
    ("policy", int64), ("drop_last", boolean), ("subspace", int64), ("grace", float64),
    ("is_nominal", int8[:]), ("n_values", int64[:]), ("rng", int64[:]), ("scratch", float64[:]),
    # nodes
    ("n_nodes", int64), ("root", int64),
    ("ntype", int8[:]), ("satt", int64[:]), ("skind", int8[:]), ("sval", float64[:]),
    ("ch_start", int64[:]), ("ch_n", int64[:]), ("slot", int64[:]),
    ("ncls", float64[:, :]), ("nclen", int64[:]),
    ("CH", int64[:]), ("n_ch", int64),
    # light leaf pool (one per active leaf)
    ("n_slots", int64), ("free", int64[:]), ("n_free", int64),
    ("lsub", int64[:]), ("hslot", int64[:]), ("mc_nb", float64[:, :]), ("w_last", float64[:]),
    ("hd_i", int64[:, :]), ("hd_f", float64[:, :]), ("hd_change", boolean[:]),
    # heavy pool (attribute observers, allocated on the leaf's first instance)
    ("n_h", int64), ("hfree", int64[:]), ("n_hfree", int64),
    ("natt", int64[:]), ("att", int64[:, :]), ("kind", int8[:, :]),
    ("NUM", float64[:, :, :, :]), ("NOM", float32[:, :, :]), ("NLEN", int32[:, :, :]),
    # counters (MOA names)
    ("n_decision", int64), ("n_active", int64), ("n_inactive", int64),
    # cache of the last Naive Bayes evaluation (node, instance) -> votes
    ("nbc_node", int64), ("nbc_x", float64[:]), ("nbc_out", float64[:]),
]


@jitclass(_spec)
class TreeState:
    def __init__(self, C, is_nominal, n_values, leaf_mode, nb_threshold, use_detector, dist_input,
                 hd_drift, hd_warn, hd_two_sided, policy, drop_last, subspace, grace, seed):
        self.C = C
        self.d = is_nominal.shape[0]
        self.is_nominal = is_nominal
        self.n_values = n_values
        self.num_idx = np.full(self.d, -1, dtype=np.int64)
        self.nom_idx = np.full(self.d, -1, dtype=np.int64)
        self.nom_off = np.full(self.d, -1, dtype=np.int64)
        n_num = 0
        n_nom = 0
        tot = 0
        for a in range(self.d):
            if is_nominal[a]:
                self.nom_idx[a] = n_nom
                self.nom_off[a] = tot
                tot += n_values[a]
                n_nom += 1
            else:
                self.num_idx[a] = n_num
                n_num += 1
        self.n_num = n_num
        self.n_nom = n_nom
        self.nom_total = tot
        kmax = self.d
        if policy == POLICY_ARF and 0 <= subspace < self.d:
            kmax = subspace
        self.kmax = max(kmax, 1)
        self.leaf_mode = leaf_mode
        self.nb_threshold = nb_threshold
        self.use_detector = use_detector
        self.dist_input = dist_input
        self.hd_drift = hd_drift
        self.hd_warn = hd_warn
        self.hd_two_sided = hd_two_sided
        self.hd_log_drift = math.log(1.0 / hd_drift)
        self.policy = policy
        self.drop_last = drop_last
        self.subspace = subspace
        self.grace = grace
        self.rng = rng_new(seed)
        self.scratch = np.zeros(C)
        self.nbc_x = np.zeros(self.d)
        self.nbc_out = np.zeros(C)
        self._alloc_nodes(64)
        self.CH = np.empty(256, dtype=np.int64)
        self._alloc_light(16)
        self._alloc_heavy(8)
        self.clear()

    def _alloc_nodes(self, cap):
        self.ntype = np.zeros(cap, dtype=np.int8)
        self.satt = np.zeros(cap, dtype=np.int64)
        self.skind = np.zeros(cap, dtype=np.int8)
        self.sval = np.zeros(cap)
        self.ch_start = np.zeros(cap, dtype=np.int64)
        self.ch_n = np.zeros(cap, dtype=np.int64)
        self.slot = np.zeros(cap, dtype=np.int64)
        self.ncls = np.zeros((cap, self.C))
        self.nclen = np.zeros(cap, dtype=np.int64)

    def _alloc_light(self, cap):
        self.free = np.zeros(cap, dtype=np.int64)
        self.lsub = np.zeros(cap, dtype=np.int64)
        self.hslot = np.zeros(cap, dtype=np.int64)
        self.mc_nb = np.zeros((cap, 2))
        self.w_last = np.zeros(cap)
        self.hd_i = np.zeros((cap, 3), dtype=np.int64)
        self.hd_f = np.zeros((cap, 3))
        self.hd_change = np.zeros(cap, dtype=np.bool_)

    def _alloc_heavy(self, cap):
        self.hfree = np.zeros(cap, dtype=np.int64)
        self.natt = np.zeros(cap, dtype=np.int64)
        self.att = np.zeros((cap, self.kmax), dtype=np.int64)
        self.kind = np.zeros((cap, self.kmax), dtype=np.int8)
        self.NUM = np.zeros((cap, 5, max(self.n_num, 1), self.C))
        # Nominal counts are sums of integer instance weights (1 or Poisson draws): float32
        # holds them exactly up to 2**24, far above any leaf weight in these streams, and
        # halves the memory of high-cardinality attributes (e.g. 293 airports).
        self.NOM = np.zeros((cap, max(self.nom_total, 1), self.C), dtype=np.float32)
        self.NLEN = np.zeros((cap, max(self.n_nom, 1), self.C), dtype=np.int32)

    def clear(self):
        self.n_nodes = 0
        self.root = -1
        self.n_ch = 0
        self.n_slots = 0
        self.n_free = 0
        self.n_h = 0
        self.n_hfree = 0
        self.n_decision = 0
        self.n_active = 0
        self.n_inactive = 0
        self.nbc_node = -1


# --------------------------------------------------------------------------- storage helpers
@njit(error_model="numpy")
def _grow_nodes(T):
    old = T.ntype.shape[0]
    cap = old * 2
    ntype, satt, skind, sval = T.ntype, T.satt, T.skind, T.sval
    ch_start, ch_n, slot, ncls, nclen = T.ch_start, T.ch_n, T.slot, T.ncls, T.nclen
    T._alloc_nodes(cap)
    T.ntype[:old] = ntype
    T.satt[:old] = satt
    T.skind[:old] = skind
    T.sval[:old] = sval
    T.ch_start[:old] = ch_start
    T.ch_n[:old] = ch_n
    T.slot[:old] = slot
    T.ncls[:old] = ncls
    T.nclen[:old] = nclen


@njit(error_model="numpy")
def _grow_light(T):
    old = T.lsub.shape[0]
    free, lsub, hslot, mc_nb, w_last = T.free, T.lsub, T.hslot, T.mc_nb, T.w_last
    hd_i, hd_f, hd_change = T.hd_i, T.hd_f, T.hd_change
    T._alloc_light(old * 2)
    T.free[:old] = free
    T.lsub[:old] = lsub
    T.hslot[:old] = hslot
    T.mc_nb[:old] = mc_nb
    T.w_last[:old] = w_last
    T.hd_i[:old] = hd_i
    T.hd_f[:old] = hd_f
    T.hd_change[:old] = hd_change


@njit(error_model="numpy")
def _grow_heavy(T):
    old = T.natt.shape[0]
    hfree, natt, att, kind, NUM, NOM, NLEN = T.hfree, T.natt, T.att, T.kind, T.NUM, T.NOM, T.NLEN
    T._alloc_heavy(old * 2)
    T.hfree[:old] = hfree
    T.natt[:old] = natt
    T.att[:old] = att
    T.kind[:old] = kind
    T.NUM[:old] = NUM
    T.NOM[:old] = NOM
    T.NLEN[:old] = NLEN


@njit(error_model="numpy")
def _alloc_children(T, n):
    while T.n_ch + n > T.CH.shape[0]:
        new = np.empty(T.CH.shape[0] * 2, dtype=np.int64)
        new[: T.n_ch] = T.CH[: T.n_ch]
        T.CH = new
    start = T.n_ch
    for i in range(n):
        T.CH[start + i] = -1
    T.n_ch += n
    return start


@njit(error_model="numpy")
def _hddm_reset(T, s):
    for k in range(3):
        T.hd_i[s, k] = 0
        T.hd_f[s, k] = 0.0
    T.hd_change[s] = False


@njit(error_model="numpy")
def _alloc_slot(T):
    if T.n_free > 0:
        T.n_free -= 1
        return T.free[T.n_free]
    if T.n_slots == T.lsub.shape[0]:
        _grow_light(T)
    s = T.n_slots
    T.n_slots += 1
    return s


@njit(error_model="numpy")
def _alloc_heavy_slot(T):
    if T.n_hfree > 0:
        T.n_hfree -= 1
        return T.hfree[T.n_hfree]
    if T.n_h == T.natt.shape[0]:
        _grow_heavy(T)
    h = T.n_h
    T.n_h += 1
    return h


@njit(error_model="numpy")
def _free_slot(T, s):
    """Release the light slot of a leaf and its observers (if any)."""
    h = T.hslot[s]
    if h >= 0:
        T.hfree[T.n_hfree] = h
        T.n_hfree += 1
        T.hslot[s] = -1
    T.free[T.n_free] = s
    T.n_free += 1


@njit(error_model="numpy")
def _init_slot(T, s, node):
    T.hslot[s] = -1
    T.lsub[s] = T.subspace
    T.mc_nb[s, 0] = 0.0
    T.mc_nb[s, 1] = 0.0
    T.w_last[s] = dv_sum(T.ncls[node], T.nclen[node])
    _hddm_reset(T, s)


@njit(error_model="numpy")
def _new_node(T):
    if T.n_nodes == T.ntype.shape[0]:
        _grow_nodes(T)
    n = T.n_nodes
    T.n_nodes += 1
    return n


@njit(error_model="numpy")
def new_leaf(T, src, clen):
    n = _new_node(T)
    T.ntype[n] = ACTIVE
    for k in range(T.C):
        T.ncls[n, k] = 0.0
    for k in range(clen):
        T.ncls[n, k] = src[k]
    T.nclen[n] = clen
    s = _alloc_slot(T)
    T.slot[n] = s
    _init_slot(T, s, n)
    return n


# --------------------------------------------------------------------------- HDDM_A (leaf detector)
@njit(error_model="numpy")
def _jsqrt(v):
    return math.sqrt(v) if v >= 0.0 else np.nan


@njit(error_model="numpy")
def _mean_incr(c_min, n_min, total_c, total_n, conf):
    if n_min == total_n:
        return False
    m = (total_n - n_min) / n_min * (1.0 / total_n)
    cota = _jsqrt(m / 2 * math.log(2.0 / conf))
    return total_c / total_n - c_min / n_min >= cota


@njit(error_model="numpy")
def _hddm_input(T, s, value):
    """``HDDM_A_Test.input`` on the detector of leaf slot ``s``."""
    hi = T.hd_i[s]
    hf = T.hd_f[s]
    hi[1] += 1
    hf[1] += value
    if hi[0] == 0:
        hi[0] = hi[1]
        hf[0] = hf[1]
    if hi[2] == 0:
        hi[2] = hi[1]
        hf[2] = hf[1]
    cota = _jsqrt(1.0 / (2 * hi[0]) * T.hd_log_drift)
    cota1 = _jsqrt(1.0 / (2 * hi[1]) * T.hd_log_drift)
    if hf[0] / hi[0] + cota >= hf[1] / hi[1] + cota1:
        hf[0] = hf[1]
        hi[0] = hi[1]
    cota = _jsqrt(1.0 / (2 * hi[2]) * T.hd_log_drift)
    if hf[2] / hi[2] - cota <= hf[1] / hi[1] - cota1:
        hf[2] = hf[1]
        hi[2] = hi[1]
    if _mean_incr(hf[0], hi[0], hf[1], hi[1], T.hd_drift):
        hi[0] = 0
        hi[1] = 0
        hi[2] = 0
        hf[0] = 0.0
        hf[1] = 0.0
        hf[2] = 0.0
        T.hd_change[s] = True
    else:
        T.hd_change[s] = False
    if T.hd_two_sided and hi[2] != hi[1]:
        m = (hi[1] - hi[2]) / hi[2] * (1.0 / hi[1])
        cota = _jsqrt(m / 2 * math.log(2.0 / T.hd_drift))
        if hf[2] / hi[2] - hf[1] / hi[1] >= cota:
            hi[0] = 0
            hi[1] = 0
            hi[2] = 0
            hf[0] = 0.0
            hf[1] = 0.0
            hf[2] = 0.0


# --------------------------------------------------------------------------- observers
@njit(error_model="numpy")
def _init_observers(T, s):
    d = T.d
    k = T.lsub[s]
    chosen = np.empty(d, dtype=np.int64)
    nc = 0
    if T.policy == POLICY_ALL:
        for a in range(d):
            chosen[a] = a
        nc = d
    elif T.policy == POLICY_ARF:
        if k >= d or k < 0:
            for a in range(d):
                chosen[a] = a
            nc = d
        else:
            pool = np.arange(d)
            psize = d
            for i in range(k):
                idx = rng_next_int(T.rng, psize)
                chosen[i] = pool[idx]
                for j in range(idx, psize - 1):
                    pool[j] = pool[j + 1]
                psize -= 1
            nc = k
        if T.drop_last:
            nc -= 1
    else:  # ARTE: rejection sampling of unique indices
        for i in range(k):
            while True:
                a = rng_next_int(T.rng, d)
                unique = True
                for j in range(i):
                    if chosen[j] == a:
                        unique = False
                        break
                if unique:
                    break
            chosen[i] = a
        nc = k
    sel = np.sort(chosen[:nc])
    h = _alloc_heavy_slot(T)
    T.hslot[s] = h
    T.natt[h] = nc
    for j in range(nc):
        a = sel[j]
        T.att[h, j] = a
        T.kind[h, j] = 1 if T.is_nominal[a] else 0
    T.NUM[h] = 0.0
    T.NOM[h] = 0.0
    T.NLEN[h] = 0


@njit(error_model="numpy")
def _observe(T, s, x, c, w):
    # jitclass fields are hoisted into locals: each field access in a loop
    # would otherwise pay an atomic reference-count update
    h = T.hslot[s]
    NUM = T.NUM[h]
    NOM = T.NOM[h]
    NLEN = T.NLEN[h]
    att = T.att[h]
    kind = T.kind[h]
    num_idx = T.num_idx
    nom_idx = T.nom_idx
    nom_off = T.nom_off
    for j in range(T.natt[h]):
        a = att[j]
        v = x[a]
        if math.isnan(v):
            continue
        if kind[j] == 0:
            p = num_idx[a]
            ws = NUM[0, p, c]
            if ws == 0.0:
                NUM[3, p, c] = v
                NUM[4, p, c] = v
            else:
                if v < NUM[3, p, c]:
                    NUM[3, p, c] = v
                if v > NUM[4, p, c]:
                    NUM[4, p, c] = v
            if math.isinf(v):
                continue
            if ws > 0.0:
                ws += w
                NUM[0, p, c] = ws
                last = NUM[1, p, c]
                NUM[1, p, c] = last + w * (v - last) / ws
                NUM[2, p, c] += w * (v - last) * (v - NUM[1, p, c])
            else:
                NUM[1, p, c] = v
                NUM[0, p, c] = w
        else:
            iv = int(v)
            NOM[nom_off[a] + iv, c] += w
            q = nom_idx[a]
            if iv + 1 > NLEN[q, c]:
                NLEN[q, c] = iv + 1


@njit(error_model="numpy")
def _naive_bayes(T, node, x, out):
    s = T.slot[node]
    clen = T.nclen[node]
    cls = T.ncls[node]
    total = dv_sum(cls, clen)
    h = T.hslot[s]
    if h < 0:
        for c in range(clen):
            out[c] = cls[c] / total
        return
    na = T.natt[h]
    NUM = T.NUM[h]
    NOM = T.NOM[h]
    NLEN = T.NLEN[h]
    att = T.att[h]
    kind = T.kind[h]
    num_idx = T.num_idx
    nom_idx = T.nom_idx
    nom_off = T.nom_off
    for c in range(clen):
        p = cls[c] / total
        for j in range(na):
            a = att[j]
            v = x[a]
            if math.isnan(v):
                continue
            if kind[j] == 0:
                q = num_idx[a]
                p *= gaussian_pdf(v, NUM[0, q, c], NUM[1, q, c], NUM[2, q, c])
            else:
                n = NLEN[nom_idx[a], c]
                if n == 0:
                    p *= 0.0
                else:
                    off = nom_off[a]
                    iv = int(v)
                    cnt = NOM[off + iv, c] if iv < n else 0.0
                    tot = 0.0
                    for kk in range(n):
                        tot += NOM[off + kk, c]
                    p *= (cnt + 1.0) / (tot + n)
        out[c] = p


@njit(error_model="numpy")
def _naive_bayes_cached(T, node, x, out):
    """``_naive_bayes`` reusing the previous result when the same leaf (unchanged
    since) is evaluated on the same instance -- e.g. the ensemble vote followed
    by the NBAdaptive bookkeeping of the training step. Bit-identical."""
    clen = T.nclen[node]
    if T.nbc_node == node:
        same = True
        xx = T.nbc_x
        for a in range(T.d):
            if xx[a] != x[a]:
                same = False
                break
        if same:
            for c in range(clen):
                out[c] = T.nbc_out[c]
            return
    _naive_bayes(T, node, x, out)
    T.nbc_node = node
    for a in range(T.d):
        T.nbc_x[a] = x[a]
    for c in range(clen):
        T.nbc_out[c] = out[c]


# --------------------------------------------------------------------------- routing / learning / voting
@njit(error_model="numpy")
def tree_filter(T, x):
    ntype = T.ntype
    satt = T.satt
    skind = T.skind
    sval = T.sval
    CH = T.CH
    ch_start = T.ch_start
    ch_n = T.ch_n
    node = T.root
    parent = -1
    branch = -1
    while node >= 0 and ntype[node] == SPLIT:
        v = x[satt[node]]
        if math.isnan(v):
            return node, parent, branch
        kd = skind[node]
        if kd == NUMERIC_BINARY:
            b = 0 if v <= sval[node] else 1
        elif kd == NOMINAL_MULTIWAY:
            b = int(v)
        else:
            b = 0 if int(v) == sval[node] else 1
        child = CH[ch_start[node] + b] if b < ch_n[node] else -1
        if child < 0:
            return -1, node, b
        parent = node
        branch = b
        node = child
    return node, parent, branch


@njit(error_model="numpy")
def _set_child(T, parent, b, child):
    # children blocks are allocated with the attribute's full number of values
    T.CH[T.ch_start[parent] + b] = child
    if b + 1 > T.ch_n[parent]:
        T.ch_n[parent] = b + 1


@njit(error_model="numpy")
def tree_train(T, x, c, w):
    """``trainOnInstanceImpl`` up to the split decision.

    Returns (status, node): 0 nothing to do, 1 periodic split attempt,
    2 adaptive split attempt (leaf detector fired).
    """
    if T.root < 0:
        T.root = new_leaf(T, T.scratch, 0)
        T.n_active = 1
    node, parent, b = tree_filter(T, x)
    if node < 0:
        node = new_leaf(T, T.scratch, 0)
        _set_child(T, parent, b, node)
        T.n_active += 1
    t = T.ntype[node]
    if t == ACTIVE:
        s = T.slot[node]
        clen = T.nclen[node]
        if T.leaf_mode == LEAF_NBA:
            mc_right = 1.0
            nb_right = 1.0
            if dv_max_index(T.ncls[node], clen) == c:
                T.mc_nb[s, 0] += w
                mc_right = 0.0
            _naive_bayes_cached(T, node, x, T.scratch)
            if utils_max_index(T.scratch, clen) == c:
                T.mc_nb[s, 1] += w
                nb_right = 0.0
            if T.use_detector:
                if T.dist_input:
                    val = entropy(T.ncls[node], clen)
                else:
                    val = mc_right if T.mc_nb[s, 0] > T.mc_nb[s, 1] else nb_right
                _hddm_input(T, s, val)
        T.ncls[node, c] += w
        if c + 1 > T.nclen[node]:
            T.nclen[node] = c + 1
        if T.hslot[s] < 0:
            _init_observers(T, s)
        _observe(T, s, x, c, w)
        T.nbc_node = -1  # leaf statistics changed
        ws = dv_sum(T.ncls[node], T.nclen[node])
        if T.use_detector and T.hd_change[s]:
            return 2, node
        if ws - T.w_last[s] >= T.grace:
            T.w_last[s] = ws
            nz = 0
            for k in range(T.nclen[node]):
                if T.ncls[node, k] != 0.0:
                    nz += 1
            if nz >= 2:
                return 1, node
        return 0, node
    if t == INACTIVE:
        T.ncls[node, c] += w
        if c + 1 > T.nclen[node]:
            T.nclen[node] = c + 1
    return 0, node


@njit(error_model="numpy")
def tree_votes(T, x):
    """``getVotesForInstance``."""
    if T.root < 0:
        return np.zeros(T.C)
    node, parent, _ = tree_filter(T, x)
    if node < 0:
        node = parent
    clen = T.nclen[node]
    out = np.empty(clen)
    if T.ntype[node] == ACTIVE:
        s = T.slot[node]
        use_mc = (T.leaf_mode == LEAF_MC
                  or (T.leaf_mode == LEAF_NBA and T.mc_nb[s, 0] > T.mc_nb[s, 1])
                  or (T.leaf_mode == LEAF_NB and dv_sum(T.ncls[node], clen) < T.nb_threshold))
        if not use_mc:
            _naive_bayes_cached(T, node, x, out)
            return out
    for k in range(clen):
        out[k] = T.ncls[node, k]
    return out


@njit(error_model="numpy")
def tree_correct(T, x, y):
    v = tree_votes(T, x)
    return utils_max_index(v, v.shape[0]) == y


@njit(error_model="numpy")
def tree_vote_into(T, x, acc, weight, add):
    """Ensemble voting step for one member: raw votes -> predicted class; if
    ``add`` and the votes sum is positive, add the normalised vote (times
    ``weight`` when ``weight > 0``) to ``acc`` (MOA DoubleVector arithmetic)."""
    v = tree_votes(T, x)
    n = v.shape[0]
    pred = utils_max_index(v, n)
    if not add:
        return pred, False
    s = 0.0
    for k in range(n):
        s += v[k]
    if s > 0.0:
        sa = 0.0
        for k in range(n):
            sa += v[k] if v[k] > 0.0 else -v[k]
        inv = 1.0 / sa
        if weight > 0.0:
            for k in range(n):
                acc[k] += (v[k] * inv) * weight
        else:
            for k in range(n):
                acc[k] += v[k] * inv
        return pred, True
    return pred, False


@njit(error_model="numpy")
def tree_train_check(T, x, c, w):
    """``tree_train`` followed by ``correctlyClassifies`` when no split attempt is pending."""
    status, node = tree_train(T, x, c, w)
    if status == 0:
        return status, node, tree_correct(T, x, c)
    return status, node, False


# --------------------------------------------------------------------------- splitting
@njit(error_model="numpy")
def tree_suggestions(T, node, num_bins, binary_only, min_frac, merit, kind, value, nb, dists, lens, valid):
    """Best split per monitored attribute of the leaf (attribute order).

    Numeric candidates: Gaussian bins (HT, ARF) or one random cut point drawn
    with the tree's random generator (ARTE, reconstructed observer).
    Returns the number of monitored attributes.
    """
    s = T.slot[node]
    C = T.C
    h = T.hslot[s]
    na = T.natt[h] if h >= 0 else 0
    pre = np.zeros(C)
    for k in range(T.nclen[node]):
        pre[k] = T.ncls[node, k]
    buf = np.zeros((dists.shape[1], C))
    blen = np.zeros(dists.shape[1], dtype=np.int64)
    pts = np.empty(max(num_bins, 1))
    NUM = T.NUM[h if h >= 0 else 0]
    NOM = T.NOM[h if h >= 0 else 0]
    NLEN = T.NLEN[h if h >= 0 else 0]
    for j in range(na):
        valid[j] = False
        a = T.att[h, j]
        if T.kind[h, j] == 0:
            q = T.num_idx[a]
            lo = math.inf
            hi = -math.inf
            for c in range(C):
                if NUM[0, q, c] > 0.0:
                    if NUM[3, q, c] < lo:
                        lo = NUM[3, q, c]
                    if NUM[4, q, c] > hi:
                        hi = NUM[4, q, c]
            npts = 0
            if T.policy == POLICY_ARTE:
                if lo < math.inf and hi > lo:
                    pts[0] = lo + rng_next_double(T.rng) * (hi - lo)
                    npts = 1
            elif lo < math.inf:
                rng = hi - lo
                for i in range(num_bins):
                    sv = rng / (num_bins + 1.0) * (i + 1) + lo
                    if sv > lo and sv < hi:
                        dup = False
                        for t in range(npts):
                            if pts[t] == sv:
                                dup = True
                        if not dup:
                            pts[npts] = sv
                            npts += 1
                pts[:npts] = np.sort(pts[:npts])
            for pi in range(npts):
                sv = pts[pi]
                for k in range(C):
                    buf[0, k] = 0.0
                    buf[1, k] = 0.0
                blen[0] = 0
                blen[1] = 0
                for c in range(C):
                    ws = NUM[0, q, c]
                    if ws == 0.0:
                        continue
                    if sv < NUM[3, q, c]:
                        buf[1, c] += ws
                        blen[1] = c + 1
                    elif sv >= NUM[4, q, c]:
                        buf[0, c] += ws
                        blen[0] = c + 1
                    else:
                        mean = NUM[1, q, c]
                        vsum = NUM[2, q, c]
                        eq = gaussian_pdf(sv, ws, mean, vsum) * ws
                        var = vsum / (ws - 1.0) if ws > 1.0 else 0.0
                        sd = math.sqrt(var)
                        if sd > 0.0:
                            lt = normal_probability((sv - mean) / sd) * ws - eq
                        else:
                            lt = ws - eq if sv < mean else 0.0
                        gt = ws - eq - lt
                        if gt < 0.0:
                            gt = 0.0
                        buf[0, c] += lt + eq
                        buf[1, c] += gt
                        blen[0] = c + 1
                        blen[1] = c + 1
                m = info_gain_merit(pre, buf, 2, min_frac, C)
                if (not valid[j]) or m > merit[j]:
                    valid[j] = True
                    merit[j] = m
                    kind[j] = NUMERIC_BINARY
                    value[j] = sv
                    nb[j] = 2
                    for b in range(2):
                        lens[j, b] = blen[b]
                        for k in range(C):
                            dists[j, b, k] = buf[b, k]
        else:
            qn = T.nom_idx[a]
            off = T.nom_off[a]
            max_vals = 0
            for c in range(C):
                if NLEN[qn, c] > max_vals:
                    max_vals = NLEN[qn, c]
            if not binary_only:
                for b in range(max_vals):
                    blen[b] = 0
                    for k in range(C):
                        buf[b, k] = 0.0
                for c in range(C):
                    for v in range(NLEN[qn, c]):
                        buf[v, c] += NOM[off + v, c]
                        blen[v] = c + 1
                m = info_gain_merit(pre, buf, max_vals, min_frac, C)
                valid[j] = True
                merit[j] = m
                kind[j] = NOMINAL_MULTIWAY
                value[j] = -1.0
                nb[j] = max_vals
                for b in range(max_vals):
                    lens[j, b] = blen[b]
                    for k in range(C):
                        dists[j, b, k] = buf[b, k]
            for val in range(max_vals):
                for k in range(C):
                    buf[0, k] = 0.0
                    buf[1, k] = 0.0
                blen[0] = 0
                blen[1] = 0
                for c in range(C):
                    for v in range(NLEN[qn, c]):
                        if v == val:
                            buf[0, c] += NOM[off + v, c]
                            blen[0] = c + 1
                        else:
                            buf[1, c] += NOM[off + v, c]
                            blen[1] = c + 1
                m = info_gain_merit(pre, buf, 2, min_frac, C)
                if (not valid[j]) or m > merit[j]:
                    valid[j] = True
                    merit[j] = m
                    kind[j] = NOMINAL_BINARY
                    value[j] = val
                    nb[j] = 2
                    for b in range(2):
                        lens[j, b] = blen[b]
                        for k in range(C):
                            dists[j, b, k] = buf[b, k]
    return na


@njit(error_model="numpy")
def tree_do_split(T, node, kind, att, value, dists, lens):
    """Replace active leaf ``node`` by a split node (same id) with new leaves."""
    T.nbc_node = -1
    nbr = dists.shape[0]
    _free_slot(T, T.slot[node])
    T.ntype[node] = SPLIT
    T.satt[node] = att
    T.skind[node] = kind
    T.sval[node] = value
    cap = nbr
    if kind == NOMINAL_MULTIWAY and T.n_values[att] > cap:
        cap = T.n_values[att]
    start = _alloc_children(T, cap)
    T.ch_start[node] = start
    T.ch_n[node] = nbr
    for i in range(nbr):
        if kind == NOMINAL_MULTIWAY and T.policy == POLICY_ARTE:
            if rng_next_double(T.rng) > 0.5:
                continue
        child = new_leaf(T, dists[i], lens[i])
        T.CH[start + i] = child
    T.n_active -= 1
    T.n_decision += 1
    T.n_active += nbr


@njit(error_model="numpy")
def tree_deactivate(T, node):
    T.nbc_node = -1
    _free_slot(T, T.slot[node])
    T.ntype[node] = INACTIVE
    T.n_active -= 1
    T.n_inactive += 1


@njit(error_model="numpy")
def tree_activate(T, node):
    T.nbc_node = -1
    T.ntype[node] = ACTIVE
    s = _alloc_slot(T)
    T.slot[node] = s
    _init_slot(T, s, node)
    T.n_active += 1
    T.n_inactive -= 1


@njit(error_model="numpy")
def tree_learning_nodes(T):
    """``findLearningNodes``: leaves in depth-first (pre-order) order."""
    out = np.empty(T.n_nodes, dtype=np.int64)
    n = 0
    if T.root < 0:
        return out[:0]
    stack = np.empty(T.n_nodes + 1, dtype=np.int64)
    sp = 0
    stack[sp] = T.root
    sp += 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if T.ntype[node] != SPLIT:
            out[n] = node
            n += 1
        else:
            for i in range(T.ch_n[node] - 1, -1, -1):
                ch = T.CH[T.ch_start[node] + i]
                if ch >= 0:
                    stack[sp] = ch
                    sp += 1
    return out[:n]


@njit(error_model="numpy")
def tree_promise(T, node):
    clen = T.nclen[node]
    total = dv_sum(T.ncls[node], clen)
    if total > 0.0:
        return total - T.ncls[node, dv_max_index(T.ncls[node], clen)]
    return 0.0


@njit(error_model="numpy")
def tree_leaf_bytes(T, node):
    """Same approximation of MOA SizeOf as the object implementation."""
    if T.ntype[node] != ACTIVE:
        return 64 + 8 * T.C
    s = T.slot[node]
    size = 160 + 8 * T.C
    h = T.hslot[s]
    if h >= 0:
        for j in range(T.natt[h]):
            size += 400 + 56 * T.C if T.kind[h, j] == 0 else 200 + 32 * T.C
    return size


@njit(error_model="numpy")
def tree_byte_size(T):
    total = 200
    for node in range(T.n_nodes):
        if T.ntype[node] == SPLIT:
            total += 96 + 8 * T.C + 8 * T.ch_n[node]
    for node in tree_learning_nodes(T):
        total += tree_leaf_bytes(T, node)
    return total


@njit(error_model="numpy")
def tree_depth_of(T, target):
    """Depth of ``target`` (root = 0), -1 if unreachable."""
    if T.root < 0:
        return -1
    stack = np.empty(T.n_nodes + 1, dtype=np.int64)
    depth = np.empty(T.n_nodes + 1, dtype=np.int64)
    sp = 0
    stack[0] = T.root
    depth[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        dd = depth[sp]
        if node == target:
            return dd
        if T.ntype[node] == SPLIT:
            for i in range(T.ch_n[node]):
                ch = T.CH[T.ch_start[node] + i]
                if ch >= 0:
                    stack[sp] = ch
                    depth[sp] = dd + 1
                    sp += 1
    return -1


# --------------------------------------------------------------------------- Python wrapper
def _java_double_key(m):
    if m != m:
        return (1, 0.0, 0)
    return (0, m, 1 if math.copysign(1.0, m) > 0 else 0)


class FastTree:
    """HT (``detector=False``) or HLAST/HLAST_D (``detector=True``) with leaf
    attribute policy ``all`` (HT), ``arf`` or ``arte``.

    Same constructor defaults and behaviour as the object implementation; see
    :func:`make_fast_tree` for the configurations used by each ensemble.
    """

    def __init__(self, grace_period=200, split_confidence=1e-7, tie_threshold=0.05, binary_splits=False,
                 no_pre_prune=False, leaf_prediction="NBA", nb_threshold=0, num_bins=10, min_branch_frac=0.01,
                 memory_estimate_period=1_000_000, max_byte_size=33_554_432, random_seed=1,
                 detector=False, distribution=False, hddm_drift=0.001, hddm_warning=0.005, hddm_two_sided=True,
                 policy="all", subspace_size=-1, drop_last=False, name=None):
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_splits = binary_splits
        self.no_pre_prune = no_pre_prune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.num_bins = num_bins
        self.min_branch_frac = min_branch_frac
        self.memory_estimate_period = memory_estimate_period
        self.max_byte_size = max_byte_size
        self.random_seed = random_seed
        self.detector = detector
        self.distribution = distribution
        self.hddm_drift = hddm_drift
        self.hddm_warning = hddm_warning
        self.hddm_two_sided = hddm_two_sided
        self.policy = policy
        self._subspace_size = subspace_size
        self.drop_last = drop_last
        self.name = name or ("HT" if not detector else ("HLAST_D" if distribution else "HLAST"))
        if detector and leaf_prediction != "NBA":
            raise ValueError("HLAST leaf detectors are only fed by NBAdaptive leaves")
        self.schema = None
        self.T = None

    _PARAMS = ("grace_period", "split_confidence", "tie_threshold", "binary_splits", "no_pre_prune",
               "leaf_prediction", "nb_threshold", "num_bins", "min_branch_frac", "memory_estimate_period",
               "max_byte_size", "random_seed", "detector", "distribution", "hddm_drift", "hddm_warning",
               "hddm_two_sided", "policy", "drop_last", "name")

    def params(self) -> dict:
        p = {k: getattr(self, k) for k in self._PARAMS}
        p["subspace_size"] = self._subspace_size
        return p

    def clone(self):
        t = self.__class__(**self.params())
        if self.schema is not None:
            t.set_schema(self.schema)
        return t

    @property
    def subspace_size(self):
        return self._subspace_size

    @subspace_size.setter
    def subspace_size(self, v):
        self._subspace_size = int(v)
        if self.T is not None:
            self.T.subspace = int(v)

    def set_schema(self, schema) -> None:
        self.schema = schema
        pol = {"all": POLICY_ALL, "arf": POLICY_ARF, "arte": POLICY_ARTE}[self.policy]
        leaf = {"MC": LEAF_MC, "NB": LEAF_NB, "NBA": LEAF_NBA}[self.leaf_prediction]
        self.T = TreeState(int(schema.n_classes), schema.is_nominal.astype(np.int8), schema.n_values.astype(np.int64),
                           leaf, float(self.nb_threshold), bool(self.detector), bool(self.distribution),
                           float(self.hddm_drift), float(self.hddm_warning), bool(self.hddm_two_sided), pol,
                           bool(self.drop_last), int(self._subspace_size), float(self.grace_period),
                           int(self.random_seed))
        self._C = int(schema.n_classes)
        nom = schema.n_values[schema.is_nominal]
        self._max_branches = max(2, int(nom.max()) if nom.size else 0)
        self.reset()

    def reset(self) -> None:
        if self.T is not None:
            self.T.clear()
            rng_set_seed(self.T.rng, self.random_seed)
        self.training_weight_seen = 0.0
        self.active_leaf_byte_size_estimate = 0.0
        self.inactive_leaf_byte_size_estimate = 0.0
        self.byte_size_estimate_overhead_fraction = 1.0
        self.growth_allowed = True
        self.n_splits_periodic = 0
        self.n_splits_adaptive = 0
        self.n_attempts_periodic = 0
        self.n_attempts_adaptive = 0

    # ---------------------------------------------------------------- API
    def train(self, x, y, weight=1.0):
        if weight <= 0.0:
            return
        self.training_weight_seen += weight
        T = self.T
        if self.growth_allowed:
            status, node = tree_train(T, x, int(y), float(weight))
            if status == 1:
                self._attempt_periodic(node)
            elif status == 2:
                self._attempt_adaptive(node)
        else:  # pragma: no cover - stopMemManagement is never set in the paper
            raise NotImplementedError
        if math.fmod(self.training_weight_seen, self.memory_estimate_period) == 0:
            self._estimate_model_byte_sizes()

    def predict_votes(self, x):
        return tree_votes(self.T, x)

    def correctly_classifies(self, x, y) -> bool:
        return tree_correct(self.T, x, int(y))

    def vote_into(self, x, acc, weight=-1.0, add=True):
        """See :func:`tree_vote_into`; returns (predicted class, voted)."""
        return tree_vote_into(self.T, x, acc, float(weight), add)

    def train_check(self, x, y, weight) -> bool:
        """``trainOnInstance(weighted)`` then ``correctlyClassifies(x)``."""
        y = int(y)
        if weight <= 0.0:
            return bool(tree_correct(self.T, x, y))
        self.training_weight_seen += weight
        status, node, correct = tree_train_check(self.T, x, y, float(weight))
        if status == 1:
            self._attempt_periodic(node)
        elif status == 2:
            self._attempt_adaptive(node)
        if math.fmod(self.training_weight_seen, self.memory_estimate_period) == 0:
            self._estimate_model_byte_sizes()
            status = 3
        if status:
            correct = tree_correct(self.T, x, y)
        return bool(correct)

    @property
    def n_nodes(self) -> int:
        T = self.T
        return T.n_decision + T.n_active + T.n_inactive

    def depth_of(self, node) -> int:
        return int(tree_depth_of(self.T, node))

    # ---------------------------------------------------------------- splitting
    def _suggestions(self, node):
        T = self.T
        C = self._C
        sugg = []
        if not self.no_pre_prune:
            pre = np.zeros(C)
            clen = T.nclen[node]
            pre[:clen] = T.ncls[node, :clen]
            d = pre.reshape(1, C).copy()
            sugg.append((info_gain_merit(pre, d, 1, self.min_branch_frac, C), -1, -1, 0.0, None, None))
        h = int(T.hslot[T.slot[node]])
        k = int(T.natt[h]) if h >= 0 else 0
        if k:
            mb = self._max_branches
            merit = np.empty(k)
            kind = np.zeros(k, dtype=np.int64)
            value = np.zeros(k)
            nb = np.zeros(k, dtype=np.int64)
            dists = np.zeros((k, mb, C))
            lens = np.zeros((k, mb), dtype=np.int64)
            valid = np.zeros(k, dtype=np.bool_)
            tree_suggestions(T, node, self.num_bins, self.binary_splits, self.min_branch_frac, merit, kind, value,
                             nb, dists, lens, valid)
            att = T.att[h, :k].copy()
            for j in range(k):
                if valid[j]:
                    sugg.append((merit[j], int(kind[j]), int(att[j]), value[j], dists[j, : nb[j]], lens[j, : nb[j]]))
        sugg.sort(key=lambda s: _java_double_key(s[0]))
        return sugg

    def _hoeffding_bound(self, node):
        T = self.T
        clen = T.nclen[node]
        n_classes = clen if clen > 2 else 2
        r = math.log(n_classes) / math.log(2.0)
        n = dv_sum(T.ncls[node], clen)
        return math.sqrt(((r * r) * math.log(1.0 / self.split_confidence)) / (2.0 * n))

    def _attempt_periodic(self, node):
        self.n_attempts_periodic += 1
        sugg = self._suggestions(node)
        if len(sugg) < 2:
            should = len(sugg) > 0
        else:
            hb = self._hoeffding_bound(node)
            should = (sugg[-1][0] - sugg[-2][0] > hb) or (hb < self.tie_threshold)
        if should and self._do_split(node, sugg[-1]):
            self.n_splits_periodic += 1

    def adaptive_split_accepts(self, node, suggestions) -> bool:
        """HLAST adaptive acceptance rule (Algorithm 1, line 18): ``G(Xa) > 0``.

        ``suggestions``: list sorted ascending by merit (>= 2 entries) of tuples
        ``(merit, kind, attribute, value, dists, lens)``; ``node`` is the leaf
        id (``self.depth_of(node)`` gives its depth). Override in HLAST-R.
        """
        return suggestions[-1][0] > 0.0

    def _attempt_adaptive(self, node):
        self.n_attempts_adaptive += 1
        sugg = self._suggestions(node)
        if len(sugg) < 2:
            should = len(sugg) > 0
        else:
            should = self.adaptive_split_accepts(node, sugg)
        if should and self._do_split(node, sugg[-1]):
            self.n_splits_adaptive += 1

    def _do_split(self, node, decision) -> bool:
        merit, kind, att, value, dists, lens = decision
        if kind < 0:
            tree_deactivate(self.T, node)
            split = False
        else:
            tree_do_split(self.T, node, kind, att, float(value), np.ascontiguousarray(dists),
                          np.ascontiguousarray(lens))
            split = True
        self._enforce_tracker_limit()
        return split

    # ---------------------------------------------------------------- memory management (MOA)
    def _enforce_tracker_limit(self):
        T = self.T
        est = ((T.n_active * self.active_leaf_byte_size_estimate
                + T.n_inactive * self.inactive_leaf_byte_size_estimate) * self.byte_size_estimate_overhead_fraction)
        if T.n_inactive > 0 or est > self.max_byte_size:
            nodes = list(tree_learning_nodes(T))
            nodes.sort(key=lambda n: tree_promise(T, n))
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
                if T.ntype[nodes[i]] == ACTIVE:
                    tree_deactivate(T, nodes[i])
            for i in range(cutoff, len(nodes)):
                if T.ntype[nodes[i]] == INACTIVE:
                    tree_activate(T, nodes[i])

    def _estimate_model_byte_sizes(self):
        T = self.T
        tot_a = tot_i = 0
        for n in tree_learning_nodes(T):
            if T.ntype[n] == ACTIVE:
                tot_a += tree_leaf_bytes(T, n)
            else:
                tot_i += tree_leaf_bytes(T, n)
        if tot_a > 0:
            self.active_leaf_byte_size_estimate = tot_a / T.n_active
        if tot_i > 0:
            self.inactive_leaf_byte_size_estimate = tot_i / T.n_inactive
        actual = tree_byte_size(T)
        est = T.n_active * self.active_leaf_byte_size_estimate + T.n_inactive * self.inactive_leaf_byte_size_estimate
        self.byte_size_estimate_overhead_fraction = actual / est if est else float("inf")
        if actual > self.max_byte_size:
            self._enforce_tracker_limit()


def make_fast_tree(ensemble: str, tree: str, cls=None, **kw) -> FastTree:
    """Tree configuration used by each ensemble (see :mod:`hast.experiments`).

    ``cls`` lets a subclass of :class:`FastTree` (e.g. a regularized HLAST-R
    overriding ``adaptive_split_accepts``) be used with the same settings.
    """
    det = tree != "HT"
    dist = tree == "HLAST_D"
    base = dict(detector=det, distribution=dist, name=tree)
    if ensemble == "LevBag":
        pass
    elif ensemble == "SRP":
        base.update(grace_period=50, split_confidence=0.01)
    elif ensemble in ("ARF", "ARE"):
        base.update(grace_period=50, split_confidence=0.01, memory_estimate_period=2_000_000, policy="arf",
                    subspace_size=2, drop_last=det and not kw.pop("fix_arf_subspace_bug", False))
    elif ensemble == "ARTE":
        base.update(grace_period=100, split_confidence=0.01, memory_estimate_period=2_000_000, policy="arte",
                    subspace_size=2)
    elif ensemble != "single":
        raise ValueError(ensemble)
    base.update(kw)
    return (cls or FastTree)(**base)
