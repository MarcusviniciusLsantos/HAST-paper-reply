"""Numba kernels for leaf statistics of Hoeffding-based trees.

Each active leaf stores the MOA attribute observers of the attributes it
monitors (``att``, sorted by attribute index, as MOA's ``AutoExpandVector``):

* numeric (``GaussianNumericAttributeClassObserver``): ``NUM[0..4, j, c]`` =
  weightSum, mean, varianceSum, min, max of attribute ``att[j]`` for class ``c``.
  A class "has an estimator" iff its weightSum > 0.
* nominal (``NominalAttributeClassObserver``): ``NOM[j, v, c]`` counts and
  ``NLEN[j, c]`` = length of MOA's auto-expanding ``DoubleVector`` (0 = null).

Operation order follows the Java code so results match MOA to the bit.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from hast.core.mstats import NORMAL_CONSTANT, entropy, normal_probability

KIND_NUMERIC = 0
KIND_NOMINAL = 1

SPLIT_NUMERIC_BINARY = 0
SPLIT_NOMINAL_MULTIWAY = 1
SPLIT_NOMINAL_BINARY = 2

NEG_INF = -math.inf


# --------------------------------------------------------------------------- utils
@njit(cache=True, error_model="numpy")
def utils_max_index(a, n):
    """``moa.core.Utils.maxIndex`` over ``a[:n]`` (returns 0 for empty arrays)."""
    maximum = 0.0
    idx = 0
    for i in range(n):
        if i == 0 or a[i] > maximum:
            idx = i
            maximum = a[i]
    return idx


@njit(cache=True, error_model="numpy")
def dv_max_index(a, n):
    """``DoubleVector.maxIndex`` over ``a[:n]`` (returns -1 for empty arrays)."""
    m = -1
    for i in range(n):
        if m < 0 or a[i] > a[m]:
            m = i
    return m


@njit(cache=True, error_model="numpy")
def dv_sum(a, n):
    s = 0.0
    for i in range(n):
        s += a[i]
    return s


# --------------------------------------------------------------------------- observe
@njit(cache=True, error_model="numpy")
def observe(x, c, w, att, kind, NUM, NOM, NLEN):
    """Update the observers of every monitored attribute with instance ``x``."""
    for j in range(att.shape[0]):
        v = x[att[j]]
        if kind[j] == KIND_NUMERIC:
            if math.isnan(v):
                continue  # Utils.isMissingValue
            ws = NUM[0, j, c]
            if ws == 0.0:  # estimator created now
                NUM[3, j, c] = v
                NUM[4, j, c] = v
            else:
                if v < NUM[3, j, c]:
                    NUM[3, j, c] = v
                if v > NUM[4, j, c]:
                    NUM[4, j, c] = v
            if math.isinf(v):
                continue
            if ws > 0.0:
                ws += w
                NUM[0, j, c] = ws
                last_mean = NUM[1, j, c]
                NUM[1, j, c] = last_mean + w * (v - last_mean) / ws
                NUM[2, j, c] += w * (v - last_mean) * (v - NUM[1, j, c])
            else:
                NUM[1, j, c] = v
                NUM[0, j, c] = w
        else:
            if math.isnan(v):
                continue
            iv = int(v)
            NOM[j, iv, c] += w
            if iv + 1 > NLEN[j, c]:
                NLEN[j, c] = iv + 1


# --------------------------------------------------------------------------- densities
@njit(cache=True, error_model="numpy")
def gaussian_pdf(value, ws, mean, vsum):
    """``GaussianEstimator.probabilityDensity``."""
    if ws > 0.0:
        var = vsum / (ws - 1.0) if ws > 1.0 else 0.0
        sd = math.sqrt(var)
        if sd > 0.0:
            diff = value - mean
            return (1.0 / (NORMAL_CONSTANT * sd)) * math.exp(-(diff * diff / (2.0 * sd * sd)))
        return 1.0 if value == mean else 0.0
    return 0.0


@njit(cache=True, error_model="numpy")
def nominal_prob(NOM, NLEN, j, iv, c):
    n = NLEN[j, c]
    if n == 0:
        return 0.0
    cnt = NOM[j, iv, c] if iv < n else 0.0
    s = 0.0
    for k in range(n):
        s += NOM[j, k, c]
    return (cnt + 1.0) / (s + n)


@njit(cache=True, error_model="numpy")
def naive_bayes(x, cls, clen, att, kind, NUM, NOM, NLEN, out):
    """``NaiveBayes.doNaiveBayesPrediction``; writes ``clen`` votes into ``out``."""
    total = dv_sum(cls, clen)
    for c in range(clen):
        p = cls[c] / total
        for j in range(att.shape[0]):
            v = x[att[j]]
            if math.isnan(v):
                continue
            if kind[j] == KIND_NUMERIC:
                p *= gaussian_pdf(v, NUM[0, j, c], NUM[1, j, c], NUM[2, j, c])
            else:
                p *= nominal_prob(NOM, NLEN, j, int(v), c)
        out[c] = p


@njit(cache=True, error_model="numpy")
def nb_adaptive_learn(x, c, w, cls, clen, att, kind, NUM, NOM, NLEN, scratch, mc_nb, distribution):
    """Pre-training bookkeeping of ``LearningNodeNBAdaptive.learnFromInstance``.

    Updates ``mc_nb = [mcCorrectWeight, nbCorrectWeight]`` and returns the value
    HLAST feeds to its leaf change detector: the error of the currently better
    leaf predictor (HLAST) or the class-distribution entropy (HLAST_D). The
    caller then adds the instance with :func:`observe`.
    """
    mc_right = 1.0
    nb_right = 1.0
    if dv_max_index(cls, clen) == c:
        mc_nb[0] += w
        mc_right = 0.0
    naive_bayes(x, cls, clen, att, kind, NUM, NOM, NLEN, scratch)
    if utils_max_index(scratch, clen) == c:
        mc_nb[1] += w
        nb_right = 0.0
    if distribution:
        return entropy(cls, clen)
    return mc_right if mc_nb[0] > mc_nb[1] else nb_right


# --------------------------------------------------------------------------- split evaluation
@njit(cache=True, error_model="numpy")
def info_gain_merit(pre, dists, nb, min_frac, C):
    """``InfoGainSplitCriterion.getMeritOfSplit`` for ``dists[:nb, :C]``."""
    total = 0.0
    sums = np.zeros(nb)
    for i in range(nb):
        s = 0.0
        for k in range(C):
            s += dists[i, k]
        sums[i] = s
        total += s
    n_greater = 0
    for i in range(nb):
        if sums[i] / total > min_frac:
            n_greater += 1
    if n_greater < 2:
        return NEG_INF
    # computeEntropy(double[][])
    tw = 0.0
    dw = np.zeros(nb)
    for i in range(nb):
        s = 0.0
        for k in range(C):
            s += dists[i, k]
        dw[i] = s
        tw += s
    ent = 0.0
    for i in range(nb):
        ent += dw[i] * entropy(dists[i], C)
    return entropy(pre, C) - ent / tw


@njit(cache=True, error_model="numpy")
def null_split_merit(pre, C, min_frac):
    d = np.empty((1, C))
    for k in range(C):
        d[0, k] = pre[k]
    return info_gain_merit(pre, d, 1, min_frac, C)


@njit(cache=True, error_model="numpy")
def _numeric_post_split(NUM, j, C, split_value, buf, lens):
    """``GaussianNumericAttributeClassObserver.getClassDistsResultingFromBinarySplit``."""
    for k in range(C):
        buf[0, k] = 0.0
        buf[1, k] = 0.0
    lens[0] = 0
    lens[1] = 0
    for c in range(C):
        ws = NUM[0, j, c]
        if ws == 0.0:
            continue
        if split_value < NUM[3, j, c]:
            buf[1, c] += ws
            lens[1] = c + 1
        elif split_value >= NUM[4, j, c]:
            buf[0, c] += ws
            lens[0] = c + 1
        else:
            mean = NUM[1, j, c]
            vsum = NUM[2, j, c]
            eq = gaussian_pdf(split_value, ws, mean, vsum) * ws
            var = vsum / (ws - 1.0) if ws > 1.0 else 0.0
            sd = math.sqrt(var)
            if sd > 0.0:
                lt = normal_probability((split_value - mean) / sd) * ws - eq
            else:
                lt = ws - eq if split_value < mean else 0.0
            gt = ws - eq - lt
            if gt < 0.0:
                gt = 0.0
            buf[0, c] += lt + eq
            buf[1, c] += gt
            lens[0] = c + 1
            lens[1] = c + 1


@njit(cache=True, error_model="numpy")
def best_split_suggestions(pre, C, att, kind, NUM, NOM, NLEN, points, npoints, binary_only, min_frac,
                           out_merit, out_kind, out_value, out_nb, out_dists, out_lens, out_valid):
    """Best split of every monitored attribute (``getBestEvaluatedSplitSuggestion``).

    ``points[j, :npoints[j]]`` are the candidate thresholds of numeric attribute
    ``j`` (Gaussian bins in HT, see :func:`gaussian_split_points_all`; random cut
    points in ARTE).
    """
    buf = np.zeros((out_dists.shape[1], C))
    lens = np.zeros(out_dists.shape[1], dtype=np.int64)
    for j in range(att.shape[0]):
        out_valid[j] = False
        if kind[j] == KIND_NUMERIC:
            for p in range(npoints[j]):
                sv = points[j, p]
                _numeric_post_split(NUM, j, C, sv, buf, lens)
                merit = info_gain_merit(pre, buf, 2, min_frac, C)
                if (not out_valid[j]) or merit > out_merit[j]:
                    out_valid[j] = True
                    out_merit[j] = merit
                    out_kind[j] = SPLIT_NUMERIC_BINARY
                    out_value[j] = sv
                    out_nb[j] = 2
                    for b in range(2):
                        out_lens[j, b] = lens[b]
                        for k in range(C):
                            out_dists[j, b, k] = buf[b, k]
        else:
            max_vals = 0
            for c in range(C):
                if NLEN[j, c] > max_vals:
                    max_vals = NLEN[j, c]
            if not binary_only:
                for b in range(max_vals):
                    lens[b] = 0
                    for k in range(C):
                        buf[b, k] = 0.0
                for c in range(C):
                    for v in range(NLEN[j, c]):
                        buf[v, c] += NOM[j, v, c]
                        lens[v] = c + 1
                merit = info_gain_merit(pre, buf, max_vals, min_frac, C)
                out_valid[j] = True
                out_merit[j] = merit
                out_kind[j] = SPLIT_NOMINAL_MULTIWAY
                out_value[j] = -1.0
                out_nb[j] = max_vals
                for b in range(max_vals):
                    out_lens[j, b] = lens[b]
                    for k in range(C):
                        out_dists[j, b, k] = buf[b, k]
            for val in range(max_vals):
                for k in range(C):
                    buf[0, k] = 0.0
                    buf[1, k] = 0.0
                lens[0] = 0
                lens[1] = 0
                for c in range(C):
                    for v in range(NLEN[j, c]):
                        if v == val:
                            buf[0, c] += NOM[j, v, c]
                            lens[0] = c + 1
                        else:
                            buf[1, c] += NOM[j, v, c]
                            lens[1] = c + 1
                merit = info_gain_merit(pre, buf, 2, min_frac, C)
                if (not out_valid[j]) or merit > out_merit[j]:
                    out_valid[j] = True
                    out_merit[j] = merit
                    out_kind[j] = SPLIT_NOMINAL_BINARY
                    out_value[j] = val
                    out_nb[j] = 2
                    for b in range(2):
                        out_lens[j, b] = lens[b]
                        for k in range(C):
                            out_dists[j, b, k] = buf[b, k]


@njit(cache=True, error_model="numpy")
def gaussian_split_points(NUM, j, C, num_bins):
    """``GaussianNumericAttributeClassObserver.getSplitPointSuggestions`` (sorted, unique)."""
    min_v = math.inf
    max_v = -math.inf
    for c in range(C):
        if NUM[0, j, c] > 0.0:
            if NUM[3, j, c] < min_v:
                min_v = NUM[3, j, c]
            if NUM[4, j, c] > max_v:
                max_v = NUM[4, j, c]
    out = np.empty(num_bins)
    n = 0
    if min_v < math.inf:
        rng = max_v - min_v
        for i in range(num_bins):
            sv = rng / (num_bins + 1.0) * (i + 1) + min_v
            if sv > min_v and sv < max_v:
                out[n] = sv
                n += 1
    res = np.sort(out[:n])
    # TreeSet de-duplication
    m = 0
    for i in range(n):
        if m == 0 or res[i] != res[m - 1]:
            res[m] = res[i]
            m += 1
    return res[:m]


@njit(cache=True, error_model="numpy")
def gaussian_split_points_all(NUM, kind, C, num_bins, points, npoints):
    for j in range(kind.shape[0]):
        npoints[j] = 0
        if kind[j] == KIND_NUMERIC:
            pts = gaussian_split_points(NUM, j, C, num_bins)
            for p in range(pts.shape[0]):
                points[j, p] = pts[p]
            npoints[j] = pts.shape[0]
