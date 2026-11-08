"""Numerical helpers ported from ``moa.core.Statistics`` and ``moa.core.Utils``.

Same formulas and operation order as MOA, so the results match Java to the bit
in practically every case.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

LOG2 = math.log(2.0)  # moa.core.Utils.log2 divides by Math.log(2)
SQRTH = 7.07106781186547524401e-1
MAXLOG = 7.09782712893383996732e2
NORMAL_CONSTANT = math.sqrt(2 * math.pi)

_ERF_T = np.array([9.60497373987051638749e0, 9.00260197203842689217e1, 2.23200534594684319226e3,
                   7.00332514112805075473e3, 5.55923013010394962768e4])
_ERF_U = np.array([3.35617141647503099647e1, 5.21357949780152679795e2, 4.59432382970980127987e3,
                   2.26290000613890934246e4, 4.92673942608635921086e4])
_ERFC_P = np.array([2.46196981473530512524e-10, 5.64189564831068821977e-1, 7.46321056442269912687e0,
                    4.86371970985681366614e1, 1.96520832956077098242e2, 5.26445194995477358631e2,
                    9.34528527171957607540e2, 1.02755188689515710272e3, 5.57535335369399327526e2])
_ERFC_Q = np.array([1.32281951154744992508e1, 8.67072140885989742329e1, 3.54937778887819891062e2,
                    9.75708501743205489753e2, 1.82390916687909736289e3, 2.24633760818710981792e3,
                    1.65666309194161350182e3, 5.57535340817727675546e2])
_ERFC_R = np.array([5.64189583547755073984e-1, 1.27536670759978104416e0, 5.01905042251180477414e0,
                    6.16021097993053585195e0, 7.40974269950448939160e0, 2.97886665372100240670e0])
_ERFC_S = np.array([2.26052863220117276590e0, 9.39603524938001434673e0, 1.20489539808096656605e1,
                    1.70814450747565897222e1, 9.60896809063285878198e0, 3.36907645100081516050e0])


@njit(cache=True, error_model="numpy")
def polevl(x, coef, n):
    ans = coef[0]
    for i in range(1, n + 1):
        ans = ans * x + coef[i]
    return ans


@njit(cache=True, error_model="numpy")
def p1evl(x, coef, n):
    ans = x + coef[0]
    for i in range(1, n):
        ans = ans * x + coef[i]
    return ans


# erf/erfc are mutually recursive in Cephes/MOA; numba does not handle that
# reliably, so the two regimes are split into non-recursive cores.
@njit(cache=True, error_model="numpy")
def _erf_poly(x):
    z = x * x
    return x * polevl(z, _ERF_T, 4) / p1evl(z, _ERF_U, 5)


@njit(cache=True, error_model="numpy")
def _erfc_tail(a):
    """``errorFunctionComplemented`` for ``|a| >= 1``."""
    x = -a if a < 0.0 else a
    z = -a * a
    if z < -MAXLOG:
        return 2.0 if a < 0 else 0.0
    z = math.exp(z)
    if x < 8.0:
        p = polevl(x, _ERFC_P, 8)
        q = p1evl(x, _ERFC_Q, 8)
    else:
        p = polevl(x, _ERFC_R, 5)
        q = p1evl(x, _ERFC_S, 6)
    y = (z * p) / q
    if a < 0:
        y = 2.0 - y
    if y == 0.0:
        return 2.0 if a < 0 else 0.0
    return y


@njit(cache=True, error_model="numpy")
def erf(x):
    if abs(x) > 1.0:
        return 1.0 - _erfc_tail(x)
    return _erf_poly(x)


@njit(cache=True, error_model="numpy")
def erfc(a):
    x = -a if a < 0.0 else a
    if x < 1.0:
        return 1.0 - _erf_poly(a)
    return _erfc_tail(a)


@njit(cache=True, error_model="numpy")
def normal_probability(a):
    """``Statistics.normalProbability``: standard normal CDF."""
    x = a * SQRTH
    z = abs(x)
    if z < SQRTH:
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)
        if x > 0:
            y = 1.0 - y
    return y


@njit(cache=True, error_model="numpy")
def log2(a):
    return math.log(a) / LOG2


@njit(cache=True, error_model="numpy")
def entropy(dist, n):
    """``InfoGainSplitCriterion.computeEntropy(double[])`` over ``dist[:n]``."""
    ent = 0.0
    s = 0.0
    for i in range(n):
        d = dist[i]
        if d > 0.0:
            ent -= d * log2(d)
            s += d
    return (ent + s * log2(s)) / s if s > 0.0 else 0.0
