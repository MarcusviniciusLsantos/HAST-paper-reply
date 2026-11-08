"""``java.util.Random`` and ``MiscUtils.poisson`` as numba functions over a 1-element
int64 state array (same sequence as :mod:`hast.core.jrandom`)."""
from __future__ import annotations

import math

import numpy as np
from numba import njit

MULT = np.int64(0x5DEECE66D)
ADD = np.int64(0xB)
MASK = np.int64((1 << 48) - 1)
DOUBLE_UNIT = 1.0 / (1 << 53)
TWO31 = np.int64(1 << 31)
TWO32 = np.int64(1 << 32)


@njit(cache=True, error_model="numpy")
def rng_new(seed):
    s = np.zeros(1, dtype=np.int64)
    rng_set_seed(s, seed)
    return s


@njit(cache=True, error_model="numpy")
def rng_set_seed(state, seed):
    state[0] = (np.int64(seed) ^ MULT) & MASK


@njit(cache=True, error_model="numpy")
def rng_next(state, bits):
    state[0] = (state[0] * MULT + ADD) & MASK  # int64 wrap keeps the low 48 bits exact
    v = state[0] >> (48 - bits)
    if v >= TWO31:
        v -= TWO32
    return v


@njit(cache=True, error_model="numpy")
def rng_next_int(state, bound):
    r = rng_next(state, 31)
    m = bound - 1
    if (bound & m) == 0:
        return (np.int64(bound) * r) >> 31
    u = r
    while True:
        r = u % bound
        if u - r + m < TWO31:  # Java: (u - r + m) < 0 on int overflow
            return r
        u = rng_next(state, 31)


@njit(cache=True, error_model="numpy")
def rng_next_double(state):
    return ((rng_next(state, 26) << 27) + rng_next(state, 27)) * DOUBLE_UNIT


@njit(cache=True, error_model="numpy")
def rng_poisson(state, lam):
    """``MiscUtils.poisson`` for lambda < 100 (all ensembles here use 6)."""
    product = 1.0
    total = 1.0
    threshold = rng_next_double(state) * math.exp(lam)
    i = 1
    max_i = max(100, 10 * int(math.ceil(lam)))
    while i < max_i and total <= threshold:
        product *= lam / i
        total += product
        i += 1
    return i - 1
