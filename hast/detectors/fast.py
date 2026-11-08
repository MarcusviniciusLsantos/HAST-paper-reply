"""Numba version of MOA's ADWIN / ADWINChangeDetector over two state arrays
(bit-identical to :class:`hast.detectors.ADWIN`).

Layout: ``ai`` int64 = [time, WIDTH, last_bucket_row, bucket_number, row sizes...]
and ``af`` float64 = [delta, TOTAL, VARIANCE, width_t, estimation, -, -, -,
bucket totals (row*6+k)..., bucket variances...].
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

MAXBUCKETS = 5
NB = MAXBUCKETS + 1
MAXROWS = 64
_RS = 4  # offset of row sizes in ai
_TOT = 8  # offset of bucket totals in af
_VAR = _TOT + MAXROWS * NB  # offset of bucket variances in af
MIN_WINDOW_LONGITUDE = 10
MIN_WIN_LENGTH = 5
CLOCK = 32
TWO31 = np.int64(1 << 31)
TWO32 = np.int64(1 << 32)


def adwin_new(delta: float = 0.002):
    ai = np.zeros(_RS + MAXROWS, dtype=np.int64)
    af = np.zeros(_VAR + MAXROWS * NB)
    af[0] = delta
    af[4] = math.nan
    return ai, af


@njit(cache=True, error_model="numpy")
def adwin_reset(ai, af):
    delta = af[0]
    ai[:] = 0
    af[:] = 0.0
    af[0] = delta
    af[4] = np.nan


@njit(cache=True, error_model="numpy")
def _i32(v):
    v = v & (TWO32 - 1)
    return v - TWO32 if v >= TWO31 else v


@njit(cache=True, error_model="numpy")
def adwin_estimation(ai, af):
    return af[1] / ai[1]


@njit(cache=True, error_model="numpy")
def _row_insert(ai, af, r, value, variance):
    k = ai[_RS + r]
    ai[_RS + r] = k + 1
    af[_TOT + r * NB + k] = value
    af[_VAR + r * NB + k] = variance


@njit(cache=True, error_model="numpy")
def _row_compress(ai, af, r, n_del):
    base_t = _TOT + r * NB
    base_v = _VAR + r * NB
    for k in range(n_del, MAXBUCKETS + 1):
        af[base_t + k - n_del] = af[base_t + k]
        af[base_v + k - n_del] = af[base_v + k]
    for k in range(1, n_del + 1):
        af[base_t + MAXBUCKETS - k + 1] = 0.0
        af[base_v + MAXBUCKETS - k + 1] = 0.0
    ai[_RS + r] -= n_del


@njit(cache=True, error_model="numpy")
def _compress_buckets(ai, af):
    i = 0
    while i <= ai[2]:
        if ai[_RS + i] == MAXBUCKETS + 1:
            if i == ai[2]:
                ai[2] += 1  # addToTail
                ai[_RS + ai[2]] = 0
                for k in range(NB):
                    af[_TOT + ai[2] * NB + k] = 0.0
                    af[_VAR + ai[2] * NB + k] = 0.0
            n1 = np.int64(1) << i
            u1 = af[_TOT + i * NB] / n1
            u2 = af[_TOT + i * NB + 1] / n1
            inc = _i32(n1 * n1) * (u1 - u2) * (u1 - u2) / (n1 + n1)
            _row_insert(ai, af, i + 1, af[_TOT + i * NB] + af[_TOT + i * NB + 1],
                        af[_VAR + i * NB] + af[_VAR + i * NB + 1] + inc)
            ai[3] += 1
            _row_compress(ai, af, i, 2)
            if ai[_RS + i + 1] <= MAXBUCKETS:
                break
        else:
            break
        i += 1


@njit(cache=True, error_model="numpy")
def _insert_element(ai, af, value):
    ai[1] += 1
    _row_insert(ai, af, 0, value, 0.0)
    ai[3] += 1
    inc = 0.0
    W = ai[1]
    if W > 1:
        w1 = W - 1
        inc = w1 * (value - af[1] / w1) * (value - af[1] / w1) / W
    af[2] += inc
    af[1] += value
    _compress_buckets(ai, af)


@njit(cache=True, error_model="numpy")
def _delete_element(ai, af):
    r = ai[2]
    n1 = np.int64(1) << r
    ai[1] -= n1
    af[1] -= af[_TOT + r * NB]
    u1 = af[_TOT + r * NB] / n1
    d = u1 - af[1] / ai[1]
    inc = af[_VAR + r * NB] + _i32(n1 * ai[1]) * d * d / (n1 + ai[1])
    af[2] -= inc
    _row_compress(ai, af, r, 1)
    ai[3] -= 1
    if ai[_RS + r] == 0:
        ai[2] -= 1
    return n1


@njit(cache=True, error_model="numpy")
def _jsqrt(v):
    return math.sqrt(v) if v >= 0.0 else np.nan


@njit(cache=True, error_model="numpy")
def _jlog(v):
    if v > 0.0:
        return math.log(v)
    return -np.inf if v == 0.0 else np.nan


@njit(cache=True, error_model="numpy")
def adwin_set_input(ai, af, value):
    """``ADWIN.setInput``: returns True when the window was cut."""
    delta = af[0]
    change = False
    ai[0] += 1
    _insert_element(ai, af, value)
    if ai[0] % CLOCK == 0 and ai[1] > MIN_WINDOW_LONGITUDE:
        reduce_width = True
        while reduce_width:
            reduce_width = False
            exit_ = False
            n0 = 0
            n1 = ai[1]
            u0 = 0.0
            u1 = af[1]
            v0 = 0.0
            v1 = af[2]
            i = ai[2]
            while True:
                size = ai[_RS + i]
                for k in range(size):
                    n2 = float(np.int64(1) << i)
                    u2 = af[_TOT + i * NB + k]
                    if n0 > 0:
                        dd0 = u0 / n0 - u2 / n2
                        v0 += af[_VAR + i * NB + k] + n0 * n2 * dd0 * dd0 / (n0 + n2)
                    if n1 > 0:
                        dd1 = u1 / n1 - u2 / n2
                        v1 -= af[_VAR + i * NB + k] + n1 * n2 * dd1 * dd1 / (n1 + n2)
                    n0 += np.int64(1) << i
                    n1 -= np.int64(1) << i
                    u0 += af[_TOT + i * NB + k]
                    u1 -= af[_TOT + i * NB + k]
                    if i == 0 and k == size - 1:
                        exit_ = True
                        break
                    absvalue = u0 / n0 - u1 / n1
                    if n1 > MIN_WIN_LENGTH + 1 and n0 > MIN_WIN_LENGTH + 1:
                        n = ai[1]
                        ddl = _jlog(2 * _jlog(n) / delta)
                        v = af[2] / ai[1]
                        m = 1.0 / (n0 - MIN_WIN_LENGTH + 1) + 1.0 / (n1 - MIN_WIN_LENGTH + 1)
                        eps = _jsqrt(2 * m * v * ddl) + 2.0 / 3 * ddl * m
                        if abs(absvalue) > eps:
                            reduce_width = True
                            change = True
                            if ai[1] > 0:
                                n0 -= _delete_element(ai, af)
                                exit_ = True
                                break
                i -= 1
                if exit_ or i < 0:
                    break
    af[3] += ai[1]
    return change


@njit(cache=True, error_model="numpy")
def adwin_cd_input(ai, af, value):
    """``ADWINChangeDetector.input``: True when a change (error increase) is flagged."""
    err = af[1] / ai[1]
    changed = False
    if adwin_set_input(ai, af, value):
        if af[1] / ai[1] > err:
            changed = True
    af[4] = af[1] / ai[1]
    return changed
