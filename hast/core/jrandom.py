"""Bit-exact port of ``java.util.Random`` and the MOA helpers built on it.

MOA ensembles draw Poisson weights, random subspaces and seeds from
``java.util.Random``. Reproducing the exact sequence lets the Python port be
checked instance-by-instance against the original Java code.
"""
from __future__ import annotations

import math

_MULT = 0x5DEECE66D
_ADD = 0xB
_MASK = (1 << 48) - 1
_DOUBLE_UNIT = 1.0 / (1 << 53)


def _to_int32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def _to_int64(v: int) -> int:
    v &= 0xFFFFFFFFFFFFFFFF
    return v - (1 << 64) if v & 0x8000000000000000 else v


class JavaRandom:
    """Linear congruential generator identical to ``java.util.Random``."""

    __slots__ = ("_seed", "_next_next_gaussian", "_have_next_next_gaussian")

    def __init__(self, seed: int = 0):
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self._seed = (_to_int64(seed) ^ _MULT) & _MASK
        self._have_next_next_gaussian = False
        self._next_next_gaussian = 0.0

    def _next(self, bits: int) -> int:
        self._seed = (self._seed * _MULT + _ADD) & _MASK
        return _to_int32(self._seed >> (48 - bits))

    def next_int(self, bound: int | None = None) -> int:
        if bound is None:
            return self._next(32)
        if bound <= 0:
            raise ValueError("bound must be positive")
        r = self._next(31)
        m = bound - 1
        if (bound & m) == 0:  # power of two
            return _to_int32((bound * r) >> 31)
        u = r
        while True:
            r = u % bound
            # Java: u - r + m < 0 detects int overflow
            if _to_int32(u - r + m) >= 0:
                return r
            u = self._next(31)

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) * _DOUBLE_UNIT

    def next_boolean(self) -> bool:
        return self._next(1) != 0

    def next_gaussian(self) -> float:
        if self._have_next_next_gaussian:
            self._have_next_next_gaussian = False
            return self._next_next_gaussian
        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._next_next_gaussian = v2 * multiplier
        self._have_next_next_gaussian = True
        return v1 * multiplier


def moa_poisson(lam: float, r: JavaRandom) -> int:
    """``moa.core.MiscUtils.poisson``."""
    if lam < 100.0:
        product = 1.0
        total = 1.0
        threshold = r.next_double() * math.exp(lam)
        i = 1
        max_i = max(100, 10 * int(math.ceil(lam)))
        while i < max_i and total <= threshold:
            product *= lam / i
            total += product
            i += 1
        return i - 1
    x = lam + math.sqrt(lam) * r.next_gaussian()
    if x < 0.0:
        return 0
    return int(math.floor(x))


def java_string_hash(s: str) -> int:
    """``String.hashCode`` (32-bit signed)."""
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return _to_int32(h)


def java_double_to_string_integral(v: float) -> str:
    """``Double.toString`` for integral values below 1e7 (plain notation).

    ARTE seeds each member with ``Double.toString(instancesSeen).hashCode()``;
    streams in this study are shorter than 1e7, where Java switches to
    scientific notation.
    """
    if not (0 <= v < 1e7) or v != int(v):
        raise ValueError(f"unsupported value for Double.toString emulation: {v}")
    return f"{int(v)}.0"
