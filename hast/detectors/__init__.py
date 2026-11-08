"""Change detectors ported from MOA (``moa.classifiers.core.driftdetection``)."""
from __future__ import annotations

import math


def _jsqrt(v: float) -> float:
    """``Math.sqrt``: NaN for negative input instead of raising."""
    return math.sqrt(v) if v >= 0.0 else math.nan


def _jlog(v: float) -> float:
    """``Math.log``: -inf for 0, NaN for negative input instead of raising."""
    if v > 0.0:
        return math.log(v)
    return -math.inf if v == 0.0 else math.nan


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


class ChangeDetector:
    """Common interface (``AbstractChangeDetector``)."""

    is_change_detected = False
    is_warning_zone = False

    def input(self, value: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def get_change(self) -> bool:
        return self.is_change_detected

    def copy(self) -> "ChangeDetector":  # fresh detector with same options
        raise NotImplementedError


class HDDM_A(ChangeDetector):
    """``HDDM_A_Test`` (Frias-Blanco et al., 2015), MOA defaults: drift 0.001,
    warning 0.005, two-sided test."""

    __slots__ = ("drift_confidence", "warning_confidence", "two_sided", "n_min", "c_min", "total_n",
                 "total_c", "n_max", "c_max", "c_estimacion", "n_estimacion", "is_change_detected",
                 "is_warning_zone", "estimation", "delay", "_log_drift")

    def __init__(self, drift_confidence: float = 0.001, warning_confidence: float = 0.005,
                 two_sided: bool = True):
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.two_sided = two_sided
        self._log_drift = math.log(1.0 / drift_confidence)
        self.reset()

    def reset(self) -> None:
        self.is_change_detected = False
        self.is_warning_zone = False
        self.estimation = 0.0
        self.delay = 0.0
        self.n_min = 0
        self.c_min = 0.0
        self.total_n = 0
        self.total_c = 0.0
        self.n_max = 0
        self.c_max = 0.0
        self.c_estimacion = 0.0
        self.n_estimacion = 0

    def copy(self) -> "HDDM_A":
        return HDDM_A(self.drift_confidence, self.warning_confidence, self.two_sided)

    @staticmethod
    def _mean_incr(c_min, n_min, total_c, total_n, conf):
        if n_min == total_n:
            return False
        m = (total_n - n_min) / n_min * (1.0 / total_n)
        cota = _jsqrt(m / 2 * math.log(2.0 / conf))
        return total_c / total_n - c_min / n_min >= cota

    def input(self, value: float) -> None:
        self.total_n += 1
        self.total_c += value
        if self.n_min == 0:
            self.n_min = self.total_n
            self.c_min = self.total_c
        if self.n_max == 0:
            self.n_max = self.total_n
            self.c_max = self.total_c
        cota = _jsqrt(1.0 / (2 * self.n_min) * self._log_drift)
        cota1 = _jsqrt(1.0 / (2 * self.total_n) * self._log_drift)
        if self.c_min / self.n_min + cota >= self.total_c / self.total_n + cota1:
            self.c_min = self.total_c
            self.n_min = self.total_n
        cota = _jsqrt(1.0 / (2 * self.n_max) * self._log_drift)
        if self.c_max / self.n_max - cota <= self.total_c / self.total_n - cota1:
            self.c_max = self.total_c
            self.n_max = self.total_n
        if self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n, self.drift_confidence):
            self.n_estimacion = self.total_n - self.n_min
            self.c_estimacion = self.total_c - self.c_min
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0.0
            self.is_change_detected = True
            self.is_warning_zone = False
        elif self._mean_incr(self.c_min, self.n_min, self.total_c, self.total_n, self.warning_confidence):
            self.is_change_detected = False
            self.is_warning_zone = True
        else:
            self.is_change_detected = False
            self.is_warning_zone = False
        if self.two_sided and self._mean_decr():
            self.n_estimacion = self.total_n - self.n_max
            self.c_estimacion = self.total_c - self.c_max
            self.n_min = self.n_max = self.total_n = 0
            self.c_min = self.c_max = self.total_c = 0.0
        # updateEstimations
        if self.total_n >= self.n_estimacion:
            self.c_estimacion = 0.0
            self.n_estimacion = 0
            self.estimation = self.total_c / self.total_n if self.total_n else float("nan")
            self.delay = self.total_n
        else:
            self.estimation = self.c_estimacion / self.n_estimacion
            self.delay = self.n_estimacion

    def _mean_decr(self):
        if self.n_max == self.total_n:
            return False
        m = (self.total_n - self.n_max) / self.n_max * (1.0 / self.total_n)
        cota = _jsqrt(m / 2 * math.log(2.0 / self.drift_confidence))
        return self.c_max / self.n_max - self.total_c / self.total_n >= cota


_MAXBUCKETS = 5


class _Row:
    __slots__ = ("total", "variance", "size")

    def __init__(self):
        self.total = [0.0] * (_MAXBUCKETS + 1)
        self.variance = [0.0] * (_MAXBUCKETS + 1)
        self.size = 0

    def insert(self, value, variance):
        k = self.size
        self.size += 1
        self.total[k] = value
        self.variance[k] = variance

    def compress(self, n_deleted):
        t, v = self.total, self.variance
        for k in range(n_deleted, _MAXBUCKETS + 1):
            t[k - n_deleted] = t[k]
            v[k - n_deleted] = v[k]
        for k in range(1, n_deleted + 1):
            t[_MAXBUCKETS - k + 1] = 0.0
            v[_MAXBUCKETS - k + 1] = 0.0
        self.size -= n_deleted


class ADWIN:
    """``moa.classifiers.core.driftdetection.ADWIN`` (Bifet & Gavaldà, 2007).

    Rows are kept in a Python list: index 0 is the list head (row of size-1
    buckets), the last index is the tail.
    """

    __slots__ = ("delta", "time", "clock", "width_t", "last_bucket_row", "TOTAL", "VARIANCE", "WIDTH",
                 "bucket_number", "rows", "bucket_deleted")

    MIN_WINDOW_LONGITUDE = 10
    MIN_WIN_LENGTH = 5

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.time = 0
        self.clock = 32
        self.width_t = 0.0
        self.bucket_deleted = False
        self.rows = [_Row()]
        self.last_bucket_row = 0
        self.TOTAL = 0.0
        self.VARIANCE = 0.0
        self.WIDTH = 0
        self.bucket_number = 0

    def get_estimation(self) -> float:
        return self.TOTAL / self.WIDTH if self.WIDTH else float("nan")

    def _insert_element(self, value: float) -> None:
        self.WIDTH += 1
        self.rows[0].insert(value, 0.0)
        self.bucket_number += 1
        inc = 0.0
        if self.WIDTH > 1:
            w1 = self.WIDTH - 1
            inc = w1 * (value - self.TOTAL / w1) * (value - self.TOTAL / w1) / self.WIDTH
        self.VARIANCE += inc
        self.TOTAL += value
        self._compress_buckets()

    def _delete_element(self) -> int:
        node = self.rows[-1]
        n1 = 1 << self.last_bucket_row
        self.WIDTH -= n1
        self.TOTAL -= node.total[0]
        u1 = node.total[0] / n1
        d = u1 - self.TOTAL / self.WIDTH
        inc = node.variance[0] + _i32(n1 * self.WIDTH) * d * d / (n1 + self.WIDTH)
        self.VARIANCE -= inc
        node.compress(1)
        self.bucket_number -= 1
        if node.size == 0:
            self.rows.pop()
            self.last_bucket_row -= 1
        return n1

    def _compress_buckets(self) -> None:
        rows = self.rows
        i = 0
        while i < len(rows):
            cursor = rows[i]
            if cursor.size == _MAXBUCKETS + 1:
                if i + 1 == len(rows):
                    rows.append(_Row())
                    self.last_bucket_row += 1
                nxt = rows[i + 1]
                n1 = 1 << i
                u1 = cursor.total[0] / n1
                u2 = cursor.total[1] / n1
                inc = _i32(n1 * n1) * (u1 - u2) * (u1 - u2) / (n1 + n1)
                nxt.insert(cursor.total[0] + cursor.total[1], cursor.variance[0] + cursor.variance[1] + inc)
                self.bucket_number += 1
                cursor.compress(2)
                if nxt.size <= _MAXBUCKETS:
                    break
            else:
                break
            i += 1

    def _cut(self, n0, n1, u0, u1, v0, v1, absvalue, delta) -> bool:
        n = self.WIDTH
        dd = _jlog(2 * _jlog(n) / delta)
        v = self.VARIANCE / self.WIDTH
        m = 1.0 / (n0 - self.MIN_WIN_LENGTH + 1) + 1.0 / (n1 - self.MIN_WIN_LENGTH + 1)
        epsilon = _jsqrt(2 * m * v * dd) + 2.0 / 3 * dd * m
        return abs(absvalue) > epsilon

    def set_input(self, value: float, delta: float | None = None) -> bool:
        if delta is None:
            delta = self.delta
        change = False
        self.time += 1
        self._insert_element(value)
        self.bucket_deleted = False
        if self.time % self.clock == 0 and self.WIDTH > self.MIN_WINDOW_LONGITUDE:
            reduce_width = True
            while reduce_width:
                reduce_width = False
                exit_ = False
                n0 = 0
                n1 = self.WIDTH
                u0 = 0.0
                u1 = self.TOTAL
                v0 = 0.0
                v1 = self.VARIANCE
                i = self.last_bucket_row
                while True:
                    cursor = self.rows[i]
                    for k in range(cursor.size):
                        n2 = float(1 << i)
                        u2 = cursor.total[k]
                        if n0 > 0:
                            d = u0 / n0 - u2 / n2
                            v0 += cursor.variance[k] + n0 * n2 * d * d / (n0 + n2)
                        if n1 > 0:
                            d = u1 / n1 - u2 / n2
                            v1 -= cursor.variance[k] + n1 * n2 * d * d / (n1 + n2)
                        n0 += 1 << i
                        n1 -= 1 << i
                        u0 += cursor.total[k]
                        u1 -= cursor.total[k]
                        if i == 0 and k == cursor.size - 1:
                            exit_ = True
                            break
                        absvalue = u0 / n0 - u1 / n1
                        if (n1 > self.MIN_WIN_LENGTH + 1 and n0 > self.MIN_WIN_LENGTH + 1
                                and self._cut(n0, n1, u0, u1, v0, v1, absvalue, delta)):
                            self.bucket_deleted = True
                            reduce_width = True
                            change = True
                            if self.WIDTH > 0:
                                n0 -= self._delete_element()
                                exit_ = True
                                break
                    i -= 1
                    if exit_ or i < 0:
                        break
        self.width_t += self.WIDTH
        return change


class ADWINChangeDetector(ChangeDetector):
    """``ADWINChangeDetector``: flags a change only when the error estimate rises."""

    __slots__ = ("delta", "adwin", "is_change_detected", "is_warning_zone", "estimation")

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.reset()

    def reset(self) -> None:
        self.adwin = ADWIN(self.delta)
        self.is_change_detected = False
        self.is_warning_zone = False
        self.estimation = 0.0

    def copy(self) -> "ADWINChangeDetector":
        return ADWINChangeDetector(self.delta)

    def input(self, value: float) -> None:
        self.is_change_detected = False
        err_estim = self.adwin.get_estimation()
        if self.adwin.set_input(value):
            if self.adwin.get_estimation() > err_estim:
                self.is_change_detected = True
        self.estimation = self.adwin.get_estimation()
