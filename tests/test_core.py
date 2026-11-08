"""Unit tests of the low-level ports (random generator, statistics, detectors, metrics, ARFF)."""
import math
import os

import numpy as np
import pytest

from hast.core import fastrandom as F
from hast.core.arff import load_arff
from hast.core.jrandom import JavaRandom, java_double_to_string_integral, java_string_hash, moa_poisson
from hast.core.mstats import erf, erfc, normal_probability
from hast.detectors import ADWIN, HDDM_A, ADWINChangeDetector
from hast.detectors.fast import adwin_cd_input, adwin_new, adwin_set_input
from hast.eval.prequential import moa_metrics, sample_points, summarize

# Values printed by java.util.Random / moa.core.MiscUtils (JDK 21, MOA 2024.07)
JAVA = dict(
    ints42=[0, 3, 8, 4, 0, 5],
    ints42_big=[19720, 792361, 251232, 117505],
    doubles42=[0.9033722646721782, 0.36878291341130565, 0.2757480694417024],
    gauss42=[-0.19094451307087512, 1.4862133923906502, 0.8023071496873626],
    poisson6_seed1=[7, 5, 4, 5, 11, 1, 11, 10, 10, 10, 5, 5],
    neg_seed_ints=[1155869324, -423064701, -1403676116, -232591881, -2055186340],
    hash_1234=1450575236,
)


def test_java_random_matches_java():
    r = JavaRandom(42)
    assert [r.next_int(10) for _ in range(6)] == JAVA["ints42"]
    assert [r.next_int(1000003) for _ in range(4)] == JAVA["ints42_big"]
    assert [r.next_double() for _ in range(3)] == JAVA["doubles42"]
    assert [r.next_gaussian() for _ in range(3)] == JAVA["gauss42"]
    p = JavaRandom(1)
    assert [moa_poisson(6.0, p) for _ in range(12)] == JAVA["poisson6_seed1"]
    n = JavaRandom(-7)
    assert [n.next_int() for _ in range(5)] == JAVA["neg_seed_ints"]
    assert java_string_hash(java_double_to_string_integral(1234.0)) == JAVA["hash_1234"]


def test_numba_random_matches_python():
    for seed in (1, 0, -5, 2**40 + 7):
        a, b = JavaRandom(seed), F.rng_new(seed)
        for _ in range(500):
            assert a.next_int(7) == F.rng_next_int(b, 7)
            assert a.next_int(64) == F.rng_next_int(b, 64)
            assert a.next_double() == F.rng_next_double(b)
            assert moa_poisson(6.0, a) == F.rng_poisson(b, 6.0)


def test_normal_probability():
    for x in (-8.5, -3.0, -0.7, 0.0, 0.3, 1.0, 2.5, 9.0):
        assert normal_probability(x) == pytest.approx(0.5 * math.erfc(-x / math.sqrt(2)), rel=1e-12, abs=1e-15)
    for x in (-2.0, -0.5, 0.2, 1.5, 5.0):
        assert erf(x) == pytest.approx(math.erf(x), rel=1e-13)
        assert erfc(x) == pytest.approx(math.erfc(x), rel=1e-12)


def test_numba_adwin_matches_python():
    rng = np.random.default_rng(0)
    for delta in (0.002, 1e-3, 1e-5):
        xs = np.concatenate([rng.random(20000) < 0.2, rng.random(20000) < 0.45]).astype(float)
        a = ADWIN(delta)
        ai, af = adwin_new(delta)
        for v in xs:
            assert a.set_input(v) == adwin_set_input(ai, af, v)
            assert a.WIDTH == ai[1] and a.TOTAL == af[1] and a.VARIANCE == af[2]
        d = ADWINChangeDetector(delta)
        ai, af = adwin_new(delta)
        n_changes = 0
        for v in xs:
            d.input(v)
            c = adwin_cd_input(ai, af, v)
            assert d.get_change() == c
            n_changes += c
        assert n_changes >= 1  # the error rate increases mid-stream


def test_hddm_a_detects_increase_only_after_change():
    rng = np.random.default_rng(1)
    h = HDDM_A()
    fired = []
    for i in range(4000):
        h.input(float(rng.random() < (0.1 if i < 2000 else 0.5)))
        if h.get_change():
            fired.append(i)
    assert fired and fired[0] >= 2000 and fired[0] < 2200


def test_moa_metrics_nan_rules():
    cm = np.array([[5, 0, 0], [1, 3, 0], [0, 2, 0]])  # class 2 never predicted -> macro precision NaN
    m = moa_metrics(cm)
    assert m["accuracy"] == pytest.approx(100 * 8 / 11)
    assert math.isnan(m["f1"])
    assert summarize([m["f1"], 50.0]) == (25.0, 25.0)
    cm2 = np.array([[5, 1], [2, 4]])
    m2 = moa_metrics(cm2)
    P = (5 / 7 + 4 / 5) / 2
    R = (5 / 6 + 4 / 6) / 2
    assert m2["f1"] == pytest.approx(100 * 2 * P * R / (P + R))


def test_sample_points():
    assert sample_points(4000) == list(range(200, 4001, 200))
    assert len(sample_points(45312)) == 20
    assert sample_points(45312, moa_final_point=True)[-1] == 45312


def test_arff_loader(tmp_path):
    p = tmp_path / "t.arff"
    p.write_text("@relation t\n@attribute a numeric\n@attribute 'b c' {'x y',z}\n@attribute class {A,B}\n"
                 "@data\n1.5,'x y',B,\n-2,z,A,\n")
    s = load_arff(str(p), use_cache=False)
    assert s.X.tolist() == [[1.5, 0.0], [-2.0, 1.0]]
    assert s.y.tolist() == [1, 0]
    assert s.header.n_classes == 2 and s.header.is_nominal.tolist() == [False, True]
