"""Equivalence tests.

* fast (numba) trees == object (reference) trees, bit for bit;
* Python ensembles == authors' Java code on MOA 2024.07 (predictions), when the
  Java reference toolchain (java_ref/) is available.

These tests compile the numba tree once per session (~1-2 min).
"""
import os
import shutil

import numpy as np
import pytest

from hast.core.arff import load_arff
from hast.experiments import java_cli, make_model, make_tree
from hast.trees.fast import make_fast_tree
from hast.trees.hoeffding import Schema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELEC = os.path.join(ROOT, "data", "real", "elecNormNew.arff")
HAVE_DATA = os.path.exists(ELEC)
HAVE_JAVA = os.path.exists(os.path.join(ROOT, "java_ref", "build", "RefRunner.class")) and \
    os.path.exists(os.path.join(ROOT, "third_party", "jdk", "bin", "java"))

pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="data/real/elecNormNew.arff not available")


@pytest.fixture(scope="module")
def elec():
    s = load_arff(ELEC)
    return s, Schema.from_header(s.header)


@pytest.mark.parametrize("ens", ["single", "ARF", "ARTE"])
@pytest.mark.parametrize("tree", ["HT", "HLAST", "HLAST_D"])
def test_fast_tree_bit_exact(elec, ens, tree):
    s, sch = elec
    ref = make_tree("LevBag" if ens == "single" else ens, tree)
    fast = make_fast_tree(ens, tree)
    if ens != "single":
        ref.subspace_size = fast.subspace_size = 5
    ref.set_schema(sch)
    fast.set_schema(sch)
    for i in range(6000):
        x, y = s.X[i], int(s.y[i])
        assert np.array_equal(ref.predict_votes(x), fast.predict_votes(x), equal_nan=True), i
        w = float(1 + i % 3)
        ref.train(x, y, w)
        fast.train(x, y, w)
    assert ref.n_nodes == fast.n_nodes
    assert (ref.n_splits_periodic, ref.n_splits_adaptive) == (fast.n_splits_periodic, fast.n_splits_adaptive)


@pytest.mark.skipif(not HAVE_JAVA, reason="Java reference toolchain not built (see docs/README)")
@pytest.mark.parametrize("ens", ["LevBag", "ARF", "SRP", "ARTE"])
@pytest.mark.parametrize("tree", ["HT", "HLAST", "HLAST_D"])
def test_ensemble_matches_java(elec, ens, tree):
    from hast.eval.javaref import compare_votes, run_java_reference
    s, sch = elec
    n, size = 2500, 10
    ref = run_java_reference(java_cli(ens, tree, size), ELEC, n, f"test_{ens}_{tree}_{n}_{size}")
    m = make_model(ens, tree, ensemble_size=size)
    m.set_schema(sch)
    votes = []
    for i in range(n):
        x = s.X[i]
        votes.append(m.predict_votes(x))
        m.train(x, int(s.y[i]))
    c = compare_votes(ref, votes)
    assert c.pred_mismatches == 0, str(c)
    assert c.max_rel_err < 1e-12, str(c)  # only last-bit differences (JVM exp/log intrinsics)
