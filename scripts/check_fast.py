"""Bit-exact comparison: fast (numba) tree vs object tree, single trees.

usage: check_fast.py ARFF N
"""
import sys, time
import numpy as np
from hast.core.arff import load_arff
from hast.trees.hoeffding import Schema
from hast.experiments import make_tree
from hast.trees.fast import make_fast_tree

arff, n = sys.argv[1], int(sys.argv[2])
s = load_arff(arff)
sch = Schema.from_header(s.header)
n = min(n, len(s))
for ens in ["single", "ARF", "ARTE"]:
    for tree in ["HT", "HLAST", "HLAST_D"]:
        ref = make_tree("LevBag" if ens == "single" else ens, tree)
        fast = make_fast_tree(ens, tree)
        if ens != "single":
            ref.subspace_size = fast.subspace_size = max(2, int(round(0.6 * sch.n_attributes)))
        ref.set_schema(sch); fast.set_schema(sch)
        mism = 0; first = -1; tr = tf = 0.0
        for i in range(n):
            x = s.X[i]; y = int(s.y[i])
            w = float(1 + (i % 3))  # exercise weights
            t = time.perf_counter(); a = ref.predict_votes(x); ref.train(x, y, w); tr += time.perf_counter() - t
            t = time.perf_counter(); b = fast.predict_votes(x); fast.train(x, y, w); tf += time.perf_counter() - t
            if not np.array_equal(a, b, equal_nan=True):
                mism += 1
                if first < 0: first = i
        same_size = ref.n_nodes == fast.n_nodes
        print(f"{ens:6s} {tree:7s} {s.name:14s} vote_mismatch={mism} (first {first}) nodes ref={ref.n_nodes} fast={fast.n_nodes} "
              f"splits {ref.n_splits_periodic}/{ref.n_splits_adaptive} vs {fast.n_splits_periodic}/{fast.n_splits_adaptive} | "
              f"ref {1e6*tr/n:.1f}us fast {1e6*tf/n:.1f}us per inst")
