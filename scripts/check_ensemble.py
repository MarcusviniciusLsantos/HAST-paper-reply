"""Compare a Python ensemble with the Java reference, instance by instance.

usage: check_ensemble.py ENSEMBLE TREE ARFF N [SIZE]
"""
import os, sys, time
from hast.core.arff import load_arff
from hast.trees.hoeffding import Schema
from hast.experiments import make_model, java_cli
from hast.eval.javaref import run_java_reference, compare_votes

ens, tree, arff, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
size = int(sys.argv[5]) if len(sys.argv) > 5 else 10
s = load_arff(arff)
tag = f"{ens}_{tree}_{s.name}_{n}_{size}"
t = time.perf_counter()
ref = run_java_reference(java_cli(ens, tree, size), arff, n, tag)
tj = time.perf_counter() - t
m = make_model(ens, tree, ensemble_size=size, impl=os.environ.get("HAST_IMPL", "fast"))
m.set_schema(Schema.from_header(s.header))
votes = []
t = time.perf_counter()
for i in range(min(n, len(s))):
    x = s.X[i]
    votes.append(m.predict_votes(x))
    m.train(x, int(s.y[i]))
tp = time.perf_counter() - t
c = compare_votes(ref, votes)
print(f"{ens:6s} {tree:7s} {s.name:12s} {c} | java {tj:.1f}s py {tp:.1f}s | mean size {m.tree_sizes().mean():.1f}")
