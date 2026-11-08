"""Compare a Python tree against the Java reference output, instance by instance."""
import sys, time
import numpy as np
from hast.core.arff import load_arff
from hast.trees.hoeffding import HoeffdingTree, HLAST, Schema
from hast.eval.javaref import read_reference, compare_votes

kind, arff, ref, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
s = load_arff(arff)
tree = {"HT": lambda: HoeffdingTree(), "HLAST": lambda: HLAST(), "HLAST_D": lambda: HLAST(distribution=True)}[kind]()
tree.set_schema(Schema.from_header(s.header))
votes = []
t = time.perf_counter()
for i in range(min(n, len(s))):
    x = s.X[i]
    votes.append(tree.predict_votes(x))
    tree.train(x, s.y[i])
el = time.perf_counter() - t
r = read_reference(ref)
print(kind, compare_votes(r, votes), f"| nodes py={tree.n_nodes} java={r.measurements['tree size (nodes)']:.0f} | {el:.2f}s")
print("  splits periodic/adaptive:", tree.n_splits_periodic, tree.n_splits_adaptive)
