"""Check that every dataset of the study loads (building missing .npz caches serially).

Run before submitting array jobs, so tasks never parse the same ARFF at once.
usage: python scripts/check_data.py [group ...]   (default: real synthetic imbalanced)
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hast import datasets  # noqa: E402
from hast.core.arff import load_arff  # noqa: E402


def main():
    groups = sys.argv[1:] or ["real", "synthetic", "imbalanced"]
    g = datasets.groups()
    bad = 0
    for grp in groups:
        names = g.get(grp, [])
        if not names:
            print(f"AVISO: grupo {grp} vazio (faltam os dados em data/?)")
        for n in names:
            t = time.time()
            try:
                s = load_arff(datasets.path_of(n))
                print(f"ok  {grp:10s} {n:32s} n={len(s.y):8d} d={s.X.shape[1]:4d} ({time.time() - t:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"ERRO {grp:10s} {n:32s} {e}", flush=True)
    if bad:
        sys.exit(f"{bad} base(s) com problema")


if __name__ == "__main__":
    main()
