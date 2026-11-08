"""Build the imbalanced benchmark of the study (not part of the original paper).

* 6 real imbalanced streams from Aguiar, Cano & Krawczyk (2024) -- ARFF files
  from the survey repository (github.com/canoalberto/imbalanced-streams);
* 6 synthetic streams generated with MOA's ``ImbalancedStream`` (controlled
  class ratios), static and with drift. Each concept is made imbalanced inside
  ``ConceptDriftStream`` so drift positions refer to the output stream.

usage: python scripts/prepare_imbalanced.py --source third_party/imbalanced_dl/ex [-n 200000]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from generate_synthetic import generate  # noqa: E402

OUT = os.path.join(ROOT, "data", "imbalanced")

REAL = {  # name -> file in the survey archives (binary.zip / multiclass.zip)
    "creditcard": "creditcard.arff",  # binary, IR ~578
    "GMSC": "GMSC.arff",  # binary, IR ~14, missing values
    "coil2000": "coil2000.arff",  # binary, IR ~16
    "shuttle": "shuttle.arff",  # 7 classes, IR ~4559
    "kr-vs-k": "kr-vs-k.arff",  # 18 classes, IR ~169
    "thyroid": "thyroid.arff",  # 3 classes, IR ~40
}


def ratios(ir: int) -> str:
    """Binary class ratio string for MOA ImbalancedStream (minority = class 1)."""
    p = 1.0 / (ir + 1)
    return f"{1 - p:.6f};{p:.6f}"


def imb(gen: str, ratio: str, seed: int = 1) -> str:
    return f"ImbalancedStream -s ({gen}) -c {ratio} -i {seed}"


def chain(concepts: list[str], n: int, width: int) -> str:
    seg = n // len(concepts)
    s = f"({concepts[-1]})"
    for c in reversed(concepts[:-1]):
        s = f"(ConceptDriftStream -s ({c}) -d {s} -p {seg} -w {width})"
    return s[1:-1]


def synthetic(n: int) -> dict[str, str]:
    agr = lambda f: f"generators.AgrawalGenerator -f {f} -i 1"
    sea = lambda f: f"generators.SEAGenerator -f {f} -i 1"
    out = {}
    for ir in (10, 100):
        out[f"AGR-IR{ir}-static"] = imb(agr(1), ratios(ir))
        out[f"AGR-IR{ir}-abrupt"] = chain([imb(agr(f), ratios(ir), 1 + k) for k, f in enumerate((1, 3, 5))], n, 1)
    out["SEA-IR10-gradual"] = chain([imb(sea(f), ratios(10), 1 + k) for k, f in enumerate((1, 2, 3, 4))], n,
                                    n // 40)
    out["RBF5-IR60-incremental"] = imb("generators.RandomRBFGeneratorDrift -c 5 -a 10 -n 50 -k 50 -s 1e-4 -r 1 -i 1",
                                       "0.60;0.25;0.10;0.04;0.01")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.path.join(ROOT, "third_party", "imbalanced_dl", "ex"))
    ap.add_argument("-n", type=int, default=200_000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tsv = os.path.join(OUT, "streams.tsv")
    have = set()
    if os.path.exists(tsv):
        have = {l.split("\t")[0] for l in open(tsv)}
    for name, fname in REAL.items():
        dst = os.path.join(OUT, fname)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(a.source, fname), dst)
        if name not in have:
            with open(tsv, "a") as fh:
                fh.write(f"{name}\t{fname}\treal (Aguiar, Cano & Krawczyk 2024)\n")
    for name, cli in synthetic(a.n).items():
        print(generate(name, cli, OUT, a.n), flush=True)


if __name__ == "__main__":
    main()
