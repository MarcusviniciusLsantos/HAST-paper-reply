"""Parse the authors' raw result tables (results_{f1,acc}_dmkd.pdf) into a tidy CSV.

Output: results/paper/paper_results.csv with columns
metric, ensemble, dataset, tree, mean, std
"""
from __future__ import annotations

import os
import re
import subprocess

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "original_data_article")
OUT = os.path.join(ROOT, "results", "paper")

ENS = {"LeveragingBag": "LevBag", "AdaptiveRandomForest": "ARF", "StreamingRandomPatches": "SRP",
       "ARTE": "ARTE", "AdaptiveRegularizedEnsemble": "ARE"}
TREES = ["HT", "EFDT", "LAST", "LAST_D", "HLAST", "HLAST_D", "EFLAST", "EFLAST_D"]
NUM = r"(\d+\.\d+)\s*±\s*(\d+\.\d+)"


def parse(pdf: str, metric: str) -> list[dict]:
    txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True, check=True).stdout
    rows, ens = [], None
    for line in txt.splitlines():
        m = re.search(r"Table \d+: .* of (\w+)", line)
        if m:
            ens = ENS[m.group(1)]
            continue
        vals = re.findall(NUM, line)
        if ens and len(vals) == 8:
            name = line.split()[0]
            for tree, (mu, sd) in zip(TREES, vals):
                rows.append(dict(metric=metric, ensemble=ens, dataset=name, tree=tree, mean=float(mu), std=float(sd)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = parse(os.path.join(SRC, "results_f1_dmkd.pdf"), "f1") + parse(os.path.join(SRC, "results_acc_dmkd.pdf"), "accuracy")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "paper_results.csv"), index=False)
    print(df.groupby(["metric", "ensemble"]).dataset.nunique())


if __name__ == "__main__":
    main()
