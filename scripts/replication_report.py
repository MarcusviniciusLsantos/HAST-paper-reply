"""Replication report: our Python results vs the authors' raw tables.

Reads results/raw/*.json (our runs) and results/paper/paper_results.csv
(scripts/parse_paper_tables.py) and writes:

* results/replication/comparison.csv  one row per (ensemble, dataset, tree, metric)
* results/replication/summary.json     aggregates used in docs/replication_report.md

Acceptance is by trends and rankings (project proposal, Sec. 4): per ensemble we
report Spearman correlation across datasets, agreement of the best base tree per
dataset, and the win rate of HLAST/HLAST_D over HT (the paper's headline
evidence: HLAST_D beats HT in 75% of real-world cases, HLAST in 63%).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hast import datasets  # noqa: E402
from hast.eval.prequential import summarize  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
PAPER = os.path.join(ROOT, "results", "paper", "paper_results.csv")
OUT = os.path.join(ROOT, "results", "replication")
TREES = ["HT", "HLAST", "HLAST_D"]


def load_ours(raw_dir=RAW, tag="") -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(raw_dir, "*.json")):
        r = json.load(open(f))
        if r.get("extra", {}).get("tag", "") != tag:
            continue
        for metric in ("f1", "accuracy"):
            m, s = summarize(r["metrics"][metric])
            rows.append(dict(metric=metric, ensemble=r["ensemble"], dataset=datasets.paper_name(r["dataset"]),
                             tree=r["tree"], mean=m, std=s, n=r["n_instances"], cpu_s=r["cpu_seconds"],
                             tree_size=float(np.mean(r["tree_size_mean"])),
                             splits_periodic=r["splits"].get("splits_periodic", 0),
                             splits_adaptive=r["splits"].get("splits_adaptive", 0)))
    return pd.DataFrame(rows)


def win_rate(df: pd.DataFrame, tree: str, base: str = "HT") -> tuple[float, int]:
    """Fraction of (ensemble, dataset) pairs where ``tree`` has higher mean than ``base``."""
    p = df.pivot_table(index=["ensemble", "dataset"], columns="tree", values="mean")
    p = p.dropna(subset=[tree, base])
    return (float((p[tree] > p[base]).mean()) if len(p) else float("nan")), int(len(p))


def main():
    os.makedirs(OUT, exist_ok=True)
    ours = load_ours()
    if ours.empty:
        print("no results yet")
        return
    paper = pd.read_csv(PAPER)
    paper = paper[paper.tree.isin(TREES)]
    m = ours.merge(paper, on=["metric", "ensemble", "dataset", "tree"], suffixes=("_ours", "_paper"))
    m["diff"] = m.mean_ours - m.mean_paper
    m.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    real = set(datasets.REAL) | {"LADPU", "Asfault"}
    m["group"] = np.where(m.dataset.isin(real), "real", "synthetic")

    summary = {"n_runs": int(len(ours) // 2), "per_ensemble": {}, "win_rates": {}}
    for (metric, group), g in m.groupby(["metric", "group"]):
        for ens, ge in g.groupby("ensemble"):
            key = f"{metric}|{group}|{ens}"
            d = dict(n=int(len(ge)), mean_abs_diff=float(ge["diff"].abs().mean()),
                     median_diff=float(ge["diff"].median()))
            if ge.dataset.nunique() >= 3:
                d["spearman_datasets"] = float(stats.spearmanr(ge.mean_ours, ge.mean_paper).statistic)
            for tree, gt in ge.groupby("tree"):
                d[f"mean_diff_{tree}"] = float(gt["diff"].mean())
            # best tree per dataset (among HT/HLAST/HLAST_D present in both)
            agree = []
            for ds, gd in ge.groupby("dataset"):
                if gd.tree.nunique() == 3:
                    agree.append(gd.loc[gd.mean_ours.idxmax(), "tree"] == gd.loc[gd.mean_paper.idxmax(), "tree"])
            if agree:
                d["best_tree_agreement"] = float(np.mean(agree))
                d["n_datasets_complete"] = len(agree)
            summary["per_ensemble"][key] = d
        sub_o = g[["ensemble", "dataset", "tree", "mean_ours"]].rename(columns={"mean_ours": "mean"})
        sub_p = g[["ensemble", "dataset", "tree", "mean_paper"]].rename(columns={"mean_paper": "mean"})
        for tree in ("HLAST", "HLAST_D"):
            wo, no = win_rate(sub_o, tree)
            wp, npp = win_rate(sub_p, tree)
            summary["win_rates"][f"{metric}|{group}|{tree}>HT"] = dict(ours=wo, paper_same_pairs=wp, n=no)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # readable tables: F1 ours vs paper per ensemble
    lines = ["# Replicação — F1 (%) nosso × artigo", ""]
    f1 = m[m.metric == "f1"]
    for ens, ge in f1.groupby("ensemble"):
        lines += [f"## {ens}", "", "| base | " + " | ".join(f"{t} nosso | {t} artigo" for t in TREES) + " |",
                  "|---|" + "---:|---:|" * len(TREES)]
        for ds, gd in ge.groupby("dataset"):
            cells = []
            for t in TREES:
                r = gd[gd.tree == t]
                cells += ([f"{r.mean_ours.iloc[0]:.2f}", f"{r.mean_paper.iloc[0]:.2f}"] if len(r) else ["", ""])
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(os.path.join(OUT, "tables.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
