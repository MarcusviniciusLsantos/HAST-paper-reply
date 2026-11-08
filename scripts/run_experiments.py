"""Run the prequential experiments (resumable, parallel).

Each (dataset, ensemble, tree) run is written to
``results/raw/<dataset>__<ensemble>__<tree>.json``; existing files are skipped,
so the command can be re-launched after an interruption or split across
machines (``--shard i/n``).

Examples
--------
python scripts/run_experiments.py --datasets real --ensembles ARTE SRP --trees HT HLAST HLAST_D -j 4
python scripts/run_experiments.py --datasets Elec CovType --ensembles ARF --trees HT --size 100
python scripts/run_experiments.py --datasets synthetic --shard 0/3        # first of 3 machines
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hast import datasets  # noqa: E402

OUT = os.path.join(ROOT, "results", "raw")


def out_path(out_dir, ds, ens, tree, tag=""):
    safe = ds.replace("(", "_").replace(")", "").replace(",", "-")
    return os.path.join(out_dir, f"{safe}__{ens}__{tree}{tag}.json")


def _worker(job):
    warnings.simplefilter("ignore")
    ds, ens, tree, size, impl, out_dir, n_max, tag, ens_kw, tree_kw = job
    path = out_path(out_dir, ds, ens, tree, tag)
    if os.path.exists(path):
        return f"skip {os.path.basename(path)}"
    from hast.core.arff import load_arff
    from hast.eval.prequential import run_prequential
    from hast.experiments import make_model
    from hast.trees.hoeffding import Schema
    t0 = time.time()
    try:
        stream = load_arff(datasets.path_of(ds))
        stream.name = ds
        model = make_model(ens, tree, ensemble_size=size, impl=impl, tree_kw=tree_kw, **ens_kw)
        model.set_schema(Schema.from_header(stream.header))
        res = run_prequential(model, stream, n_max=n_max, ensemble_name=ens, tree_name=tree)
        res.extra.update(ensemble_size=size, impl=impl, ens_kw=ens_kw, tree_kw=tree_kw, tag=tag)
        for attr in ("n_changes", "n_trained", "n_rejected"):
            if hasattr(model, attr):
                res.extra[attr] = getattr(model, attr)
        if hasattr(model, "ensemble") and model.ensemble and hasattr(model.ensemble[0], "n_drifts"):
            res.extra["drifts"] = int(sum(m.n_drifts for m in model.ensemble))
        res.save(path + ".tmp")
        os.replace(path + ".tmp", path)
        s = res.summary()
        return (f"done {ds} {ens} {tree}: F1={s['f1']:.2f}±{s['f1_std']:.2f} acc={s['accuracy']:.2f} "
                f"size={s['tree_size']:.1f} cpu={s['cpu_s']:.0f}s wall={time.time() - t0:.0f}s")
    except Exception:
        return f"FAIL {ds} {ens} {tree}\n{traceback.format_exc()}"


def expand(names):
    g = datasets.groups()
    out = []
    for n in names:
        out.extend(g[n] if n in g else [n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["real"])
    ap.add_argument("--ensembles", nargs="+", default=["ARTE", "SRP", "ARF", "LevBag", "ARE"])
    ap.add_argument("--trees", nargs="+", default=["HT", "HLAST", "HLAST_D"])
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--impl", default="fast", choices=["fast", "object"])
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for output files (variants, calibration)")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: run the i-th of n interleaved shards (several: 0,1/7)")
    ap.add_argument("--fresh-workers", action="store_true",
                    help="one new process per run (memory-heavy datasets; pays the numba compile each time)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from hast.core.arff import load_arff
    sizes = {}
    for ds in expand(a.datasets):
        sizes[ds] = len(load_arff(datasets.path_of(ds)).y)  # also builds the .npz caches
    jobs = [(ds, e, t, a.size, a.impl, a.out, a.max_instances, a.tag, {}, {})
            for ds in sizes for e in a.ensembles for t in a.trees]
    jobs.sort(key=lambda j: -sizes[j[0]])  # longest first
    idx, n = a.shard.split("/")
    mine = {int(i) for i in idx.split(",")}
    jobs = [j for k, j in enumerate(jobs) if k % int(n) in mine]
    todo = [j for j in jobs if not os.path.exists(out_path(a.out, j[0], j[1], j[2], j[7]))]
    print(f"{len(jobs)} jobs in shard, {len(todo)} to run, {a.jobs} workers", flush=True)
    log = open(os.path.join(a.out, "run.log"), "a")
    ctx = mp.get_context("spawn")
    # long-lived workers compile numba once; --fresh-workers returns all memory after each run
    with ctx.Pool(a.jobs, maxtasksperchild=1 if a.fresh_workers else None) as pool:
        for msg in pool.imap_unordered(_worker, todo):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(stamp, msg, flush=True)
            log.write(f"{stamp} {msg}\n")
            log.flush()


if __name__ == "__main__":
    main()
