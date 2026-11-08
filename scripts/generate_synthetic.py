"""Generate the 24 synthetic streams of the paper with MOA's own generators.

The paper takes the streams from the KUE benchmark (Cano & Krawczyk, 2020):
10^6 instances each; AGR/SEA/LED with abrupt or gradual (recurrent or
monotone) drifts and RBF/HYPER with incremental drift at three speeds.
Drift positions and gradual widths are not reported, so we ASSUME:

* drifts evenly spaced (segment = 10^6 / #concepts), chained with MOA
  ``ConceptDriftStream`` (positions relative to each nested stream);
* gradual width = 10% of the segment; abrupt width = 1;
* generator defaults otherwise (AGR perturbation 5%, SEA noise 10%, LED noise
  10% with 17 irrelevant attributes, RBF 50 centroids / 5 classes / 10
  attributes, HYPER 10 attributes / 2 drifting / 5% noise).

``--seed-offset K`` (K > 0) shifts every generator seed by K, producing the
*calibration* streams for the regularization study (never used for evaluation).

usage: python scripts/generate_synthetic.py [--out data/synthetic] [--seed-offset 0] [-j 4] [--only NAME...]
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAVA = os.path.join(ROOT, "third_party", "jdk", "bin", "java")
MOA = os.path.join(ROOT, "java_ref", "lib", "moa.jar")
AGENT = os.path.join(ROOT, "java_ref", "lib", "sizeofag-1.1.0.jar")
N = 1_000_000


def chain(gens: list[str], gradual: bool, n: int = N) -> str:
    seg = n // len(gens)
    width = max(1, seg // 10) if gradual else 1
    s = f"({gens[-1]})"
    for g in reversed(gens[:-1]):
        s = f"(ConceptDriftStream -s ({g}) -d {s} -p {seg} -w {width})"
    return s[1:-1]


def streams(seed_offset: int = 0) -> dict[str, str]:
    i = 1 + seed_offset  # instance seed
    r = 1 + seed_offset  # model seed
    agr = lambda f: f"generators.AgrawalGenerator -f {f} -i {i}"
    sea = lambda f: f"generators.SEAGenerator -f {f} -i {i}"
    led = lambda d: f"generators.LEDGeneratorDrift -d {d} -i {i}"
    out: dict[str, str] = {}
    for kind, seq in (("r", (7, 8, 9, 10, 9, 8, 7)), ("", tuple(range(1, 11))), ("", tuple(range(10, 0, -1)))):
        for drift in ("a", "g"):
            name = f"AGR-f({','.join(map(str, seq))}){kind}{drift}"
            out[name] = chain([agr(f) for f in seq], drift == "g")
    for kind, seq in (("r", (1, 2, 3, 4, 3, 2, 1)), ("", (1, 2, 3, 4)), ("", (4, 3, 2, 1))):
        for drift in ("a", "g"):
            out[f"SEA-f({','.join(map(str, seq))}){kind}{drift}"] = chain([sea(f) for f in seq], drift == "g")
    for kind, seq in (("r", (7, 5, 3, 1, 3, 5, 7)), ("", (1, 3, 5, 7)), ("", (7, 5, 3, 1))):
        for drift in ("a", "g"):
            out[f"LED-f({','.join(map(str, seq))}){kind}{drift}"] = chain([led(d) for d in seq], drift == "g")
    for tag, speed in (("s", 1e-5), ("m", 1e-4), ("f", 1e-3)):
        out[f"RBF{tag}"] = f"generators.RandomRBFGeneratorDrift -c 5 -a 10 -n 50 -k 50 -s {speed} -r {r} -i {i}"
    for tag, mag in (("s", 1e-4), ("m", 1e-3), ("f", 1e-2)):
        out[f"HYPER{tag}"] = f"generators.HyperplaneGenerator -a 10 -k 2 -t {mag} -i {i}"
    return out


def safe(name: str) -> str:
    return name.replace("(", "_").replace(")", "").replace(",", "-")


def generate(name: str, cli: str, out_dir: str, n: int) -> str:
    path = os.path.join(out_dir, safe(name) + ".arff")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return f"skip {name}"
    tmp = path + ".tmp"
    task = f"WriteStreamToARFFFile -s ({cli}) -f {tmp} -m {n}"
    subprocess.run([JAVA, "-Xmx2g", "-cp", MOA, f"-javaagent:{AGENT}", "moa.DoTask", task], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, path)
    with open(os.path.join(out_dir, "streams.tsv"), "a") as fh:
        fh.write(f"{name}\t{safe(name)}.arff\t{cli}\n")
    return f"done {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "synthetic"))
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("-n", type=int, default=N)
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = {k: v for k, v in streams(a.seed_offset).items() if not a.only or k in a.only}
    with ThreadPoolExecutor(a.j) as ex:
        for msg in ex.map(lambda kv: generate(kv[0], kv[1], a.out, a.n), todo.items()):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
