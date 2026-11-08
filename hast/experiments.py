"""Experiment registry: (ensemble, base tree) configurations of the paper.

All ensembles use 100 members and MOA defaults (paper, Sec. 5). The base tree
keeps the hyper-parameters each ensemble uses by default for its Hoeffding
Tree; HLAST/HLAST_D only add the leaf change detector (HDDM_A, MOA defaults).
``java_cli`` gives the equivalent MOA command used for the reference runs.
"""
from __future__ import annotations

from hast.ensembles.arf import AdaptiveRandomForest
from hast.ensembles.levbag import LeveragingBag
from hast.ensembles.srp import StreamingRandomPatches
from hast.trees.hoeffding import HLAST, HoeffdingTree
from hast.trees.variants import ARFHLAST, ARFHoeffdingTree, ARTEHLAST, ARTEHoeffdingTree

ENSEMBLES = ("LevBag", "ARF", "SRP", "ARTE", "ARE")
TREES = ("HT", "HLAST", "HLAST_D")


def make_tree(ensemble: str, tree: str, **tree_kw):
    dist = tree == "HLAST_D"
    if ensemble == "LevBag":
        base = {}
        return HoeffdingTree(**base, **tree_kw) if tree == "HT" else HLAST(distribution=dist, **base, **tree_kw)
    if ensemble == "SRP":
        base = dict(grace_period=50, split_confidence=0.01)
        return HoeffdingTree(**base, **tree_kw) if tree == "HT" else HLAST(distribution=dist, **base, **tree_kw)
    if ensemble in ("ARF", "ARE"):  # ARE extends ARF (reconstruction, see hast/ensembles/are.py)
        return ARFHoeffdingTree(**tree_kw) if tree == "HT" else ARFHLAST(distribution=dist, **tree_kw)
    if ensemble == "ARTE":
        return ARTEHoeffdingTree(**tree_kw) if tree == "HT" else ARTEHLAST(distribution=dist, **tree_kw)
    raise ValueError(ensemble)


def make_model(ensemble: str, tree: str, ensemble_size: int = 100, tree_kw: dict | None = None,
               impl: str = "fast", tree_cls=None, **ens_kw):
    """Build an ensemble. ``impl="fast"`` uses the numba trees (default);
    ``impl="object"`` the readable reference trees (same results, slower).
    ``tree_cls``: subclass of ``FastTree`` to use instead (e.g. HLAST-R)."""
    if impl == "fast":
        from hast.trees.fast import make_fast_tree
        t = make_fast_tree(ensemble, tree, cls=tree_cls, **(tree_kw or {}))
    else:
        t = make_tree(ensemble, tree, **(tree_kw or {}))
    if ensemble == "LevBag":
        return LeveragingBag(t, ensemble_size=ensemble_size, **ens_kw)
    if ensemble == "ARF":
        return AdaptiveRandomForest(t, ensemble_size=ensemble_size, **ens_kw)
    if ensemble == "SRP":
        return StreamingRandomPatches(t, ensemble_size=ensemble_size, **ens_kw)
    if ensemble == "ARTE":
        from hast.ensembles.arte import ARTE
        return ARTE(t, ensemble_size=ensemble_size, **ens_kw)
    if ensemble == "ARE":
        from hast.ensembles.are import AdaptiveRegularizedEnsemble
        return AdaptiveRegularizedEnsemble(t, ensemble_size=ensemble_size, **ens_kw)
    raise ValueError(ensemble)


def java_cli(ensemble: str, tree: str, ensemble_size: int = 100) -> str:
    d = " -D" if tree == "HLAST_D" else ""
    if ensemble == "LevBag":
        base = "trees.HoeffdingTree" if tree == "HT" else f"(trees.HLAST -x HDDM_A_Test{d})"
        return f"meta.LeveragingBag -s {ensemble_size} -l {base}"
    if ensemble == "ARF":
        base = ("ARFHoeffdingTree -e 2000000 -g 50 -c 0.01" if tree == "HT"
                else f"ARFHLAST -e 2000000 -g 50 -c 0.01 -x HDDM_A_Test{d}")
        return f"meta.AdaptiveRandomForest -s {ensemble_size} -l ({base})"
    if ensemble == "SRP":
        base = ("trees.HoeffdingTree -g 50 -c 0.01" if tree == "HT"
                else f"trees.HLAST -g 50 -c 0.01 -x HDDM_A_Test{d}")
        return f"meta.StreamingRandomPatches -s {ensemble_size} -l ({base})"
    if ensemble == "ARTE":
        base = ("ARTEHoeffdingTree -e 2000000 -g 100 -c 0.01 -n ARTEAttributeClassObserver" if tree == "HT"
                else f"ARTEHLAST -e 2000000 -g 100 -c 0.01 -n ARTEAttributeClassObserver -x HDDM_A_Test{d}")
        return f"meta.ARTE -s {ensemble_size} -l ({base})"
    raise ValueError(f"no Java reference for {ensemble}")
