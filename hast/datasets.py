"""Dataset registry with the names used in the paper's tables."""
from __future__ import annotations

import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

REAL = {
    "Outdoor": "outdoor.arff",
    "Elec": "elecNormNew.arff",
    "Rialto": "rialto.arff",
    "Airlines": "airlines.arff",
    "CovType": "covtypeNorm.arff",
    "Nomao": "nomao.arff",
    "NOAA": "NOAA.arff",
    "Poker": "poker-lsn.arff",
    "INSECTSa": "INSECTS-abrupt_balanced.arff",
    "INSECTSi": "INSECTS-incremental_balanced.arff",
    "INSECTSg": "INSECTS-gradual_balanced.arff",
    # not available (requested from the authors): "LADPU", "Asfault"
}

# The authors' tables write the recurrent LED streams without the "r"
PAPER_ALIASES = {"LED-f(7,5,3,1,3,5,7)ra": "LED-f(7,5,3,1,3,5,7)a",
                 "LED-f(7,5,3,1,3,5,7)rg": "LED-f(7,5,3,1,3,5,7)g"}


def _synthetic(folder: str) -> dict[str, str]:
    path = os.path.join(DATA, folder, "streams.tsv")
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for name, fname, _cli in csv.reader(fh, delimiter="\t"):
                out[name] = os.path.join(DATA, folder, fname)
    return out


def synthetic() -> dict[str, str]:
    return _synthetic("synthetic")


def calibration() -> dict[str, str]:
    return _synthetic("calibration")


def imbalanced() -> dict[str, str]:
    return _synthetic("imbalanced")


def path_of(name: str) -> str:
    if name in REAL:
        return os.path.join(DATA, "real", REAL[name])
    for reg in (synthetic(), calibration(), imbalanced()):
        if name in reg:
            return reg[name]
    raise KeyError(name)


def paper_name(name: str) -> str:
    return PAPER_ALIASES.get(name, name)


def groups() -> dict[str, list[str]]:
    return {"real": list(REAL), "synthetic": list(synthetic()), "calibration": list(calibration()),
            "imbalanced": list(imbalanced())}
