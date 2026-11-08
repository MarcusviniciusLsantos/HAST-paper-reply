"""Minimal dense ARFF reader mirroring MOA's ``ArffFileStream`` (class = last attribute).

Nominal values are stored as their index in the header declaration, as in
WEKA/MOA ``Instance.value``. Parsed streams are cached as ``.npz`` files next to
the source.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ATTR_RE = re.compile(r"^@attribute\s+('(?:[^']|\\')*'|\"[^\"]*\"|[^\s{]+)\s*(.*)$", re.IGNORECASE)


@dataclass
class Header:
    relation: str
    names: list[str]
    nominal_values: list[list[str] | None]  # None for numeric attributes (class included)

    @property
    def n_attributes(self) -> int:
        """Number of input attributes (class excluded)."""
        return len(self.names) - 1

    @property
    def n_classes(self) -> int:
        return len(self.nominal_values[-1])

    @property
    def is_nominal(self) -> np.ndarray:
        return np.array([v is not None for v in self.nominal_values[:-1]], dtype=np.bool_)

    @property
    def n_values(self) -> np.ndarray:
        """Number of declared values per nominal input attribute (0 for numeric)."""
        return np.array([len(v) if v is not None else 0 for v in self.nominal_values[:-1]], dtype=np.int64)


@dataclass
class Stream:
    header: Header
    X: np.ndarray  # (n, d) float64; nominal attributes hold value indices
    y: np.ndarray  # (n,) int64
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def _split_values(spec: str) -> list[str]:
    inner = spec.strip()[1:-1]
    reader = csv.reader(io.StringIO(inner), quotechar="'", skipinitialspace=True)
    vals = next(reader)
    return [v.strip().strip('"') for v in vals]


def _parse_header(fh) -> tuple[Header, int]:
    relation = ""
    names: list[str] = []
    nominal: list[list[str] | None] = []
    n_lines = 0
    for raw in fh:
        n_lines += 1
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        low = line.lower()
        if low.startswith("@relation"):
            relation = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
        elif low.startswith("@attribute"):
            m = _ATTR_RE.match(line)
            if not m:
                raise ValueError(f"cannot parse attribute line: {line[:80]}")
            name, spec = m.group(1).strip("'\""), m.group(2).strip()
            names.append(name)
            if spec.startswith("{"):
                nominal.append(_split_values(spec[: spec.rindex("}") + 1]))
            elif spec.lower().split()[0].split("[")[0] in ("numeric", "real", "integer"):
                nominal.append(None)  # also "INTEGER [1,41]"
            else:
                raise ValueError(f"unsupported attribute type: {spec[:40]}")
        elif low.startswith("@data"):
            return Header(relation, names, nominal), n_lines
    raise ValueError("no @data section")


def load_arff(path: str, use_cache: bool = True) -> Stream:
    cache = path + ".npz"
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header, n_header_lines = _parse_header(fh)
    if use_cache and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        z = np.load(cache)
        return Stream(header, z["X"], z["y"], name)

    df = pd.read_csv(path, skiprows=n_header_lines, header=None, quotechar="'",
                     skipinitialspace=True, dtype=str, comment="%", keep_default_na=False)
    if df.shape[1] == len(header.names) + 1 and (df.iloc[:, -1].str.strip() == "").all():
        df = df.iloc[:, :-1]  # MOA's WriteStreamToARFFFile ends every row with a comma
    if df.shape[1] != len(header.names):
        raise ValueError(f"{path}: {df.shape[1]} columns, header declares {len(header.names)}")
    n = len(df)
    d = header.n_attributes
    X = np.empty((n, d), dtype=np.float64)
    for j in range(d + 1):
        col = df.iloc[:, j].str.strip().str.strip('"').str.strip()
        vals = header.nominal_values[j]
        if vals is None:
            col = col.mask(col.str.lower().isin(["?", "nan", ""]), None)
            arr = pd.to_numeric(col).to_numpy(dtype=np.float64)
        else:
            mapping = {v: float(i) for i, v in enumerate(vals)}
            mapping["?"] = np.nan
            mapped = col.map(mapping)
            if mapped.isna().any() and not (col == "?").any():
                bad = col[mapped.isna()].iloc[0]
                raise ValueError(f"{path}: value {bad!r} not declared for attribute {header.names[j]}")
            arr = mapped.to_numpy(dtype=np.float64)
        if j < d:
            X[:, j] = arr
        else:
            if np.isnan(arr).any():
                raise ValueError("missing class values are not supported")
            y = arr.astype(np.int64)
    if use_cache:  # atomic: several cluster tasks may build the same cache at once
        tmp = f"{cache}.{os.getpid()}.tmp.npz"
        np.savez(tmp, X=X, y=y)
        os.replace(tmp, cache)
    return Stream(header, X, y, name)
