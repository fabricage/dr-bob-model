"""Shared helpers used by more than one script.

Kept tiny on purpose: path bootstrapping, pretty printing, and a couple of
stats formulas that would otherwise be copy-pasted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def print_header(title: str) -> None:
    """Print a visible section title so CLI output is easy to scan."""
    bar = "=" * max(len(title) + 2, 60)
    print(f"\n{bar}\n {title}\n{bar}")


def print_frame(df: pl.DataFrame, title: str | None = None) -> None:
    """Print a Polars table with all rows/columns visible."""
    if title:
        print_header(title)
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=140, fmt_str_lengths=80):
        print(df)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. Returns NaN if there is no variation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_brown(r_half: float) -> float:
    """Lift a split-half correlation up to an estimated full-season reliability.

    If odd weeks and even weeks correlate at r, the Spearman–Brown formula
    estimates how reliable the AVERAGE of both halves (the full season) is:
        r_full = 2r / (1 + r)
    """
    if not np.isfinite(r_half):
        return float("nan")
    if r_half <= -1:
        return float("nan")
    return float(2.0 * r_half / (1.0 + r_half))


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path
