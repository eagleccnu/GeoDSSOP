#!/usr/bin/env python3
"""Plot observed versus predicted S2 from evaluate_geodssop.py output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install matplotlib to use the plotting script") from error
    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = np.asarray([float(row["observed_s2"]) for row in rows])
    predicted = np.asarray([float(row["predicted_s2"]) for row in rows])
    figure, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    axis.scatter(observed, predicted, s=8, alpha=0.35, linewidths=0)
    axis.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Observed $S^2$", ylabel="Predicted $S^2$")
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(args.output, dpi=220)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
