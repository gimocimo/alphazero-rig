"""CLI entry: `python -m alphazero.plot RUN_DIR`.

Reads metrics.csv inside RUN_DIR, writes loss.png and winrate.png next to
it (or into --output-dir if provided). Re-run any time while training is
ongoing to refresh the plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .eval.plot_metrics import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics from a run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=str, help="Directory containing metrics.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write PNGs (defaults to RUN_DIR)",
    )
    args = parser.parse_args()

    paths = plot_all(args.run_dir, args.output_dir)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
