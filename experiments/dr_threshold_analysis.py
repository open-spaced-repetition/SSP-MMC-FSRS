import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.config import DEFAULT_SEED, DEFAULT_W  # noqa: E402
from experiments.lib import (  # noqa: E402
    ensure_output_dirs,
    evaluate_dr_thresholds,
    plot_cost_vs_retention,
    setup_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DR thresholds and plot cost vs retention."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for numpy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment(args.seed)
    ensure_output_dirs()
    r_range, costs = evaluate_dr_thresholds(DEFAULT_W)
    plot_cost_vs_retention(costs, r_range)


if __name__ == "__main__":
    main()
