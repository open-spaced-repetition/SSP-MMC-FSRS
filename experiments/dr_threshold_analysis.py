import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.config import DEFAULT_SEED  # noqa: E402
from experiments.lib import (  # noqa: E402
    DEFAULT_BENCHMARK_RESULT,
    ensure_output_dirs,
    evaluate_dr_thresholds,
    load_fsrs_weights,
    plot_cost_vs_retention,
    plots_output_dir,
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
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="User ID for selecting FSRS weights and output directories.",
    )
    parser.add_argument(
        "--benchmark-result",
        type=Path,
        default=DEFAULT_BENCHMARK_RESULT,
        help="FSRS benchmark result JSONL to read user weights from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment(args.seed)
    ensure_output_dirs(user_id=args.user_id)
    w, _, _ = load_fsrs_weights(args.benchmark_result, args.user_id)
    plots_dir = plots_output_dir(args.user_id)
    r_range, costs = evaluate_dr_thresholds(w, plots_dir)
    plot_cost_vs_retention(costs, r_range, plots_dir)


if __name__ == "__main__":
    main()
