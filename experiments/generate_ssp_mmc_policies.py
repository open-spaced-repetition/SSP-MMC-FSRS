import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.config import DEFAULT_SEED  # noqa: E402
from ssp_mmc_fsrs.io import load_policy_configs  # noqa: E402
from experiments.lib import (  # noqa: E402
    DEFAULT_BENCHMARK_RESULT,
    ensure_output_dirs,
    generate_ssp_mmc_policies,
    load_fsrs_weights,
    policy_configs_path_for_user,
    setup_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SSP-MMC policy files and surface plots."
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
        help="User ID for selecting FSRS weights and policy output directory.",
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
    policy_configs_path = policy_configs_path_for_user(args.user_id)
    try:
        policy_configs = load_policy_configs(policy_configs_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Missing policy configs at {policy_configs_path}. "
            f"Run experiments/hyperparameter_optimizer.py --user-id {args.user_id} first."
        ) from exc
    w, weights_source, weights_partition = load_fsrs_weights(
        args.benchmark_result, args.user_id
    )
    generate_ssp_mmc_policies(
        policy_configs,
        w,
        user_id=args.user_id,
        weights_source=weights_source,
        weights_partition=weights_partition,
    )


if __name__ == "__main__":
    main()
