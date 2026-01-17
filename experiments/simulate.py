import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.config import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_W,
    DR_BASELINE_PATH,
    POLICY_CONFIGS_PATH,
    default_device,
)
from ssp_mmc_fsrs.io import load_policy_configs  # noqa: E402
from experiments.run_experiment_lib import (  # noqa: E402
    normalize_policy_list,
    run_experiment,
    setup_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSP-MMC-FSRS experiments.")
    parser.add_argument(
        "--simulation-type",
        default="unlim_time_lim_reviews",
        choices=["unlim_time_lim_reviews", "lim_time_unlim_reviews"],
        help="Simulation constraints mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for numpy.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Override device selection.",
    )
    parser.add_argument(
        "--policies",
        default="all",
        help=(
            "Comma-separated list of policies to evaluate. "
            "Examples: ssp-mmc,memrise,anki-sm-2,dr,interval or all."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment(args.seed)
    simulation_type = args.simulation_type
    w = DEFAULT_W
    device = args.device if args.device else default_device()

    raw_policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    if not raw_policies:
        raise SystemExit("At least one policy must be specified.")
    policies = normalize_policy_list(raw_policies)

    policy_configs = None
    if "ssp-mmc" in policies:
        try:
            policy_configs = load_policy_configs(POLICY_CONFIGS_PATH)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"Missing policy configs at {POLICY_CONFIGS_PATH}. "
                "Run the hyperparameter optimizer to generate them."
            ) from exc
    run_experiment(
        policy_configs=policy_configs,
        simulation_type=simulation_type,
        w=w,
        device=device,
        dr_baseline_path=DR_BASELINE_PATH,
        policies=policies,
    )


if __name__ == "__main__":
    main()
