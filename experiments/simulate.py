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
    default_device,
)
from ssp_mmc_fsrs.io import load_policy_configs  # noqa: E402
from experiments.lib import (  # noqa: E402
    DEFAULT_BENCHMARK_RESULT,
    DEFAULT_BUTTON_USAGE,
    dr_baseline_path_for_user,
    load_button_usage_config,
    load_fsrs_weights,
    normalize_policy_list,
    plot_pareto_frontier,
    policy_configs_path_for_user,
    plots_output_dir,
    run_experiment,
    setup_environment,
    simulation_results_path_for_user,
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
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="User ID for selecting FSRS weights and SSP-MMC policies.",
    )
    parser.add_argument(
        "--benchmark-result",
        type=Path,
        default=DEFAULT_BENCHMARK_RESULT,
        help="FSRS benchmark result JSONL to read user weights from.",
    )
    parser.add_argument(
        "--policy-configs",
        type=Path,
        default=None,
        help=(
            "Override policy configs JSON path "
            "(defaults to outputs/checkpoints/user_<id>/policy_configs.json)."
        ),
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE,
        help="Button usage JSONL to read simulation costs/probabilities from.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip simulation and only plot Pareto frontier from saved results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment(args.seed)
    if args.plot_only:
        results_path = simulation_results_path_for_user(args.user_id)
        if not results_path.exists():
            raise SystemExit(
                f"Simulation results not found at {results_path}. "
                "Run experiments/simulate.py first."
            )
        policy_configs_path = (
            args.policy_configs
            if args.policy_configs is not None
            else policy_configs_path_for_user(args.user_id)
        )
        try:
            policy_configs = load_policy_configs(policy_configs_path)
        except FileNotFoundError:
            policy_configs = []
        plots_dir = plots_output_dir(args.user_id)
        plot_pareto_frontier(
            results_path,
            policy_configs,
            plots_dir,
            user_id=args.user_id,
        )
        return
    simulation_type = args.simulation_type
    w, _, _ = load_fsrs_weights(args.benchmark_result, args.user_id)
    device = args.device if args.device else default_device()
    button_usage = load_button_usage_config(args.button_usage, args.user_id)

    raw_policies = [
        policy.strip() for policy in args.policies.split(",") if policy.strip()
    ]
    if not raw_policies:
        raise SystemExit("At least one policy must be specified.")
    policies = normalize_policy_list(raw_policies)

    policy_configs = None
    if "ssp-mmc" in policies:
        policy_configs_path = (
            args.policy_configs
            if args.policy_configs is not None
            else policy_configs_path_for_user(args.user_id)
        )
        try:
            policy_configs = load_policy_configs(policy_configs_path)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"Missing policy configs at {policy_configs_path}. "
                f"Run experiments/hyperparameter_optimizer.py --user-id {args.user_id} first."
            ) from exc
    run_experiment(
        policy_configs=policy_configs,
        simulation_type=simulation_type,
        w=w,
        device=device,
        dr_baseline_path=dr_baseline_path_for_user(args.user_id),
        policies=policies,
        user_id=args.user_id,
        button_usage=button_usage,
    )


if __name__ == "__main__":
    main()
