import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.io import load_policy_configs  # noqa: E402
from ssp_mmc_fsrs.solver import SSPMMCSolver  # noqa: E402
from experiments.lib import (  # noqa: E402
    DelayedKeyboardInterrupt,
    checkpoint_output_dir,
    policy_configs_path_for_user,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convergence checks for SSP-MMC-FSRS.")
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=ROOT_DIR.parent / "Anki-button-usage" / "button_usage.jsonl",
        help="Path to button usage JSONL file.",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=ROOT_DIR.parent / "srs-benchmark" / "result" / "FSRS-6-recency.jsonl",
        help="Path to FSRS parameters JSONL file.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Path to incremental results JSON file "
            "(defaults to outputs/checkpoints/convergence_incremental_results.json)."
        ),
    )
    parser.add_argument(
        "--unconverged",
        type=Path,
        default=None,
        help=(
            "Path to unconverged users JSON file "
            "(defaults to outputs/checkpoints/unconverged_users.json)."
        ),
    )
    parser.add_argument(
        "--policy-configs",
        type=Path,
        default=None,
        help=("Override policy configs JSON path (use a shared config for all users)."),
    )
    parser.add_argument("--workers", type=int, default=2, help="Process pool size.")
    return parser.parse_args()


def load_jsonl(path: Path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing_results(filename: Path):
    try:
        with DelayedKeyboardInterrupt():
            with open(filename, "r") as f:
                data = json.load(f)
                processed_users = data.get("processed_users", {})
                # JSON object keys are strings; normalize to int for set math.
                processed_users = {int(k): v for k, v in processed_users.items()}
                unconverged_users = data.get("unconverged_users", [])
                return processed_users, unconverged_users
    except FileNotFoundError:
        return {}, []


def save_results(processed_users, unconverged_users, filename: Path):
    # Write users in sorted order for stable incremental results.
    processed_sorted = {str(k): processed_users[k] for k in sorted(processed_users)}
    unconverged_sorted = sorted(unconverged_users)
    data = {
        "processed_users": processed_sorted,
        "unconverged_users": unconverged_sorted,
    }
    with DelayedKeyboardInterrupt():
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


def test_user(
    user_id, simulator_configs, parameters, policy_configs, policy_configs_path
):
    review_costs = np.array(simulator_configs[user_id]["review_costs"])
    first_rating_prob = np.array(simulator_configs[user_id]["first_rating_prob"])
    review_rating_prob = np.array(simulator_configs[user_id]["review_rating_prob"])
    first_rating_offsets = np.array(simulator_configs[user_id]["first_rating_offset"])
    first_session_lens = np.array(simulator_configs[user_id]["first_session_len"])
    forget_rating_offset = simulator_configs[user_id]["forget_rating_offset"]
    forget_session_len = simulator_configs[user_id]["forget_session_len"]
    w = parameters[user_id]["parameters"]["0"]

    if policy_configs is None:
        if policy_configs_path is None:
            raise ValueError("policy_configs or policy_configs_path is required.")
        policy_configs = load_policy_configs(policy_configs_path)

    for entry in policy_configs:
        solver = SSPMMCSolver(
            review_costs,
            first_rating_prob,
            review_rating_prob,
            first_rating_offsets,
            first_session_lens,
            forget_rating_offset,
            forget_session_len,
            w,
        )

        hyperparams = entry["params"]
        cost_matrix, _ = solver.solve(hyperparams)

        actual_max = np.max(cost_matrix)
        convergence_criteria = (cost_matrix == actual_max).sum() < cost_matrix.size / 20
        if not convergence_criteria:
            return False

    return True


def main():
    args = parse_args()
    output_dir = checkpoint_output_dir(None)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.results is None:
        args.results = output_dir / "convergence_incremental_results.json"
    if args.unconverged is None:
        args.unconverged = output_dir / "unconverged_users.json"

    simulator_configs = load_jsonl(args.button_usage)
    simulator_configs = {config["user"]: config for config in simulator_configs}

    parameters = load_jsonl(args.parameters)
    parameters = {param["user"]: param for param in parameters}

    results_file = args.results

    processed_users, unconverged_users = load_existing_results(results_file)
    print(f"Loaded {len(processed_users)} previously processed users")

    all_user_ids = set(parameters.keys())
    remaining_users = all_user_ids - set(processed_users.keys())

    if not remaining_users:
        print("All users have already been processed!")
        print(f"Total unconverged users: {len(unconverged_users)}")
        print("Unconverged user IDs:")
        print("\n".join(str(user_id) for user_id in unconverged_users))
        return

    print(f"Processing {len(remaining_users)} remaining users")

    policy_configs = None
    policy_configs_path = None
    if args.policy_configs is not None:
        policy_configs_path = args.policy_configs
        try:
            policy_configs = load_policy_configs(policy_configs_path)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"Missing policy configs at {policy_configs_path}. "
                "Run experiments/hyperparameter_optimizer.py or pass --policy-configs."
            ) from exc
    else:
        missing_users = [
            user_id
            for user_id in remaining_users
            if not policy_configs_path_for_user(user_id).exists()
        ]
        if missing_users:
            sample = ", ".join(str(user_id) for user_id in missing_users[:10])
            extra = ""
            if len(missing_users) > 10:
                extra = f", ... (+{len(missing_users) - 10} more)"
            raise SystemExit(
                "Missing per-user policy configs for user IDs: "
                f"{sample}{extra}. "
                "Run experiments/hyperparameter_optimizer.py --user-id <id> "
                "for those users, or pass --policy-configs to use a shared config."
            )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_user = {
            executor.submit(
                test_user,
                user_id,
                simulator_configs,
                parameters,
                policy_configs,
                policy_configs_path_for_user(user_id)
                if policy_configs is None
                else None,
            ): user_id
            for user_id in remaining_users
        }

        converged_count = sum(1 for result in processed_users.values() if result)
        tested_count = len(processed_users)
        pbar = tqdm(as_completed(future_to_user), total=len(remaining_users))

        for future in pbar:
            user_id = future_to_user[future]
            result = future.result()

            processed_users[user_id] = result

            if not result and user_id not in unconverged_users:
                unconverged_users.append(user_id)
            elif result and user_id in unconverged_users:
                unconverged_users.remove(user_id)

            save_results(processed_users, unconverged_users, results_file)

            if result:
                converged_count += 1
            tested_count += 1
            pbar.set_description(
                f"{converged_count / tested_count:.2%} converged ({converged_count}/{tested_count})"
            )

    print("Processing complete!")
    print(f"Total unconverged users: {len(unconverged_users)}")
    print("Unconverged user IDs:")
    print("\n".join(str(user_id) for user_id in unconverged_users))

    with DelayedKeyboardInterrupt():
        with open(args.unconverged, "w") as f:
            json.dump(unconverged_users, f)


if __name__ == "__main__":
    main()
