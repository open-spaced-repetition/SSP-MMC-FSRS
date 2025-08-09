import json
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from colorama import Fore, Style
from tqdm import tqdm  # Optional: for progress bar

from script import SSPMMCSolver

simulator_config_file = Path("../Anki-button-usage/button_usage.jsonl")
simulator_configs = list(
    map(lambda x: json.loads(x), open(simulator_config_file).readlines())
)
simulator_configs = {
    config["user"]: config
    for config in map(lambda x: json.loads(x), open(simulator_config_file).readlines())
}

parameters_file = Path("../srs-benchmark/result/FSRS-rs.jsonl")
parameters = list(map(lambda x: json.loads(x), open(parameters_file).readlines()))
parameters = {
    param["user"]: param
    for param in map(lambda x: json.loads(x), open(parameters_file).readlines())
}

# To prevent files from getting corrupted when manually ending tests
class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt')
        print(Fore.RED + 'Delaying KeyboardInterrupt')
        print(Style.RESET_ALL)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def test_user(user_id):
    review_costs = np.array(simulator_configs[user_id]["review_costs"])
    first_rating_prob = np.array(simulator_configs[user_id]["first_rating_prob"])
    review_rating_prob = np.array(simulator_configs[user_id]["review_rating_prob"])
    first_rating_offsets = np.array(simulator_configs[user_id]["first_rating_offset"])
    first_session_lens = np.array(simulator_configs[user_id]["first_session_len"])
    forget_rating_offset = simulator_configs[user_id]["forget_rating_offset"]
    forget_session_len = simulator_configs[user_id]["forget_session_len"]
    w = parameters[user_id]["parameters"]["0"]

    list_of_dictionaries = [
        [{'a0': 'log', 'a1': 1.04, 'a2': 10.0, 'a3': 4.14, 'a4': 1.15, 'a5': 0.03, 'a6': -5.0, 'a7': -1.03, 'a8': 5.0},
         'Maximum knowledge'],
        [{'a0': 'log', 'a1': 1.37, 'a2': 10.0, 'a3': 1.24, 'a4': -5.0, 'a5': -5.0, 'a6': -5.0, 'a7': 5.0, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 1.24, 'a2': 10.0, 'a3': 3.16, 'a4': 1.29, 'a5': -1.61, 'a6': -5.0, 'a7': 0.24, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 1.33, 'a2': 1.49, 'a3': -1.2, 'a4': -5.0, 'a5': -5.0, 'a6': -5.0, 'a7': 5.0, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 1.23, 'a2': 10.0, 'a3': 3.98, 'a4': 2.46, 'a5': -0.16, 'a6': -5.0, 'a7': -3.18, 'a8': 5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.1, 'a2': 0.1, 'a3': 5.0, 'a4': 3.44, 'a5': -3.71, 'a6': -5.0, 'a7': -3.1, 'a8': -5.0},
         None],
        [{'a0': 'log', 'a1': 1.61, 'a2': 9.83, 'a3': -0.83, 'a4': -5.0, 'a5': -5.0, 'a6': -5.0, 'a7': 4.7, 'a8': 5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.14, 'a2': 0.1, 'a3': 5.0, 'a4': 5.0, 'a5': -5.0, 'a6': -5.0, 'a7': -2.67, 'a8': -5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.17, 'a2': 0.1, 'a3': 5.0, 'a4': 5.0, 'a5': -4.55, 'a6': -5.0, 'a7': -3.46, 'a8': -5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.18, 'a2': 0.1, 'a3': 5.0, 'a4': 5.0, 'a5': -4.69, 'a6': -5.0, 'a7': -3.56, 'a8': -5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.18, 'a2': 0.11, 'a3': 5.0, 'a4': 5.0, 'a5': -4.73, 'a6': -5.0, 'a7': -3.6, 'a8': -5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.19, 'a2': 0.11, 'a3': 5.0, 'a4': 5.0, 'a5': -5.0, 'a6': -5.0, 'a7': -3.81, 'a8': -5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.19, 'a2': 0.11, 'a3': 5.0, 'a4': 5.0, 'a5': -5.0, 'a6': -5.0, 'a7': -3.95, 'a8': -5.0},
         None],
        [{'a0': 'log', 'a1': 2.84, 'a2': 10.0, 'a3': 1.45, 'a4': 1.69, 'a5': 0.7, 'a6': -5.0, 'a7': -1.74, 'a8': 2.2},
         None],
        [{'a0': 'log', 'a1': 2.71, 'a2': 4.47, 'a3': 1.55, 'a4': 2.23, 'a5': 0.15, 'a6': -5.0, 'a7': -1.41, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 3.94, 'a2': 10.0, 'a3': 0.96, 'a4': 1.36, 'a5': -0.4, 'a6': -5.0, 'a7': -2.14, 'a8': 2.31},
         None],
        [{'a0': 'log', 'a1': 6.78, 'a2': 8.31, 'a3': -2.59, 'a4': -5.0, 'a5': -2.6, 'a6': -5.0, 'a7': 0.67, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 1.84, 'a2': 0.27, 'a3': 5.0, 'a4': 2.01, 'a5': -2.97, 'a6': -5.0, 'a7': -3.91, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 10.0, 'a2': 0.34, 'a3': 4.78, 'a4': -0.86, 'a5': -0.85, 'a6': -5.0, 'a7': -3.98, 'a8': 5.0},
         None],
        [{'a0': 'no_log', 'a1': 10.0, 'a2': 0.1, 'a3': 5.0, 'a4': 4.19, 'a5': -0.96, 'a6': -3.01, 'a7': -3.18, 'a8': 5.0},
         'Balanced'],
        [{'a0': 'no_log', 'a1': 0.11, 'a2': 0.64, 'a3': -2.31, 'a4': 5.0, 'a5': 5.0, 'a6': -1.52, 'a7': -1.01, 'a8': 5.0},
         None],
        [{'a0': 'no_log', 'a1': 4.34, 'a2': 0.1, 'a3': 3.39, 'a4': 5.0, 'a5': -2.2, 'a6': 5.0, 'a7': -3.3, 'a8': 5.0},
         None],
        [{'a0': 'log', 'a1': 10.0, 'a2': 0.37, 'a3': 3.06, 'a4': 0.91, 'a5': -1.15, 'a6': -5.0, 'a7': -4.53, 'a8': 5.0},
         None],
        [{'a0': 'no_log', 'a1': 0.1, 'a2': 0.1, 'a3': 2.9, 'a4': 5.0, 'a5': -3.79, 'a6': 5.0, 'a7': -2.2, 'a8': 1.33},
         'Maximum efficiency']]

    for entry in list_of_dictionaries:
        # Solver needs to be re-initialized every time
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

        hyperparams = entry[0]
        cost_matrix, _ = solver.solve(hyperparams)

        # Stricter criteria
        actual_max = np.max(cost_matrix)
        convergence_criteria = (cost_matrix == actual_max).sum() < cost_matrix.size / 20
        if not convergence_criteria:
            return False

    # Only return True if the matrix converged with all hyperparameters
    return True


def load_existing_results(filename="results.json"):
    """Load existing results from file if it exists"""
    try:
        with DelayedKeyboardInterrupt():
            with open(filename, "r") as f:
                data = json.load(f)
                return data.get("processed_users", {}), data.get("unconverged_users", [])
    except FileNotFoundError:
        return {}, []


def save_results(processed_users, unconverged_users, filename="results.json"):
    """Save current results to file"""
    data = {
        "processed_users": processed_users,
        "unconverged_users": unconverged_users
    }
    with DelayedKeyboardInterrupt():
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    results_file = "convergence_incremental_results.json"

    # Load existing results
    processed_users, unconverged_users = load_existing_results(results_file)
    print(f"Loaded {len(processed_users)} previously processed users")

    # Get users that still need to be processed
    all_user_ids = set(parameters.keys())
    remaining_users = all_user_ids - set(processed_users.keys())

    if not remaining_users:
        print("All users have already been processed!")
        print(f"Total unconverged users: {len(unconverged_users)}")
        print("Unconverged user IDs:")
        print("\n".join(str(user_id) for user_id in unconverged_users))
    else:
        print(f"Processing {len(remaining_users)} remaining users")

        # Create process pool
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit only remaining tasks
            future_to_user = {
                executor.submit(test_user, user_id): user_id
                for user_id in remaining_users
            }

            converged_count = sum(1 for result in processed_users.values() if result)
            tested_count = len(processed_users)
            pbar = tqdm(as_completed(future_to_user), total=len(remaining_users))

            for future in pbar:
                user_id = future_to_user[future]
                result = future.result()

                # Update processed users
                processed_users[user_id] = result

                # Update unconverged users list
                if not result and user_id not in unconverged_users:
                    unconverged_users.append(user_id)
                elif result and user_id in unconverged_users:
                    unconverged_users.remove(user_id)

                # Save results after each user
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

    # Also save the unconverged users list in the original format for compatibility
    with DelayedKeyboardInterrupt():
        with open("unconverged_users.json", "w") as f:
            json.dump(unconverged_users, f)
