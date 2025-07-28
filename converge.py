import json
from pathlib import Path
import numpy as np
from script import SSPMMCSolver, COST_MAX
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Optional: for progress bar

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


def test_user(user_id):
    review_costs = np.array(simulator_configs[user_id]["review_costs"])
    first_rating_prob = np.array(simulator_configs[user_id]["first_rating_prob"])
    review_rating_prob = np.array(simulator_configs[user_id]["review_rating_prob"])
    first_rating_offsets = np.array(simulator_configs[user_id]["first_rating_offset"])
    first_session_lens = np.array(simulator_configs[user_id]["first_session_len"])
    forget_rating_offset = simulator_configs[user_id]["forget_rating_offset"]
    forget_session_len = simulator_configs[user_id]["forget_session_len"]
    w = parameters[user_id]["parameters"]["0"]

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
    cost_matrix, _ = solver.solve(verbose=False)
    return (cost_matrix == COST_MAX).sum() < cost_matrix.size / 8


if __name__ == "__main__":
    # Create process pool
    with ProcessPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_user = {
            executor.submit(test_user, user_id): user_id
            for user_id in parameters.keys()
        }

        results = []
        converged_count = 0
        tested_count = 0
        pbar = tqdm(as_completed(future_to_user), total=len(parameters))

        for future in pbar:
            user_id = future_to_user[future]
            result = future.result()
            results.append((user_id, result))
            if result:
                converged_count += 1
            tested_count += 1
            pbar.set_description(
                f"{converged_count/tested_count:.2%} converged ({converged_count}/{tested_count})"
            )

    # Get unconverged users
    unconverged_users = [user_id for user_id, result in results if not result]

    print("Unconverged user IDs:")
    print("\n".join(str(user_id) for user_id in unconverged_users))
    with open("unconverged_users.json", "w") as f:
        json.dump(unconverged_users, f)
