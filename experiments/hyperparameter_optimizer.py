import argparse
import json
import os
import sys
import warnings
import re
from pathlib import Path

import numpy as np
import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.config import (  # noqa: E402
    DEFAULT_LEARN_COSTS,
    DEFAULT_FIRST_RATING_OFFSETS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_FIRST_SESSION_LENS,
    DEFAULT_FORGET_RATING_OFFSET,
    DEFAULT_FORGET_SESSION_LEN,
    DEFAULT_REVIEW_COSTS,
    DEFAULT_REVIEW_RATING_PROB,
    DECK_SIZE,
    DR_BASELINE_PATH,
    LEARN_LIMIT_PER_DAY,
    LEARN_SPAN,
    MAX_STUDYING_TIME_PER_DAY,
    PARALLEL,
    CHECKPOINTS_DIR,
    R_MAX,
    R_MIN,
    REVIEW_LIMIT_PER_DAY,
    S_MAX,
    default_device,
)
from ssp_mmc_fsrs.core import next_interval_torch  # noqa: E402
from ssp_mmc_fsrs.io import (
    load_dr_baseline,
    save_dr_baseline,
    save_policy_configs,
)  # noqa: E402
from ssp_mmc_fsrs.policies import create_dr_policy  # noqa: E402
from ssp_mmc_fsrs.simulation import simulate  # noqa: E402
from ssp_mmc_fsrs.solver import SSPMMCSolver  # noqa: E402
from experiments.lib import (  # noqa: E402
    DEFAULT_BENCHMARK_RESULT,
    DEFAULT_BUTTON_USAGE,
    DelayedKeyboardInterrupt,
    checkpoint_output_dir,
    dr_baseline_path_for_user,
    load_button_usage_config,
    load_fsrs_weights,
    normalize_button_usage,
    policy_configs_path_for_user,
)

warnings.filterwarnings("ignore")


review_costs = DEFAULT_REVIEW_COSTS
first_rating_prob = DEFAULT_FIRST_RATING_PROB
review_rating_prob = DEFAULT_REVIEW_RATING_PROB
first_rating_offsets = DEFAULT_FIRST_RATING_OFFSETS
first_session_lens = DEFAULT_FIRST_SESSION_LENS
forget_rating_offset = DEFAULT_FORGET_RATING_OFFSET
forget_session_len = DEFAULT_FORGET_SESSION_LEN
learn_costs = DEFAULT_LEARN_COSTS

DEVICE = default_device()
_DR_BASELINE_CACHE = None
W = None
W_LIST = None
AGGREGATE_MODE = "mean"
DR_BASELINE_PATH_LOCAL = DR_BASELINE_PATH
POLICY_CONFIGS_PATH_LOCAL = None
checkpoint_filename = None
ax = None
completed_trials = 0
loaded_flag = False
# After this many trials, start proposing candidates manually.
MANUAL_CANDIDATE_START_TRIAL = 40
# Propose a new candidate every N trials.
MANUAL_CANDIDATE_INTERVAL = 10
# Early stop if hypervolume improvement stays below this tolerance.
HYPERVOLUME_TOLERANCE = 1e-3
# Stop after this many low-improvement checks in a row.
HYPERVOLUME_PATIENCE = 3
# Check hypervolume every N trials.
HYPERVOLUME_CHECK_INTERVAL = 5
# Reference point for hypervolume (must be worse than all points).
HYPERVOLUME_REF_POINT = (0.0, 0.0)
# Threshold to treat hypervolume as meaningfully better.
HYPERVOLUME_EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimizer and DR baseline generator."
    )
    parser.add_argument(
        "--dr-baseline-only",
        action="store_true",
        help="Only run DR simulations and write dr_baseline.json.",
    )
    parser.add_argument(
        "--regen-dr-baseline",
        action="store_true",
        help="Regenerate dr_baseline.json before running optimizer.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="User ID for selecting FSRS weights and saving outputs.",
    )
    parser.add_argument(
        "--user-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated user IDs or ranges (e.g. '1,2,5-10') for multi-user "
            "optimization. Overrides --user-id when provided."
        ),
    )
    parser.add_argument(
        "--user-ids-file",
        type=Path,
        default=None,
        help=(
            "Text file containing user IDs or ranges (one per line or comma-separated)."
        ),
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Aggregation method across users (default: mean).",
    )
    parser.add_argument(
        "--benchmark-result",
        type=Path,
        default=DEFAULT_BENCHMARK_RESULT,
        help="FSRS benchmark result JSONL to read user weights from.",
    )
    parser.add_argument(
        "--button-usage",
        type=Path,
        default=DEFAULT_BUTTON_USAGE,
        help="Button usage JSONL to read simulation costs/probabilities from.",
    )
    return parser.parse_args()


def _require_weights():
    if W is None and not W_LIST:
        raise RuntimeError("FSRS weights are not initialized.")


def _aggregate(values, mode):
    if not values:
        return 0.0
    if mode == "median":
        return float(np.median(values))
    return float(np.mean(values))


def _parse_user_ids(value):
    if value is None:
        return []
    tokens = []
    for raw in re.split(r"[,\s]+", value.strip()):
        token = raw.strip()
        if token:
            tokens.append(token)
    user_ids = []
    for token in tokens:
        try:
            if "-" in token:
                start_raw, end_raw = token.split("-", 1)
                if not start_raw or not end_raw:
                    raise ValueError("Empty start or end in range")
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    start, end = end, start
                user_ids.extend(range(start, end + 1))
            else:
                user_ids.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid user ID or range: '{token}'")
    return sorted(set(user_ids))


def _load_user_ids_from_file(path):
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"User IDs file not found: {path}")
    combined = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            combined.extend(_parse_user_ids(line))
    return sorted(set(combined))


def _user_ids_label(user_ids):
    if not user_ids:
        return "users"
    if user_ids == list(range(user_ids[0], user_ids[-1] + 1)):
        label = f"{user_ids[0]}-{user_ids[-1]}"
    elif len(user_ids) <= 10:
        label = "_".join(str(uid) for uid in user_ids)
    else:
        label = f"{user_ids[0]}-{user_ids[-1]}_n{len(user_ids)}"
    return f"multi_users_{label}"


def _aggregate_usage(usages, mode):
    if not usages:
        return normalize_button_usage(None)
    if mode == "median":
        reducer = np.median
    else:
        reducer = np.mean

    aggregated = {}
    array_keys = [
        "learn_costs",
        "review_costs",
        "first_rating_prob",
        "review_rating_prob",
        "first_rating_offsets",
        "first_session_lens",
    ]
    scalar_keys = ["forget_rating_offset", "forget_session_len"]

    for key in array_keys:
        stacked = np.stack([usage[key] for usage in usages])
        aggregated[key] = reducer(stacked, axis=0)

    for key in scalar_keys:
        aggregated[key] = float(reducer([usage[key] for usage in usages]))

    return normalize_button_usage(aggregated)


def simulate_policy(policy, weights):
    _require_weights()
    (
        review_cnt_per_day,
        _,
        memorized_cnt_per_day,
        cost_per_day,
    ) = simulate(
        parallel=PARALLEL,
        w=weights,
        policy=policy,
        device=DEVICE,
        deck_size=DECK_SIZE,
        learn_span=LEARN_SPAN,
        learn_limit_perday=LEARN_LIMIT_PER_DAY,
        review_limit_perday=REVIEW_LIMIT_PER_DAY,
        max_cost_perday=MAX_STUDYING_TIME_PER_DAY,
        learn_costs=learn_costs,
        review_costs=review_costs,
        first_rating_prob=first_rating_prob,
        review_rating_prob=review_rating_prob,
        first_rating_offset=first_rating_offsets,
        first_session_len=first_session_lens,
        forget_rating_offset=forget_rating_offset,
        forget_session_len=forget_session_len,
        s_max=S_MAX,
    )

    return review_cnt_per_day, cost_per_day, memorized_cnt_per_day


def generate_dr_baseline():
    _require_weights()
    dr_baseline = []
    r_range = np.arange(R_MIN, R_MAX, 0.01)
    for r in r_range:

        def _compute_dr_metrics(weights):
            dr_policy = create_dr_policy(r, w=weights)
            _, cost_per_day, memorized_cnt_per_day = simulate_policy(dr_policy, weights)
            accum_cost = np.cumsum(cost_per_day, axis=-1)
            accum_time_average = accum_cost.mean() / 3600
            memorized_average = memorized_cnt_per_day.mean()
            if accum_time_average <= 0:
                avg_accum_memorized_per_hour = 0.0
            else:
                avg_accum_memorized_per_hour = memorized_average / accum_time_average
            return memorized_average, accum_time_average, avg_accum_memorized_per_hour

        if W_LIST:
            knowledge_values = []
            efficiency_values = []
            time_values = []
            for weights in W_LIST:
                (
                    memorized_average,
                    accum_time_average,
                    avg_accum_memorized_per_hour,
                ) = _compute_dr_metrics(weights)
                knowledge_values.append(memorized_average)
                time_values.append(accum_time_average)
                efficiency_values.append(avg_accum_memorized_per_hour)

            memorized_average = _aggregate(knowledge_values, AGGREGATE_MODE)
            accum_time_average = _aggregate(time_values, AGGREGATE_MODE)
            if accum_time_average <= 0:
                avg_accum_memorized_per_hour = 0.0
            else:
                avg_accum_memorized_per_hour = memorized_average / accum_time_average
        else:
            (
                memorized_average,
                accum_time_average,
                avg_accum_memorized_per_hour,
            ) = _compute_dr_metrics(W)
        dr_baseline.append(
            {
                "dr": float(r),
                "average_knowledge": float(memorized_average),
                "average_knowledge_per_hour": float(avg_accum_memorized_per_hour),
            }
        )
    save_dr_baseline(dr_baseline, DR_BASELINE_PATH_LOCAL)
    print(f"Saved DR baseline to {DR_BASELINE_PATH_LOCAL}")
    return dr_baseline


def get_dr_baseline(force=False):
    global _DR_BASELINE_CACHE
    if _DR_BASELINE_CACHE is not None and not force:
        return _DR_BASELINE_CACHE
    if force:
        dr_baseline = generate_dr_baseline()
    else:
        try:
            dr_baseline = load_dr_baseline(DR_BASELINE_PATH_LOCAL)
        except FileNotFoundError:
            dr_baseline = generate_dr_baseline()
    _DR_BASELINE_CACHE = dr_baseline
    return dr_baseline


def multi_objective_function(param_dict):
    _require_weights()

    def _evaluate_candidate(weights):
        solver = SSPMMCSolver(
            review_costs=review_costs,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            first_rating_offsets=first_rating_offsets,
            first_session_lens=first_session_lens,
            forget_rating_offset=forget_rating_offset,
            forget_session_len=forget_session_len,
            w=weights,
        )

        cost_matrix, retention_matrix = solver.solve(param_dict)
        retention_matrix_tensor = torch.tensor(retention_matrix, device=DEVICE)

        def ssp_mmc_policy(s, d):
            d_index = solver.d2i_torch(d)
            s_index = solver.s2i_torch(s)
            mask = (d_index >= solver.d_size) | (s_index >= solver.s_size - 1)
            optimal_interval = torch.zeros_like(s)
            optimal_interval[~mask] = next_interval_torch(
                s[~mask],
                retention_matrix_tensor[d_index[~mask], s_index[~mask]],
                -weights[20],
            )
            optimal_interval[mask] = np.inf
            return optimal_interval

        review_cnt_per_day, cost_per_day, memorized_cnt_per_day = simulate_policy(
            ssp_mmc_policy, weights
        )

        accum_cost = np.cumsum(cost_per_day, axis=-1)
        accum_time_average = accum_cost.mean() / 3600
        memorized_average = memorized_cnt_per_day.mean()
        avg_accum_memorized_per_hour = memorized_average / accum_time_average
        return memorized_average, avg_accum_memorized_per_hour

    if W_LIST:
        knowledge_values = []
        efficiency_values = []
        for weights in W_LIST:
            memorized_average, avg_accum_memorized_per_hour = _evaluate_candidate(
                weights
            )
            knowledge_values.append(memorized_average)
            efficiency_values.append(avg_accum_memorized_per_hour)

        memorized_average = _aggregate(knowledge_values, AGGREGATE_MODE)
        avg_accum_memorized_per_hour = _aggregate(efficiency_values, AGGREGATE_MODE)
        print("")
        print(param_dict)
        print(
            f"Aggregated ({AGGREGATE_MODE}) memorized={memorized_average:.0f} cards "
            f"across {len(W_LIST)} users"
        )
        print(
            f"Aggregated memorized/hours={avg_accum_memorized_per_hour:.1f} cards/hour"
        )
        print("")
    else:
        memorized_average, avg_accum_memorized_per_hour = _evaluate_candidate(W)
        print("")
        print(param_dict)
        print(f"Average memorized={memorized_average:.0f} cards")
        print(f"Average memorized/hours={avg_accum_memorized_per_hour:.1f} cards/hour")
        print("")

    return {
        "average_knowledge": (memorized_average, None),
        "average_knowledge_per_hour": (avg_accum_memorized_per_hour, None),
    }


total_trials = 500
ax_seed = S_MAX

parameters = [
    {"name": "a0", "type": "choice", "values": ["no_log", "log"]},
    {
        "name": "a1",
        "type": "range",
        "bounds": [0.1, 10],
        "log_scale": True,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a2",
        "type": "range",
        "bounds": [0.1, 10],
        "log_scale": True,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a3",
        "type": "range",
        "bounds": [-5, 5],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a4",
        "type": "range",
        "bounds": [-5, 5],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a5",
        "type": "range",
        "bounds": [-5, 5],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a6",
        "type": "range",
        "bounds": [-5, 5],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a7",
        "type": "range",
        "bounds": [-5, 5],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a8",
        "type": "range",
        "bounds": [-5, 5],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "a9",
        "type": "range",
        "bounds": [0, 3],
        "log_scale": False,
        "value_type": "float",
        "digits": 2,
    },
]

objectives = {
    "average_knowledge": ObjectiveProperties(minimize=False),
    "average_knowledge_per_hour": ObjectiveProperties(minimize=False),
}


def pareto(frontier, calc_knee=False):
    print("")
    print("Pareto optimal points:")
    twod_list = []
    for number, dictionary in list(frontier.items()):
        params = dictionary[0]
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = (
            params["a0"],
            params["a1"],
            params["a2"],
            params["a3"],
            params["a4"],
            params["a5"],
            params["a6"],
            params["a7"],
            params["a8"],
            params["a9"],
        )
        average_knowledge, average_knowledge_per_hour = (
            dictionary[1][0]["average_knowledge"],
            dictionary[1][0]["average_knowledge_per_hour"],
        )

        twod_list.append(
            [
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9,
                average_knowledge,
                average_knowledge_per_hour,
            ]
        )

    x = []
    y = []
    hyperparams = []

    twod_list = sorted(twod_list, key=lambda x: x[-1])
    for minilist in twod_list:
        (
            a0,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            average_knowledge,
            average_knowledge_per_hour,
        ) = minilist
        param_dict = {
            "a0": a0,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "a5": a5,
            "a6": a6,
            "a7": a7,
            "a8": a8,
            "a9": a9,
        }
        print(
            f"    parameters={param_dict}, objectives=({average_knowledge:.0f}, {average_knowledge_per_hour:.1f})"
        )

        if calc_knee:
            x.append(average_knowledge)
            y.append(average_knowledge_per_hour)
            hyperparams.append(param_dict)

    if len(x) > 2 and calc_knee:
        norm_x = [(x_i - min(x)) / (max(x) - min(x)) for x_i in x]
        norm_y = [(y_i - min(y)) / (max(y) - min(y)) for y_i in y]
        assert max(norm_x) == 1, f"{max(norm_x)}"
        assert min(norm_x) == 0, f"{min(norm_x)}"
        assert max(norm_y) == 1, f"{max(norm_y)}"
        assert min(norm_y) == 0, f"{min(norm_y)}"

        distances = []
        for norm_x_i, norm_y_i in zip(norm_x, norm_y):
            distance_i = np.sqrt(np.power(norm_x_i - 1, 2) + np.power(norm_y_i - 1, 2))
            distances.append(distance_i)

        knee_index = distances.index(min(distances))
        print("Knee point:")
        print(
            f"    parameters={hyperparams[knee_index]}, objectives=({x[knee_index]:.0f}, "
            f"{y[knee_index]:.2f})"
        )

    print("")


def _dr_baseline_points():
    dr_baseline = get_dr_baseline()
    return [
        [
            entry["dr"],
            entry["average_knowledge"],
            entry["average_knowledge_per_hour"],
        ]
        for entry in dr_baseline
    ]


def _extract_ssp_mmc_points(frontier):
    twod_list_ssp_mmc = []
    for _, dictionary in list(frontier.items()):
        params = dictionary[0]
        average_knowledge, average_knowledge_per_hour = (
            dictionary[1][0]["average_knowledge"],
            dictionary[1][0]["average_knowledge_per_hour"],
        )
        twod_list_ssp_mmc.append(
            [
                {
                    "a0": params["a0"],
                    "a1": params["a1"],
                    "a2": params["a2"],
                    "a3": params["a3"],
                    "a4": params["a4"],
                    "a5": params["a5"],
                    "a6": params["a6"],
                    "a7": params["a7"],
                    "a8": params["a8"],
                    "a9": params["a9"],
                },
                average_knowledge,
                average_knowledge_per_hour,
            ]
        )
    return sorted(twod_list_ssp_mmc, key=lambda x: x[1])


def _calculate_advantage(twod_list_dr, twod_list_ssp_mmc):
    dr_differences = []
    dr_pairs = []
    crappy_ssp_mmc_indices = []
    for index, ssp_mmc_list in enumerate(twod_list_ssp_mmc):
        knowledge_differences = []
        efficiency_differences = []
        for dr_list in twod_list_dr:
            knowledge_diff = abs(ssp_mmc_list[1] - dr_list[1])
            efficiency_diff = abs(ssp_mmc_list[2] - dr_list[2])
            knowledge_differences.append(knowledge_diff)
            efficiency_differences.append(efficiency_diff)

            if ssp_mmc_list[1] < dr_list[1] and ssp_mmc_list[2] < dr_list[2]:
                if index not in crappy_ssp_mmc_indices:
                    crappy_ssp_mmc_indices.append(index)

        closest_knowledge_dr_index = knowledge_differences.index(
            min(knowledge_differences)
        )
        closest_efficiency_dr_index = efficiency_differences.index(
            min(efficiency_differences)
        )
        closest_knowledge_dr = twod_list_dr[closest_knowledge_dr_index][0]
        closest_efficiency_dr = twod_list_dr[closest_efficiency_dr_index][0]
        dr_differences.append(closest_knowledge_dr - closest_efficiency_dr)
        dr_pairs.append([closest_knowledge_dr, closest_efficiency_dr])

    max_diff_params = None
    max_diff_drs = None
    if dr_differences and max(dr_differences) > 0:
        max_diff_index = dr_differences.index(max(dr_differences))
        max_diff_params = twod_list_ssp_mmc[max_diff_index]
        max_diff_drs = dr_pairs[max_diff_index]

    return {
        "max_diff_params": max_diff_params,
        "max_diff_drs": max_diff_drs,
        "crappy_ssp_mmc_indices": crappy_ssp_mmc_indices,
    }


def _generate_policy_configs_from_frontier(
    twod_list_dr, twod_list_ssp_mmc, max_diff_params
):
    policy_configs = []
    for dr_list in twod_list_dr:
        abs_knowledge_differences = []
        for ssp_mmc_list in twod_list_ssp_mmc:
            abs_knowledge_differences.append(abs(ssp_mmc_list[1] - dr_list[1]))

        closest_index = abs_knowledge_differences.index(min(abs_knowledge_differences))
        closest_params = twod_list_ssp_mmc[closest_index][0]

        if max_diff_params is not None and closest_params == max_diff_params[0]:
            entry = {"params": closest_params, "label": "Balanced"}
        else:
            entry = {"params": closest_params, "label": None}

        if entry not in policy_configs:
            policy_configs.append(entry)

    policy_configs.reverse()
    if policy_configs:
        policy_configs[0]["label"] = "Maximum knowledge"
        policy_configs[-1]["label"] = "Maximum efficiency"
    return policy_configs


def _propose_new_candidate(twod_list_ssp_mmc, crappy_ssp_mmc_indices):
    if len(crappy_ssp_mmc_indices) == 0:
        print("No need to manually propose a new candidate")
        return None
    worse_candidate = twod_list_ssp_mmc[min(crappy_ssp_mmc_indices)][0]
    better_candidate = twod_list_ssp_mmc[min(crappy_ssp_mmc_indices) - 1][0]
    all_keys = better_candidate.keys()

    if better_candidate.get("a0") == worse_candidate.get("a0"):
        strategy = "average"
    else:
        strategy = "mutate"

    new_candidate = {}
    for key in all_keys:
        # To prevent new candidates from being the same
        # Better make it deterministic for reproducibility, so I made this weird thingy
        seed = (
            50
            * twod_list_ssp_mmc[-1][-1]
            * sum(abs(x) for x in list(better_candidate.values())[1:])
            * sum(abs(x) for x in list(worse_candidate.values())[1:])
            / len(twod_list_ssp_mmc)
        )
        np.random.seed(int(seed))
        if key == "a0":
            new_candidate.update({"a0": better_candidate.get(key)})
        else:
            if strategy == "average":
                better_param = better_candidate.get(key)
                worse_param = worse_candidate.get(key)

                random_weight_better = float(np.random.uniform(1.5, 4))
                random_weight_worse = float(np.random.uniform(0.7, 1))

                w_avg_param = (
                    random_weight_better * better_param
                    + random_weight_worse * worse_param
                ) / (random_weight_better + random_weight_worse)
                new_candidate.update({key: round(w_avg_param, 2)})
            elif strategy == "mutate":
                better_param = better_candidate.get(key)

                mutation = float(np.random.normal(0, 0.1))

                if key in ["a1", "a2"]:
                    new_param = max(
                        min(round(better_param * (1 + mutation), 2), 10.0), 0.1
                    )
                elif key == "a9":
                    new_param = max(
                        min(round(better_param * (1 + mutation), 2), 3.0), 0.0
                    )
                else:
                    new_param = max(
                        min(round(better_param * (1 + mutation), 2), 5.0), -5.0
                    )
                new_candidate.update({key: new_param})
            else:
                raise Exception("Unknown candidate generation strategy")

    print(f"Manually proposed new candidate: {new_candidate}")
    return new_candidate


def _frontier_points(frontier):
    points = []
    for _, payload in frontier.items():
        metrics = payload[1][0]
        points.append(
            (
                float(metrics["average_knowledge"]),
                float(metrics["average_knowledge_per_hour"]),
            )
        )
    return points


def _hypervolume_2d(points, ref_point):
    ref_x, ref_y = ref_point
    filtered = [(x, y) for x, y in points if x > ref_x and y > ref_y]
    if not filtered:
        return 0.0
    filtered.sort(key=lambda point: point[0], reverse=True)
    max_y = ref_y
    area = 0.0
    for idx, (x, y) in enumerate(filtered):
        if y > max_y:
            max_y = y
        x_next = filtered[idx + 1][0] if idx + 1 < len(filtered) else ref_x
        width = x - x_next
        if width > 0:
            area += width * (max_y - ref_y)
    return area


def advantage_maximizer(frontier, propose_candidate=False, print_for_script=False):
    twod_list_dr = _dr_baseline_points()
    twod_list_ssp_mmc = _extract_ssp_mmc_points(frontier)
    advantage = _calculate_advantage(twod_list_dr, twod_list_ssp_mmc)
    max_diff_params = advantage["max_diff_params"]
    max_diff_drs = advantage["max_diff_drs"]

    if max_diff_params is not None and not propose_candidate:
        print(
            f"    Hyperparameters that provide the biggest advantage={max_diff_params[0]}"
        )
        print(f"    You get the average knowledge of DR={100 * max_diff_drs[0]:.0f}%")
        print(f"    You get the efficiency of DR={100 * max_diff_drs[1]:.0f}%")
        print("")

    if print_for_script:
        policy_configs = _generate_policy_configs_from_frontier(
            twod_list_dr, twod_list_ssp_mmc, max_diff_params
        )
        print(json.dumps(policy_configs, indent=2, sort_keys=True))
        return policy_configs

    if propose_candidate:
        return _propose_new_candidate(
            twod_list_ssp_mmc, advantage["crappy_ssp_mmc_indices"]
        )

    return None


def run_optimizer():
    global completed_trials
    if checkpoint_filename is None:
        raise RuntimeError("Checkpoint filename is not initialized.")
    if POLICY_CONFIGS_PATH_LOCAL is None:
        raise RuntimeError("Policy configs path is not initialized.")

    printed_flag = False
    stable_hypervolume_checks = 0
    best_hypervolume = None
    best_frontier = None
    if completed_trials < total_trials:
        for i in range(completed_trials, total_trials):
            if loaded_flag and not printed_flag:
                frontier = ax.get_pareto_optimal_parameters()
                pareto(frontier)
                advantage_maximizer(frontier)
                best_hypervolume = _hypervolume_2d(
                    _frontier_points(frontier),
                    HYPERVOLUME_REF_POINT,
                )
                best_frontier = frontier
                printed_flag = True
            elif i > 0 and i % HYPERVOLUME_CHECK_INTERVAL == 0:
                frontier = ax.get_pareto_optimal_parameters()
                pareto(frontier)
                advantage_maximizer(frontier)
                hypervolume = _hypervolume_2d(
                    _frontier_points(frontier),
                    HYPERVOLUME_REF_POINT,
                )
                if best_hypervolume is None:
                    best_hypervolume = hypervolume
                    best_frontier = frontier
                    improvement = 0.0
                else:
                    improvement = hypervolume - best_hypervolume
                print(
                    "Hypervolume improvement: "
                    f"{improvement:.6f} (current={hypervolume:.6f}, best={best_hypervolume:.6f})"
                )
                if improvement > HYPERVOLUME_EPS:
                    best_hypervolume = hypervolume
                    best_frontier = frontier
                if improvement < HYPERVOLUME_TOLERANCE:
                    stable_hypervolume_checks += 1
                else:
                    stable_hypervolume_checks = 0
                if stable_hypervolume_checks >= HYPERVOLUME_PATIENCE:
                    print(
                        "Hypervolume improvement below tolerance. "
                        "Stopping early to avoid redundant trials."
                    )
                    break

            print(f"Starting trial {i}/{total_trials}")
            if i >= MANUAL_CANDIDATE_START_TRIAL and i % MANUAL_CANDIDATE_INTERVAL == 0:
                frontier = ax.get_pareto_optimal_parameters()
                parameters = advantage_maximizer(frontier, propose_candidate=True)
                if parameters is not None:
                    trial_indices = ax.attach_trial(parameters=parameters)
                    trial_index = trial_indices[1]
                else:
                    parameters, trial_index = ax.get_next_trial()
            else:
                parameters, trial_index = ax.get_next_trial()

            torch.cuda.empty_cache()
            ax.complete_trial(
                trial_index=trial_index, raw_data=multi_objective_function(parameters)
            )

            with DelayedKeyboardInterrupt():
                ax.save_to_json_file(checkpoint_filename)

    frontier = best_frontier or ax.get_pareto_optimal_parameters()
    pareto(frontier)
    policy_configs = advantage_maximizer(frontier, print_for_script=True)
    if policy_configs:
        save_policy_configs(policy_configs, POLICY_CONFIGS_PATH_LOCAL)
        print(f"Saved policy configs to {POLICY_CONFIGS_PATH_LOCAL}")


def _init_ax(checkpoint_dir):
    global checkpoint_filename, ax, completed_trials, loaded_flag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = checkpoint_dir / (
        f"SSP-MMC_Smax={ax_seed}_params={len(parameters)}_avg.json"
    )

    if os.path.isfile(checkpoint_filename):
        try:
            with open(checkpoint_filename, "r") as f:
                content = f.read()
                try:
                    json_content = json.loads(content)
                    if "experiment" not in json_content:
                        print("WARNING: File doesn't contain 'experiment' key")
                except json.JSONDecodeError:
                    print("ERROR: File does not contain valid JSON")
        except Exception as exc:
            print(f"Error reading file: {exc}")
    else:
        print("Checkpoint file does not exist")

    loaded_flag = False
    if os.path.isfile(checkpoint_filename):
        try:
            print(f"Loading checkpoint from {checkpoint_filename}")
            with DelayedKeyboardInterrupt():
                ax = AxClient.load_from_json_file(checkpoint_filename)
            ax._random_seed = ax_seed
            assert ax._random_seed == ax_seed
            completed_trials = len(ax.experiment.trials)
            loaded_flag = completed_trials > 0
            print(
                f"Successfully loaded experiment with {completed_trials} completed trials"
            )
        except Exception as exc:
            print(f"Error loading checkpoint: {exc}")
            raise SystemExit(1) from exc
    else:
        ax = AxClient(random_seed=ax_seed, verbose_logging=False)
        ax.create_experiment(
            name="SSP-MMC, Bayesian search",
            parameters=parameters,
            objectives=objectives,
        )
        completed_trials = 0
        ax.save_to_json_file(checkpoint_filename)


def main():
    args = parse_args()
    global \
        W, \
        W_LIST, \
        DR_BASELINE_PATH_LOCAL, \
        POLICY_CONFIGS_PATH_LOCAL, \
        _DR_BASELINE_CACHE
    global AGGREGATE_MODE
    global review_costs, first_rating_prob, review_rating_prob
    global first_rating_offsets, first_session_lens
    global forget_rating_offset, forget_session_len, learn_costs

    AGGREGATE_MODE = args.aggregate

    user_ids = []
    user_ids.extend(_parse_user_ids(args.user_ids))
    user_ids.extend(_load_user_ids_from_file(args.user_ids_file))
    user_ids = sorted(set(user_ids))

    if user_ids:
        W = None
        W_LIST = [
            load_fsrs_weights(args.benchmark_result, user_id)[0] for user_id in user_ids
        ]
        usage_list = [
            normalize_button_usage(load_button_usage_config(args.button_usage, user_id))
            for user_id in user_ids
        ]
        usage = _aggregate_usage(usage_list, AGGREGATE_MODE)
        label = _user_ids_label(user_ids)
        checkpoint_dir = CHECKPOINTS_DIR / label
        DR_BASELINE_PATH_LOCAL = checkpoint_dir / "dr_baseline.json"
        POLICY_CONFIGS_PATH_LOCAL = checkpoint_dir / "policy_configs.json"
        print(f"Running multi-user optimization for {len(user_ids)} users: {user_ids}")
        print(f"Aggregate mode: {AGGREGATE_MODE}")
    else:
        W_LIST = None
        W, _, _ = load_fsrs_weights(args.benchmark_result, args.user_id)
        usage = normalize_button_usage(
            load_button_usage_config(args.button_usage, args.user_id)
        )
        DR_BASELINE_PATH_LOCAL = dr_baseline_path_for_user(args.user_id)
        POLICY_CONFIGS_PATH_LOCAL = policy_configs_path_for_user(args.user_id)
        checkpoint_dir = checkpoint_output_dir(args.user_id)

    learn_costs = usage["learn_costs"]
    review_costs = usage["review_costs"]
    first_rating_prob = usage["first_rating_prob"]
    review_rating_prob = usage["review_rating_prob"]
    first_rating_offsets = usage["first_rating_offsets"]
    first_session_lens = usage["first_session_lens"]
    forget_rating_offset = usage["forget_rating_offset"]
    forget_session_len = usage["forget_session_len"]
    _DR_BASELINE_CACHE = None
    _init_ax(checkpoint_dir)
    if args.dr_baseline_only:
        generate_dr_baseline()
        return
    if args.regen_dr_baseline:
        get_dr_baseline(force=True)
    run_optimizer()


if __name__ == "__main__":
    main()
