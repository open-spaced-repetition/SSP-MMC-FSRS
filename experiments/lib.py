from pathlib import Path
import json
import logging
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    from colorama import Fore, Style
except ModuleNotFoundError:
    class _NoColor:
        RED = ""
        RESET_ALL = ""

    Fore = _NoColor()
    Style = _NoColor()

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssp_mmc_fsrs.config import (  # noqa: E402
    DEFAULT_FIRST_RATING_OFFSETS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_FIRST_SESSION_LENS,
    DEFAULT_FORGET_RATING_OFFSET,
    DEFAULT_FORGET_SESSION_LEN,
    DEFAULT_REVIEW_COSTS,
    DEFAULT_REVIEW_RATING_PROB,
    DECK_SIZE,
    LEARN_SPAN,
    PARALLEL,
    PLOTS_DIR,
    POLICIES_DIR,
    R_MAX,
    R_MIN,
    REVIEW_LIMIT_PER_DAY,
    S_MAX,
    SIMULATION_DIR,
    SIMULATION_RESULTS_PATH,
    resolve_simulation_limits,
)
from ssp_mmc_fsrs.core import next_interval  # noqa: E402
from ssp_mmc_fsrs.io import (  # noqa: E402
    load_policy,
    load_simulation_results,
    save_dr_baseline,
    save_policy,
    save_simulation_results,
)
from ssp_mmc_fsrs.policies import (  # noqa: E402
    anki_sm2_policy,
    create_dr_policy,
    create_fixed_interval_policy,
    memrise_policy,
)
from ssp_mmc_fsrs.simulation import simulate  # noqa: E402
from ssp_mmc_fsrs.solver import SSPMMCSolver  # noqa: E402


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt")
        print(f"{Fore.RED}Delaying KeyboardInterrupt{Style.RESET_ALL}")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def setup_environment(seed):
    plt.style.use("ggplot")
    np.random.seed(seed)


def ensure_output_dirs():
    for path in (PLOTS_DIR, SIMULATION_DIR, POLICIES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_policy_list(policies):
    if not policies:
        raise ValueError("At least one policy must be specified.")

    normalized = set()
    raw = [policy.strip().lower().replace("_", "-") for policy in policies]
    if "all" in raw:
        return {"ssp-mmc", "memrise", "anki-sm-2", "dr", "interval"}

    for policy in raw:
        if policy in {"ssp-mmc", "sspmmc", "ssp-mmc-fsrs"}:
            normalized.add("ssp-mmc")
        elif policy in {"memrise"}:
            normalized.add("memrise")
        elif policy in {"anki-sm-2", "anki-sm2", "anki"}:
            normalized.add("anki-sm-2")
        elif policy in {"dr", "desired-retention", "fixed-dr"}:
            normalized.add("dr")
        elif policy in {"interval", "fixed-interval", "fixed-intervals"}:
            normalized.add("interval")
        else:
            raise ValueError(f"Unknown policy: {policy}")

    return normalized


def simulate_policy_factory(w, device, learn_limit_per_day, max_studying_time_per_day):
    def simulate_policy(policy):
        (
            review_cnt_per_day,
            _,
            memorized_cnt_per_day,
            cost_per_day,
        ) = simulate(
            parallel=PARALLEL,
            w=w,
            policy=policy,
            device=device,
            deck_size=DECK_SIZE,
            learn_span=LEARN_SPAN,
            learn_limit_perday=learn_limit_per_day,
            review_limit_perday=REVIEW_LIMIT_PER_DAY,
            max_cost_perday=max_studying_time_per_day,
            s_max=S_MAX,
        )

        return review_cnt_per_day, cost_per_day, memorized_cnt_per_day

    return simulate_policy


def compute_simulation_metrics(review_cnt_per_day, cost_per_day, memorized_cnt_per_day):
    reviews_average = review_cnt_per_day.mean()
    time_average = cost_per_day.mean() / 60
    accum_cost = np.cumsum(cost_per_day, axis=-1)
    accum_time_average = accum_cost.mean() / 3600
    memorized_average = memorized_cnt_per_day.mean()
    avg_accum_memorized_per_hour = memorized_average / accum_time_average

    assert not isinstance(reviews_average, np.ndarray), f"{reviews_average}"
    assert not isinstance(time_average, np.ndarray), f"{time_average}"
    assert not isinstance(memorized_average, np.ndarray), f"{memorized_average}"
    assert not isinstance(avg_accum_memorized_per_hour, np.ndarray), (
        f"{avg_accum_memorized_per_hour}"
    )

    return (
        reviews_average,
        time_average,
        memorized_average,
        avg_accum_memorized_per_hour,
    )


def update_simulation_results(result, results_path):
    try:
        results = load_simulation_results(results_path)
    except FileNotFoundError:
        results = []
    results = [entry for entry in results if entry["title"] != result["title"]]
    results.append(result)
    save_simulation_results(results, results_path)


def plot_simulation(policy, title, results_path, simulate_policy):
    review_cnt_per_day, cost_per_day, memorized_cnt_per_day = simulate_policy(policy)

    metrics = compute_simulation_metrics(
        review_cnt_per_day, cost_per_day, memorized_cnt_per_day
    )
    update_simulation_results(
        {
            "title": title,
            "reviews_average": float(metrics[0]),
            "time_average": float(metrics[1]),
            "memorized_average": float(metrics[2]),
            "avg_accum_memorized_per_hour": float(metrics[3]),
        },
        results_path,
    )

    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_subplot(131)
    ax.plot(review_cnt_per_day[0])
    ax.set_title("Review Count")
    ax = fig.add_subplot(132)
    ax.plot(cost_per_day[0], label=f"Total Cost: {cost_per_day[0].sum():.2f}")
    ax.set_title("Cost")
    ax.legend()
    ax = fig.add_subplot(133)
    ax.plot(
        memorized_cnt_per_day[0],
        label=f"Total Memorized: {memorized_cnt_per_day[0][-1]:.2f}",
    )
    ax.set_title("Memorized Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SIMULATION_DIR / f"{title}.png")
    plt.close()


def plot_policy_surfaces(solver, cost_matrix, retention_matrix, avg_cost, avg_retention, w):
    s_state_mesh_2d, d_state_mesh_2d = np.meshgrid(solver.s_state, solver.d_state)
    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, cost_matrix, cmap="viridis")
    ax.set_xlabel("Stability")
    ax.set_ylabel("Difficulty")
    ax.set_zlabel("Cost")
    ax.set_title(f"Avg Init Cost: {avg_cost:.2f}")
    ax.set_box_aspect(None, zoom=0.8)

    ax = fig.add_subplot(132, projection="3d")
    ax.plot_surface(
        s_state_mesh_2d, d_state_mesh_2d, retention_matrix, cmap="viridis"
    )
    ax.set_xlabel("Stability")
    ax.set_ylabel("Difficulty")
    ax.set_zlabel("Retention")
    ax.set_title(f"Avg Retention: {avg_retention:.2f}")
    ax.set_box_aspect(None, zoom=0.8)

    ax = fig.add_subplot(133, projection="3d")
    interval_matrix = next_interval(s_state_mesh_2d, retention_matrix, -w[20])
    ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, interval_matrix, cmap="viridis")
    ax.set_xlabel("Stability")
    ax.set_ylabel("Difficulty")
    ax.set_zlabel("Interval")
    ax.set_title("Interval")
    ax.set_box_aspect(None, zoom=0.8)
    return fig


def ssp_mmc_title(policy_config):
    label = policy_config.get("label")
    return f"SSP-MMC-FSRS ({label})" if label else "SSP-MMC-FSRS"


def _jsonify_params(params):
    normalized = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def _policy_key(title, params):
    normalized = _jsonify_params(params)
    normalized_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return (title, normalized_json)


def _load_policy_index(policy_dir):
    policy_dir = Path(policy_dir)
    index = {}
    for meta_path in policy_dir.glob("*.json"):
        try:
            with meta_path.open("r") as f:
                meta = json.load(f)
        except json.JSONDecodeError:
            continue
        title = meta.get("title")
        hyperparams = meta.get("hyperparams")
        if not title or hyperparams is None:
            continue
        index[_policy_key(title, hyperparams)] = meta_path
    return index


def run_ssp_mmc_configs(policy_configs, results_path, simulate_policy, device):
    last_solver = None
    last_retention_matrix = None
    last_init_stabilities = None
    last_init_difficulties = None

    policy_index = _load_policy_index(POLICIES_DIR)
    if not policy_index:
        raise FileNotFoundError(
            f"No SSP-MMC policies found in {POLICIES_DIR}. "
            "Run experiments/generate_ssp_mmc_policies.py first."
        )

    for policy_config in policy_configs:
        title = ssp_mmc_title(policy_config)
        policy_key = _policy_key(title, policy_config["params"])
        meta_path = policy_index.get(policy_key)
        if meta_path is None:
            raise FileNotFoundError(
                f"Missing SSP-MMC policy for {title}. "
                "Run experiments/generate_ssp_mmc_policies.py first."
            )
        policy_data = load_policy(meta_path, device=device)
        solver = policy_data["solver"]
        retention_matrix = policy_data["retention_matrix"]

        plot_simulation(policy_data["policy"], title, results_path, simulate_policy)

        last_solver = solver
        last_retention_matrix = retention_matrix
        last_init_stabilities = solver.init_s(np.arange(1, 5))
        last_init_difficulties = solver.init_d_with_short_term(np.arange(1, 5))

    return (
        last_solver,
        last_retention_matrix,
        last_init_stabilities,
        last_init_difficulties,
    )


def generate_ssp_mmc_policies(policy_configs, w):
    for policy_config in policy_configs:
        solver = SSPMMCSolver(
            review_costs=DEFAULT_REVIEW_COSTS,
            first_rating_prob=DEFAULT_FIRST_RATING_PROB,
            review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
            first_rating_offsets=DEFAULT_FIRST_RATING_OFFSETS,
            first_session_lens=DEFAULT_FIRST_SESSION_LENS,
            forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
            forget_session_len=DEFAULT_FORGET_SESSION_LEN,
            w=w,
        )

        cost_matrix, retention_matrix = solver.solve(policy_config["params"])
        init_stabilities = solver.init_s(np.arange(1, 5))
        init_difficulties = solver.init_d_with_short_term(np.arange(1, 5))
        init_cost = cost_matrix[
            solver.d2i(init_difficulties), solver.s2i(init_stabilities)
        ]
        avg_cost = init_cost @ DEFAULT_FIRST_RATING_PROB
        print(f"Average cost: {avg_cost:.2f}")
        avg_retention = retention_matrix.mean()
        print(f"Average retention: {avg_retention:.2f}")

        fig = plot_policy_surfaces(
            solver, cost_matrix, retention_matrix, avg_cost, avg_retention, w
        )

        title = ssp_mmc_title(policy_config)
        save_policy(
            POLICIES_DIR,
            title,
            solver,
            cost_matrix,
            retention_matrix,
            policy_config["params"],
        )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{title}.png")
        plt.close(fig)


def plot_optimal_policy_vs_stability(
    solver, retention_matrix, init_stabilities, init_difficulties, w
):
    def optimal_policy_for_rating_sequence(rating_sequence):
        s_list = []
        r_list = []
        ivl_list = []
        g_list = []
        d_index = None
        s_index = None
        cur_s = None
        cur_d = None
        for i, rating in enumerate(rating_sequence):
            g_list.append(rating)
            if i == 0:
                d_index, s_index = (
                    solver.d2i(init_difficulties[rating - 1]),
                    solver.s2i(init_stabilities[rating - 1]),
                )
                cur_s = solver.s_state[s_index]
                cur_d = solver.d_state[d_index]
            else:
                optimal_r = retention_matrix[d_index, s_index]
                s_list.append(cur_s)
                r_list.append(optimal_r)
                ivl_list.append(next_interval(cur_s, optimal_r, -w[20]))
                cur_s = solver.stability_after_success(cur_s, cur_d, optimal_r, rating)
                cur_d = solver.next_d(cur_d, rating)
                d_index, s_index = solver.d2i(cur_d), solver.s2i(cur_s)

            if cur_s > S_MAX:
                break

        return s_list, r_list, ivl_list, g_list

    def plot_sequence(rating_sequence):
        s_list, r_list, ivl_list, g_list = optimal_policy_for_rating_sequence(
            rating_sequence
        )
        fig = plt.figure(figsize=(16, 8.5))
        ax = fig.add_subplot(121)
        ax.plot(s_list, r_list, "*-")
        ax.set_xlabel("Stability")
        ax.set_ylabel("Optimal Retention")
        ax.set_title("Optimal Retention vs Stability")
        ax = fig.add_subplot(122)
        ax.plot(s_list, ivl_list, "*-", label="Optimal")
        ax.plot(s_list, s_list, "--", alpha=0.5, label="R=90%")
        for s, ivl in zip(s_list, ivl_list):
            ax.text(s + 1, ivl - 10, f"{ivl:.0f}", fontsize=10)
        ax.set_xlabel("Stability")
        ax.set_ylabel("Optimal Interval")
        ax.set_title("Optimal Interval vs Stability")
        ax.legend()
        fig.suptitle(f"Rating Sequence: {','.join(map(str, g_list))}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"OR-OI-{','.join(map(str, g_list))}.png")
        plt.close(fig)

    for rating in range(1, 5):
        plot_sequence([rating] + [3 for _ in range(100)])


def dr_range():
    return np.arange(R_MIN, R_MAX, 0.01)


def evaluate_dr_thresholds(w):
    costs = []
    r_values = dr_range()

    for r in r_values:
        print("--------------------------------")
        start = time.time()
        solver = SSPMMCSolver(
            review_costs=DEFAULT_REVIEW_COSTS,
            first_rating_prob=DEFAULT_FIRST_RATING_PROB,
            review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
            first_rating_offsets=DEFAULT_FIRST_RATING_OFFSETS,
            first_session_lens=DEFAULT_FIRST_SESSION_LENS,
            forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
            forget_session_len=DEFAULT_FORGET_SESSION_LEN,
            w=w,
        )
        solver._init_state_spaces()
        s_state_mesh_2d, d_state_mesh_2d = np.meshgrid(solver.s_state, solver.d_state)
        cost_matrix, r_state_mesh_2d = solver.evaluate_r_threshold(r)
        end = time.time()
        print(f"Time: {end - start:.2f}s")
        init_stabilities = solver.init_s(np.arange(1, 5))
        init_difficulties = solver.init_d_with_short_term(np.arange(1, 5))
        init_cost = cost_matrix[
            solver.d2i(init_difficulties), solver.s2i(init_stabilities)
        ]
        avg_cost = init_cost @ DEFAULT_FIRST_RATING_PROB
        avg_retention = r_state_mesh_2d.mean()
        print(f"Desired Retention: {r * 100:.2f}%")
        print(f"True Retention: {avg_retention * 100:.2f}%")
        print(f"Average Cost: {avg_cost}")
        costs.append(avg_cost)
        fig = plt.figure(figsize=(16, 8.5))
        ax = fig.add_subplot(121, projection="3d")
        ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, cost_matrix, cmap="viridis")
        ax.set_xlabel("Stability")
        ax.set_ylabel("Difficulty")
        ax.set_zlabel("Cost")
        ax.set_title(f"Desired Retention: {r * 100:.2f}%, Avg Cost: {avg_cost:.2f}")
        ax.set_box_aspect(None, zoom=0.8)
        ax = fig.add_subplot(122, projection="3d")
        ax.plot_surface(
            s_state_mesh_2d, d_state_mesh_2d, r_state_mesh_2d, cmap="viridis"
        )
        ax.set_xlabel("Stability")
        ax.set_ylabel("Difficulty")
        ax.set_zlabel("Retention")
        ax.set_title(f"True Retention: {avg_retention:.2f}")
        ax.set_box_aspect(None, zoom=0.8)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"DR={r:.2f}.png")
        plt.close(fig)

    return r_values, costs


def simulate_dr_policies(results_path, simulate_policy):
    for r in dr_range():
        dr_policy = create_dr_policy(r)
        plot_simulation(dr_policy, f"DR={r:.2f}", results_path, simulate_policy)


def plot_cost_vs_retention(costs, r_range):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    optimal_retention = r_range[np.argmin(costs)]
    min_cost = np.min(costs)
    ax.plot(r_range, costs)
    ax.set_xlabel("Desired Retention")
    ax.set_ylabel("Average Cost")
    ax.set_title(
        f"Optimal Retention: {optimal_retention * 100:.2f}%, Min Cost: {min_cost:.2f}"
    )
    plt.savefig(PLOTS_DIR / "cost_vs_retention.png")
    plt.close(fig)


def run_fixed_interval_policies(results_path, simulate_policy):
    for fixed_interval in [7, 14, 20, 30, 50, 75, 100]:
        fixed_policy = create_fixed_interval_policy(fixed_interval)
        plot_simulation(
            fixed_policy, f"Interval={fixed_interval}", results_path, simulate_policy
        )


def save_dr_baseline_from_results(results_path, path):
    simulation_results = load_simulation_results(results_path)
    dr_baseline = []
    for entry in simulation_results:
        title = entry["title"]
        memorized_average = entry["memorized_average"]
        avg_accum_memorized_per_hour = entry["avg_accum_memorized_per_hour"]
        if title.startswith("DR="):
            try:
                dr_value = float(title.split("=")[1])
            except ValueError:
                continue
            dr_baseline.append(
                {
                    "dr": dr_value,
                    "average_knowledge": float(memorized_average),
                    "average_knowledge_per_hour": float(avg_accum_memorized_per_hour),
                }
            )
    dr_baseline.sort(key=lambda entry: entry["dr"])
    save_dr_baseline(dr_baseline, path)
    print(f"Saved DR baseline to {path}")


def plot_pareto_frontier(results_path, policy_configs):
    simulation_results = load_simulation_results(results_path)
    if not simulation_results:
        print(f"No simulation results found at {results_path}. Skipping Pareto plot.")
        return
    ssp_mmc_titles = []
    ssp_mmc_x = []
    ssp_mmc_y = []

    fixed_dr_titles = []
    fixed_dr_x = []
    fixed_dr_y = []

    fixed_intervals_titles = []
    fixed_intervals_x = []
    fixed_intervals_y = []

    other_titles = []
    other_x = []
    other_y = []

    for entry in simulation_results:
        title = entry["title"]
        reviews_average = entry["reviews_average"]
        time_average = entry["time_average"]
        memorized_average = entry["memorized_average"]
        avg_accum_memorized_per_hour = entry["avg_accum_memorized_per_hour"]
        if "SSP-MMC" in title:
            ssp_mmc_titles.append(title)
            ssp_mmc_x.append(memorized_average)
            ssp_mmc_y.append(avg_accum_memorized_per_hour)
        elif "DR" in title:
            fixed_dr_titles.append(title)
            fixed_dr_x.append(memorized_average)
            fixed_dr_y.append(avg_accum_memorized_per_hour)
        elif "Interval" in title:
            fixed_intervals_titles.append(title)
            fixed_intervals_x.append(memorized_average)
            fixed_intervals_y.append(avg_accum_memorized_per_hour)
        else:
            other_titles.append(title)
            other_x.append(memorized_average)
            other_y.append(avg_accum_memorized_per_hour)

    assert len(ssp_mmc_x) == len(ssp_mmc_y), f"{len(ssp_mmc_x)}, {len(ssp_mmc_y)}"
    assert len(fixed_dr_x) == len(fixed_dr_y), f"{len(fixed_dr_x)}, {len(fixed_dr_y)}"
    assert len(fixed_intervals_x) == len(
        fixed_intervals_y
    ), f"{len(fixed_intervals_x)}, {len(fixed_intervals_y)}"
    assert len(other_x) == len(other_y), f"{len(other_x)}, {len(other_y)}"

    def border_aware_text(x_min, x_max, y_min, y_max, x, y, text, **kwargs):
        x_rel = (x - x_min) / (x_max - x_min) if x_max != x_min else 0.5

        margin_x = (x_max - x_min) * 0.02
        extra_margin = (x_max - x_min) * 0.008

        if x_rel < 0.1:
            ha = "left"
            text_x = max(x + extra_margin, x_min + margin_x + extra_margin)
        elif x_rel > 0.9:
            ha = "right"
            text_x = min(x - extra_margin, x_max - margin_x - extra_margin)
        else:
            ha = "left"
            text_x = x + extra_margin

        va = "center"
        text_y = y

        kwargs.setdefault("ha", ha)
        kwargs.setdefault("va", va)

        return plt.text(text_x, text_y, text, **kwargs)

    plt.figure(figsize=(12, 9))

    balanced_index = -1
    for policy_config in policy_configs:
        if policy_config.get("label") == "Balanced":
            balanced_index = policy_configs.index(policy_config)

    if ssp_mmc_x:
        plt.plot(
            ssp_mmc_x,
            ssp_mmc_y,
            label="SSP-MMC-FSRS",
            linewidth=2,
            color="#00b050",
            marker="o",
        )
        if balanced_index != -1:
            plt.plot(
                ssp_mmc_x[balanced_index],
                ssp_mmc_y[balanced_index],
                linewidth=2,
                color="#00b050",
                marker=(5, 1, 15),
                ms=20,
            )
    if fixed_dr_x:
        plt.plot(
            fixed_dr_x,
            fixed_dr_y,
            label="Fixed DR (FSRS)",
            linewidth=2,
            color="#5b9bd5",
            marker="s",
        )
    if other_x:
        plt.plot(
            other_x,
            other_y,
            label="Other scheduling policies",
            linewidth=2,
            linestyle="",
            color="red",
            marker="^",
            ms=7.5,
        )
    if fixed_intervals_x:
        plt.plot(
            fixed_intervals_x,
            fixed_intervals_y,
            label="Fixed intervals",
            linewidth=2,
            color="black",
            marker="x",
            ms=7.5,
        )

    x_min = 200 * np.floor(
        (min([entry["memorized_average"] for entry in simulation_results]) / 200)
    )
    x_max = 200 * np.ceil(
        (max([entry["memorized_average"] for entry in simulation_results]) / 200)
    )
    y_min = 0
    y_max = max(
        [entry["avg_accum_memorized_per_hour"] for entry in simulation_results]
    ) * 1.03

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

    if ssp_mmc_x and balanced_index != -1:
        border_aware_text(
            x_min,
            x_max,
            y_min,
            y_max,
            ssp_mmc_x[balanced_index] + 10,
            ssp_mmc_y[balanced_index],
            "Balanced",
            fontsize=11,
        )

    if fixed_dr_x:
        border_aware_text(
            x_min,
            x_max,
            y_min,
            y_max,
            fixed_dr_x[0],
            fixed_dr_y[0],
            f"{fixed_dr_titles[0]}",
            fontsize=11,
        )
        border_aware_text(
            x_min,
            x_max,
            y_min,
            y_max,
            fixed_dr_x[-1],
            fixed_dr_y[-1],
            f"{fixed_dr_titles[-1]}",
            fontsize=11,
        )

    for n in range(len(fixed_intervals_x)):
        border_aware_text(
            x_min,
            x_max,
            y_min,
            y_max,
            fixed_intervals_x[n],
            fixed_intervals_y[n],
            f"{fixed_intervals_titles[n]}",
            fontsize=11,
        )

    for n in range(len(other_x)):
        border_aware_text(
            x_min,
            x_max,
            y_min,
            y_max,
            other_x[n],
            other_y[n],
            f"{other_titles[n]}",
            fontsize=11,
        )

    plt.xlabel("Memorized cards (average, all days)\n(higher=better)", fontsize=18)
    plt.ylabel(
        "Memorized/hours spent (average, all days)\n(higher=better)", fontsize=18
    )
    plt.xticks(fontsize=16, color="black")
    plt.yticks(fontsize=16, color="black")
    plt.title(
        f"Pareto frontier (duration of the simulation={LEARN_SPAN/365:.0f} years,"
        f"\ndeck size={DECK_SIZE}, new cards/day=10, S_max={S_MAX/365:.0f} years",
        fontsize=24,
    )
    plt.grid(True, ls="--")
    plt.legend(fontsize=18, loc="lower left", facecolor="white")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "Pareto frontier.png")
    plt.show()
    plt.close()


def print_simulation_summary(results_path):
    simulation_results = load_simulation_results(results_path)
    for entry in simulation_results:
        title = entry["title"]
        reviews_average = entry["reviews_average"]
        time_average = entry["time_average"]
        memorized_average = entry["memorized_average"]
        avg_accum_memorized_per_hour = entry["avg_accum_memorized_per_hour"]
        print(
            f"| {title} | {reviews_average:.1f} | {time_average:.1f} | "
            f"{memorized_average:.0f} | {avg_accum_memorized_per_hour:.1f} |"
        )


def run_experiment(
    policy_configs,
    simulation_type,
    w,
    device,
    dr_baseline_path,
    policies,
):
    ensure_output_dirs()
    save_simulation_results([], SIMULATION_RESULTS_PATH)

    policies = normalize_policy_list(policies)

    learn_limit_per_day, max_studying_time_per_day = resolve_simulation_limits(
        simulation_type
    )
    simulate_policy = simulate_policy_factory(
        w, device, learn_limit_per_day, max_studying_time_per_day
    )

    solver = None
    retention_matrix = None
    init_stabilities = None
    init_difficulties = None

    if "ssp-mmc" in policies:
        if not policy_configs:
            raise ValueError("policy_configs is required when evaluating SSP-MMC.")
        (
            solver,
            retention_matrix,
            init_stabilities,
            init_difficulties,
        ) = run_ssp_mmc_configs(
            policy_configs,
            SIMULATION_RESULTS_PATH,
            simulate_policy,
            device,
        )

    if "memrise" in policies:
        plot_simulation(
            memrise_policy, "Memrise", SIMULATION_RESULTS_PATH, simulate_policy
        )
    if "anki-sm-2" in policies:
        plot_simulation(
            anki_sm2_policy, "Anki-SM-2", SIMULATION_RESULTS_PATH, simulate_policy
        )

    if solver is not None and retention_matrix is not None:
        plot_optimal_policy_vs_stability(
            solver, retention_matrix, init_stabilities, init_difficulties, w
        )

    if "dr" in policies:
        simulate_dr_policies(SIMULATION_RESULTS_PATH, simulate_policy)
        print("--------------------------------")
        save_dr_baseline_from_results(SIMULATION_RESULTS_PATH, dr_baseline_path)
    else:
        print("Skipping DR sweep and baseline generation.")

    if "interval" in policies:
        run_fixed_interval_policies(SIMULATION_RESULTS_PATH, simulate_policy)

    print(
        "| Scheduling Policy | Reviews per day (average, lower=better) | "
        "Minutes per day (average, lower=better) | "
        "Memorized cards (average, all days, higher=better) | "
        "Memorized/hours spent (average, all days, higher=better) |"
    )
    print("| --- | --- | --- | --- | --- |")

    print_simulation_summary(SIMULATION_RESULTS_PATH)

    plot_pareto_frontier(SIMULATION_RESULTS_PATH, policy_configs or [])
