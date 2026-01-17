from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

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
    DEFAULT_SEED,
    DEFAULT_W,
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
    default_device,
    resolve_simulation_limits,
)
from ssp_mmc_fsrs.core import next_interval, next_interval_torch  # noqa: E402
from ssp_mmc_fsrs.io import save_policy  # noqa: E402
from ssp_mmc_fsrs.policies import (  # noqa: E402
    anki_sm2_policy,
    create_dr_policy,
    create_fixed_interval_policy,
    memrise_policy,
)
from ssp_mmc_fsrs.simulation import simulate  # noqa: E402
from ssp_mmc_fsrs.solver import SSPMMCSolver  # noqa: E402


def _ensure_output_dirs():
    for path in (PLOTS_DIR, SIMULATION_DIR, POLICIES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def main():
    plt.style.use("ggplot")
    np.random.seed(DEFAULT_SEED)

    review_costs = DEFAULT_REVIEW_COSTS
    first_rating_prob = DEFAULT_FIRST_RATING_PROB
    review_rating_prob = DEFAULT_REVIEW_RATING_PROB
    first_rating_offsets = DEFAULT_FIRST_RATING_OFFSETS
    first_session_lens = DEFAULT_FIRST_SESSION_LENS
    forget_rating_offset = DEFAULT_FORGET_RATING_OFFSET
    forget_session_len = DEFAULT_FORGET_SESSION_LEN

    simulation_type = "unlim_time_lim_reviews"
    save_policies = True
    learn_limit_per_day, max_studying_time_per_day = resolve_simulation_limits(
        simulation_type
    )

    w = DEFAULT_W
    device = default_device()

    _ensure_output_dirs()

    simulation_table = []

    list_of_dictionaries = [[{'a0': 'log', 'a1': 0.47, 'a2': 0.1, 'a3': 3.99, 'a4': -5.0, 'a5': 5.0, 'a6': -5.0, 'a7': -3.52, 'a8': 5.0}, 'Maximum knowledge'], [{'a0': 'log', 'a1': 1.16, 'a2': 0.1, 'a3': 4.27, 'a4': -5.0, 'a5': 0.14, 'a6': -5.0, 'a7': 1.25, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 0.57, 'a2': 10.0, 'a3': 5.0, 'a4': 5.0, 'a5': 1.08, 'a6': -5.0, 'a7': 2.35, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 0.9, 'a2': 10.0, 'a3': 3.48, 'a4': -1.76, 'a5': -0.72, 'a6': -5.0, 'a7': -0.64, 'a8': 5.0}, None], [{'a0': 'no_log', 'a1': 0.1, 'a2': 10.0, 'a3': 3.5, 'a4': -5.0, 'a5': -5.0, 'a6': -5.0, 'a7': -2.17, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 1.74, 'a2': 10.0, 'a3': 3.9, 'a4': 0.21, 'a5': -0.02, 'a6': -5.0, 'a7': -5.0, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 1.59, 'a2': 10.0, 'a3': 3.3, 'a4': 1.56, 'a5': 0.03, 'a6': -5.0, 'a7': 0.1, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 2.17, 'a2': 10.0, 'a3': 3.41, 'a4': 3.01, 'a5': -0.08, 'a6': -5.0, 'a7': -0.76, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 1.26, 'a2': 10.0, 'a3': 1.34, 'a4': -5.0, 'a5': -5.0, 'a6': -5.0, 'a7': 5.0, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 1.27, 'a2': 10.0, 'a3': 3.52, 'a4': 0.81, 'a5': -5.0, 'a6': -5.0, 'a7': 0.19, 'a8': -5.0}, None], [{'a0': 'log', 'a1': 1.6, 'a2': 10.0, 'a3': 4.54, 'a4': 4.96, 'a5': -5.0, 'a6': -5.0, 'a7': 1.12, 'a8': -5.0}, None], [{'a0': 'log', 'a1': 2.02, 'a2': 10.0, 'a3': 2.56, 'a4': 3.19, 'a5': 0.13, 'a6': -5.0, 'a7': -1.61, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 1.14, 'a2': 10.0, 'a3': 3.58, 'a4': 3.9, 'a5': -5.0, 'a6': -5.0, 'a7': 2.4, 'a8': -5.0}, None], [{'a0': 'log', 'a1': 4.29, 'a2': 10.0, 'a3': 2.37, 'a4': 5.0, 'a5': 5.0, 'a6': -5.0, 'a7': -2.03, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 4.23, 'a2': 10.0, 'a3': 2.23, 'a4': 5.0, 'a5': 5.0, 'a6': -5.0, 'a7': -0.8, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 3.63, 'a2': 5.13, 'a3': 3.27, 'a4': 4.06, 'a5': 0.36, 'a6': -3.8, 'a7': -3.47, 'a8': -3.09}, None], [{'a0': 'log', 'a1': 3.62, 'a2': 9.18, 'a3': 1.86, 'a4': 4.31, 'a5': 4.22, 'a6': -4.8, 'a7': -0.44, 'a8': -2.08}, None], [{'a0': 'log', 'a1': 3.75, 'a2': 10.0, 'a3': 1.82, 'a4': 5.0, 'a5': 5.0, 'a6': -5.0, 'a7': 0.56, 'a8': -2.49}, None], [{'a0': 'log', 'a1': 3.59, 'a2': 10.0, 'a3': 1.68, 'a4': 5.0, 'a5': 5.0, 'a6': -5.0, 'a7': 1.02, 'a8': -5.0}, None], [{'a0': 'log', 'a1': 3.64, 'a2': 3.06, 'a3': 4.0, 'a4': 4.07, 'a5': -1.5, 'a6': -3.22, 'a7': -5.0, 'a8': -2.74}, None], [{'a0': 'log', 'a1': 3.44, 'a2': 2.89, 'a3': 3.78, 'a4': 3.85, 'a5': -1.42, 'a6': -3.05, 'a7': -5.0, 'a8': -2.59}, None], [{'a0': 'log', 'a1': 3.04, 'a2': 2.56, 'a3': 3.35, 'a4': 3.41, 'a5': -1.26, 'a6': -2.7, 'a7': -4.44, 'a8': -2.29}, None], [{'a0': 'log', 'a1': 2.96, 'a2': 6.94, 'a3': 2.26, 'a4': 4.86, 'a5': -0.08, 'a6': -5.0, 'a7': -5.0, 'a8': -2.87}, None], [{'a0': 'log', 'a1': 2.88, 'a2': 2.42, 'a3': 3.18, 'a4': 3.24, 'a5': -1.19, 'a6': -2.56, 'a7': -4.63, 'a8': -2.17}, None], [{'a0': 'log', 'a1': 3.46, 'a2': 10.0, 'a3': 0.75, 'a4': 5.0, 'a5': 3.09, 'a6': -5.0, 'a7': -0.85, 'a8': 5.0}, None], [{'a0': 'log', 'a1': 2.79, 'a2': 4.07, 'a3': 2.46, 'a4': 3.9, 'a5': -1.34, 'a6': -3.59, 'a7': -5.0, 'a8': -5.0}, 'Maximum efficiency']]

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

    def plot_simulation(policy, title):
        review_cnt_per_day, cost_per_day, memorized_cnt_per_day = simulate_policy(
            policy
        )

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

        simulation_table.append(
            (
                title,
                reviews_average,
                time_average,
                memorized_average,
                avg_accum_memorized_per_hour,
            )
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

    for param_dict_with_name in list_of_dictionaries:
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

        def ssp_mmc_policy(s, d):
            d_index = solver.d2i_torch(d)
            s_index = solver.s2i_torch(s)
            mask = (d_index >= solver.d_size) | (s_index >= solver.s_size - 1)
            optimal_interval = torch.zeros_like(s)
            optimal_interval[~mask] = next_interval_torch(
                s[~mask],
                retention_matrix_tensor[d_index[~mask], s_index[~mask]],
                -w[20],
            )
            optimal_interval[mask] = np.inf
            return optimal_interval

        cost_matrix, retention_matrix = solver.solve(param_dict_with_name[0])
        retention_matrix_tensor = torch.tensor(retention_matrix, device=device)
        init_stabilities = solver.init_s(np.arange(1, 5))
        init_difficulties = solver.init_d_with_short_term(np.arange(1, 5))
        init_cost = cost_matrix[
            solver.d2i(init_difficulties), solver.s2i(init_stabilities)
        ]
        avg_cost = init_cost @ first_rating_prob
        print(f"Average cost: {avg_cost:.2f}")
        avg_retention = retention_matrix.mean()
        print(f"Average retention: {avg_retention:.2f}")

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
        ax.plot_surface(
            s_state_mesh_2d, d_state_mesh_2d, interval_matrix, cmap="viridis"
        )
        ax.set_xlabel("Stability")
        ax.set_ylabel("Difficulty")
        ax.set_zlabel("Interval")
        ax.set_title("Interval")
        ax.set_box_aspect(None, zoom=0.8)

        title = (
            f"SSP-MMC-FSRS ({param_dict_with_name[-1]})"
            if param_dict_with_name[-1] is not None
            else "SSP-MMC-FSRS"
        )

        if save_policies:
            save_policy(
                POLICIES_DIR,
                title,
                solver,
                cost_matrix,
                retention_matrix,
                param_dict_with_name[0],
            )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{title}.png")
        plt.close()

        plot_simulation(ssp_mmc_policy, title)

    plot_simulation(memrise_policy, "Memrise")
    plot_simulation(anki_sm2_policy, "Anki-SM-2")

    def optimal_policy_for_rating_sequence(rating_sequence: list[int]):
        s_list = []
        r_list = []
        ivl_list = []
        g_list = []
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

    def plot_optimal_policy_vs_stability(rating_sequence: list[int]):
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
        plt.close()

    for rating in range(1, 5):
        plot_optimal_policy_vs_stability([rating] + [3 for _ in range(100)])

    costs = []

    r_range = np.arange(R_MIN, R_MAX, 0.01)

    for r in r_range:
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
        avg_cost = init_cost @ first_rating_prob
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
        plt.close()
        dr_policy = create_dr_policy(r)
        plot_simulation(dr_policy, f"DR={r:.2f}")

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
    plt.close()

    for fixed_interval in [7, 14, 20, 30, 50, 75, 100]:
        fixed_policy = create_fixed_interval_policy(fixed_interval)
        plot_simulation(fixed_policy, f"Interval={fixed_interval}")

    print("--------------------------------")

    print(
        "| Scheduling Policy | Reviews per day (average, lower=better) | "
        "Minutes per day (average, lower=better) | "
        "Memorized cards (average, all days, higher=better) | "
        "Memorized/hours spent (average, all days, higher=better) |"
    )
    print("| --- | --- | --- | --- | --- |")

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

    for (
        title,
        reviews_average,
        time_average,
        memorized_average,
        avg_accum_memorized_per_hour,
    ) in simulation_table:
        print(
            f"| {title} | {reviews_average:.1f} | {time_average:.1f} | {memorized_average:.0f} | {avg_accum_memorized_per_hour:.1f} |"
        )

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
    for list_with_dict in list_of_dictionaries:
        if list_with_dict[1] == "Balanced":
            balanced_index = list_of_dictionaries.index(list_with_dict)

    plt.plot(
        ssp_mmc_x,
        ssp_mmc_y,
        label="SSP-MMC-FSRS",
        linewidth=2,
        color="#00b050",
        marker="o",
    )
    plt.plot(
        ssp_mmc_x[balanced_index],
        ssp_mmc_y[balanced_index],
        linewidth=2,
        color="#00b050",
        marker=(5, 1, 15),
        ms=20,
    )
    plt.plot(
        fixed_dr_x,
        fixed_dr_y,
        label="Fixed DR (FSRS)",
        linewidth=2,
        color="#5b9bd5",
        marker="s",
    )
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
    plt.plot(
        fixed_intervals_x,
        fixed_intervals_y,
        label="Fixed intervals",
        linewidth=2,
        color="black",
        marker="x",
        ms=7.5,
    )

    x_min = 200 * np.floor((min([_[3] for _ in simulation_table]) / 200))
    x_max = 200 * np.ceil((max([_[3] for _ in simulation_table]) / 200))
    y_min = 0
    y_max = max([_[4] for _ in simulation_table]) * 1.03

    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])

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


if __name__ == "__main__":
    main()
