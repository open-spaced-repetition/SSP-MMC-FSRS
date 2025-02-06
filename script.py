import time
import numpy as np
import matplotlib.pyplot as plt
from simulator import (
    DEFAULT_REVIEW_COSTS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_REVIEW_RATING_PROB,
    DEFAULT_FIRST_RATING_OFFSETS,
    DEFAULT_FIRST_SESSION_LENS,
    DEFAULT_FORGET_RATING_OFFSET,
    DEFAULT_FORGET_SESSION_LEN,
    power_forgetting_curve,
    next_interval,
    simulate,
)

plt.style.use("ggplot")

review_costs = DEFAULT_REVIEW_COSTS
first_rating_prob = DEFAULT_FIRST_RATING_PROB
review_rating_prob = DEFAULT_REVIEW_RATING_PROB
first_rating_offsets = DEFAULT_FIRST_RATING_OFFSETS
first_session_lens = DEFAULT_FIRST_SESSION_LENS
forget_rating_offset = DEFAULT_FORGET_RATING_OFFSET
forget_session_len = DEFAULT_FORGET_SESSION_LEN
loss_aversion = 2.5

S_MIN = 0.1
S_MAX = 365 * 3
SHORT_STEP = np.log(2) / 15
LONG_STEP = 10

D_MIN = 1
D_MAX = 10
D_EPS = 1 / 3

R_MIN = 0.70
R_MAX = 0.97
R_EPS = 0.01

w = [
    0.40255,
    1.18385,
    3.173,
    15.69105,
    7.1949,
    0.5345,
    1.4604,
    0.0046,
    1.54575,
    0.1192,
    1.01925,
    1.9395,
    0.11,
    0.29605,
    2.2698,
    0.2315,
    2.9898,
    0.51655,
    0.6621,
]


class SSPMMCSolver:
    def __init__(
        self,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        first_rating_offsets,
        first_session_lens,
        forget_rating_offset,
        forget_session_len,
        loss_aversion,
        w,
    ):
        self.review_costs = review_costs
        self.first_rating_prob = first_rating_prob
        self.review_rating_prob = review_rating_prob
        self.first_rating_offsets = first_rating_offsets
        self.first_session_lens = first_session_lens
        self.forget_rating_offset = forget_rating_offset
        self.forget_session_len = forget_session_len
        self.loss_aversion = loss_aversion
        self.w = w

        # Initialize state spaces
        self._init_state_spaces()

    def stability_after_success(self, s, d, r, g):
        """Calculate stability after a successful review."""
        return s * (
            1
            + np.exp(self.w[8])
            * (11 - d)
            * np.power(s, -self.w[9])
            * (np.exp((1 - r) * self.w[10]) - 1)
            * (self.w[15] if g == 2 else 1)
            * (self.w[16] if g == 4 else 1)
        )

    def stability_after_failure(self, s, d, r):
        """Calculate stability after a failed review."""
        return np.maximum(
            self.s_min,
            np.minimum(
                self.w[11]
                * np.power(d, -self.w[12])
                * (np.power(s + 1, self.w[13]) - 1)
                * np.exp((1 - r) * self.w[14]),
                s / np.exp(self.w[17] * self.w[18]),
            ),
        )

    def stability_short_term(self, s):
        """Calculate short-term stability."""
        return s * np.exp(
            self.w[17]
            * (self.forget_rating_offset + self.forget_session_len * self.w[18])
        )

    def init_s(self, rating):
        """Initialize stability for a given rating."""
        return np.choose(
            rating - 1,
            np.array(self.w[0:4])
            * np.exp(
                self.w[17]
                * (self.first_rating_offsets + self.first_session_lens * self.w[18])
            ),
        )

    def init_d(self, rating):
        """Initialize difficulty for a given rating."""
        return self.w[4] - np.exp(self.w[5] * (rating - 1)) + 1

    def init_d_with_short_term(self, rating):
        """Initialize difficulty with short-term adjustment."""
        rating_offset = np.choose(rating - 1, self.first_rating_offsets)
        new_d = self.init_d(rating) - self.w[6] * rating_offset
        return np.clip(new_d, 1, 10)

    def linear_damping(self, delta_d, old_d):
        """Apply linear damping to difficulty change."""
        return delta_d * (10 - old_d) / 9

    def next_d(self, d, g):
        """Calculate next difficulty after a review."""
        delta_d = -self.w[6] * (g - 3)
        new_d = d + self.linear_damping(delta_d, d)
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d.clip(1, 10)

    def mean_reversion(self, init, current):
        """Apply mean reversion to difficulty."""
        return self.w[7] * init + (1 - self.w[7]) * current

    def s2i(self, s):
        """Convert stability to index."""
        result = np.zeros_like(s, dtype=int)
        small_mask = s <= self.s_mid

        # Handle small values (logarithmic scale)
        result[small_mask] = np.clip(
            np.floor(
                (np.log(s[small_mask]) - np.log(self.s_min)) / self.short_step
            ).astype(int),
            0,
            len(self.s_state_small) - 1,
        )

        # Handle large values (linear scale)
        result[~small_mask] = len(self.s_state_small) + np.clip(
            np.floor(
                (s[~small_mask] - self.s_state_small[-1] - self.long_step)
                / self.long_step
            ).astype(int),
            0,
            len(self.s_state_large) - 1,
        )

        return result

    def d2i(self, d):
        """Convert difficulty to index."""
        return np.clip(
            np.floor((d - self.d_min) / (self.d_max - self.d_min) * self.d_size).astype(
                int
            ),
            0,
            self.d_size - 1,
        )

    def r2i(self, r):
        """Convert retention to index."""
        return np.clip(
            np.floor((r - self.r_min) / (self.r_max - self.r_min) * self.r_size).astype(
                int
            ),
            0,
            self.r_size - 1,
        )

    def _init_state_spaces(
        self,
        s_min=S_MIN,
        s_max=S_MAX,
        short_step=SHORT_STEP,
        long_step=LONG_STEP,
        d_min=D_MIN,
        d_max=D_MAX,
        d_eps=D_EPS,
        r_min=R_MIN,
        r_max=R_MAX,
        r_eps=R_EPS,
    ):
        # Stability state space
        self.s_min = s_min
        self.s_max = s_max
        self.short_step = short_step
        self.long_step = long_step
        self.s_mid = min(self.long_step / (1 - np.exp(-self.short_step)), self.s_max)

        self.s_state_small = np.exp(
            np.arange(np.log(self.s_min), np.log(self.s_mid), self.short_step)
        )
        self.s_state_large = np.arange(
            max(self.s_state_small) + self.long_step, self.s_max, self.long_step
        )
        self.s_state = np.concatenate([self.s_state_small, self.s_state_large])
        self.s_size = len(self.s_state)

        # Difficulty state space
        self.d_min = d_min
        self.d_max = d_max
        self.d_eps = d_eps
        self.d_size = np.ceil((self.d_max - self.d_min) / self.d_eps + 1).astype(int)
        self.d_state = np.linspace(self.d_min, self.d_max, self.d_size)

        # Retention state space
        self.r_min = r_min
        self.r_max = r_max
        self.r_eps = r_eps
        self.r_size = np.ceil((self.r_max - self.r_min) / self.r_eps + 1).astype(int)
        self.r_state = np.linspace(self.r_min, self.r_max, self.r_size)

        # Initialize matrices
        self.cost_matrix = np.full((self.d_size, self.s_size), 1000)
        self.cost_matrix[:, -1] = 0
        self.retention_matrix = np.zeros_like(self.cost_matrix)

        # Create meshgrids
        self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d = np.meshgrid(
            self.s_state, self.d_state, self.r_state
        )

    def solve(self, n_iter=10000, verbose=True):
        """Solve the SSP-MMC problem using value iteration."""
        start = time.time()
        i = 0
        cost_diff = 1000

        # Initial setup
        ivl_mesh = next_interval(self.s_state_mesh_3d, self.r_state_mesh_3d)
        self.r_state_mesh_3d = power_forgetting_curve(ivl_mesh, self.s_state_mesh_3d)

        while i < n_iter and cost_diff > 1e-4 * self.s_size * self.d_size:
            expected_cost = self._calculate_expected_cost()

            # Update cost matrix
            optimal_cost = np.minimum(self.cost_matrix, expected_cost.min(axis=2))
            cost_diff = np.sum(self.cost_matrix - optimal_cost)
            self.cost_matrix = optimal_cost

            # Update retention matrix
            last_retention_matrix = self.retention_matrix.copy()
            self.retention_matrix = self.r_state[np.argmin(expected_cost, axis=2)]
            r_diff = np.abs(self.retention_matrix - last_retention_matrix).sum()

            if verbose and i % 10 == 0:
                print(
                    f"iteration {i:>5}, cost diff {cost_diff:.2f}, "
                    f"retention diff {r_diff:.2f}, "
                    f"elapsed time {time.time() - start:.1f}s"
                )
            i += 1

        if verbose:
            print(f"Time: {time.time() - start:.2f}s")

        return self.cost_matrix, self.retention_matrix

    def evaluate_r_threshold(self, r_threshold, n_iter=10000):
        """Evaluate the cost and retention for a given r threshold."""
        self.r_state_mesh_2d = r_threshold * np.ones_like(self.cost_matrix)
        self.s_state_mesh_2d, self.d_state_mesh_2d = np.meshgrid(
            self.s_state, self.d_state
        )
        ivl_mesh = next_interval(self.s_state_mesh_2d, self.r_state_mesh_2d)
        self.r_state_mesh_2d = power_forgetting_curve(ivl_mesh, self.s_state_mesh_2d)

        i = 0
        cost_diff = 1000
        while i < n_iter and cost_diff > 1e-4 * self.s_size * self.d_size:
            next_s_again = self.stability_short_term(
                self.stability_after_failure(
                    self.s_state_mesh_2d, self.d_state_mesh_2d, self.r_state_mesh_2d
                )
            )
            next_d_again = self.next_d(self.d_state_mesh_2d, 1)
            cost_again = (
                self._get_cost(next_s_again, next_d_again)
                + self.review_costs[0] * self.loss_aversion
            )

            # Calculate costs for each rating
            costs = []
            for g, review_cost in zip([2, 3, 4], self.review_costs[1:]):
                next_s = self.stability_after_success(
                    self.s_state_mesh_2d, self.d_state_mesh_2d, self.r_state_mesh_2d, g
                )
                next_d = self.next_d(self.d_state_mesh_2d, g)
                costs.append(self._get_cost(next_s, next_d) + review_cost)

            expected_cost = (
                self.r_state_mesh_2d
                * (
                    self.review_rating_prob[0] * costs[0]
                    + self.review_rating_prob[1] * costs[1]
                    + self.review_rating_prob[2] * costs[2]
                )
                + (1 - self.r_state_mesh_2d) * cost_again
            )
            # update cost matrix
            optimal_cost = np.minimum(self.cost_matrix, expected_cost)
            cost_diff = self.cost_matrix.sum() - optimal_cost.sum()
            self.cost_matrix = optimal_cost
            i += 1
        return self.cost_matrix, self.r_state_mesh_2d

    def _calculate_expected_cost(self):
        """Calculate expected cost for all possible next states."""
        # Again case
        next_s_again = self.stability_short_term(
            self.stability_after_failure(
                self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d
            )
        )
        next_d_again = self.next_d(self.d_state_mesh_3d, 1)
        cost_again = (
            self._get_cost(next_s_again, next_d_again)
            + self.review_costs[0] * self.loss_aversion
        )

        # Calculate costs for each rating
        costs = []
        for g, review_cost in zip([2, 3, 4], self.review_costs[1:]):
            next_s = self.stability_after_success(
                self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d, g
            )
            next_d = self.next_d(self.d_state_mesh_3d, g)
            costs.append(self._get_cost(next_s, next_d) + review_cost)

        # Combine costs according to probabilities
        return (
            self.r_state_mesh_3d
            * (
                self.review_rating_prob[0] * costs[0]
                + self.review_rating_prob[1] * costs[1]
                + self.review_rating_prob[2] * costs[2]
            )
            + (1 - self.r_state_mesh_3d) * cost_again
        )

    def _get_cost(self, s, d):
        """Get cost from cost matrix for given stability and difficulty."""
        return self.cost_matrix[self.d2i(d), self.s2i(s)]


if __name__ == "__main__":
    solver = SSPMMCSolver(
        review_costs=DEFAULT_REVIEW_COSTS,
        first_rating_prob=DEFAULT_FIRST_RATING_PROB,
        review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
        first_rating_offsets=DEFAULT_FIRST_RATING_OFFSETS,
        first_session_lens=DEFAULT_FIRST_SESSION_LENS,
        forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
        forget_session_len=DEFAULT_FORGET_SESSION_LEN,
        loss_aversion=2.5,
        w=w,
    )

    cost_matrix, retention_matrix = solver.solve()
    init_stabilities = solver.init_s(np.arange(1, 5))
    init_difficulties = solver.init_d_with_short_term(np.arange(1, 5))
    init_cost = cost_matrix[solver.d2i(init_difficulties), solver.s2i(init_stabilities)]
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
    ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, retention_matrix, cmap="viridis")
    ax.set_xlabel("Stability")
    ax.set_ylabel("Difficulty")
    ax.set_zlabel("Retention")
    ax.set_title(f"Avg Retention: {avg_retention:.2f}")
    ax.set_box_aspect(None, zoom=0.8)

    ax = fig.add_subplot(133, projection="3d")
    interval_matrix = next_interval(s_state_mesh_2d, retention_matrix)
    ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, interval_matrix, cmap="viridis")
    ax.set_xlabel("Stability")
    ax.set_ylabel("Difficulty")
    ax.set_zlabel("Interval")
    ax.set_title("Interval")
    ax.set_box_aspect(None, zoom=0.8)

    plt.tight_layout()
    plt.savefig("./plot/SSP-MMC.png")
    plt.close()

    def ssp_mmc_policy(s, d):
        d_index = solver.d2i(d)
        s_index = solver.s2i(s)
        # Handle array inputs by checking each element
        mask = (d_index >= solver.d_size) | (s_index >= solver.s_size - 1)
        optimal_interval = np.zeros_like(s)
        optimal_interval[~mask] = next_interval(
            s[~mask], retention_matrix[d_index[~mask], s_index[~mask]]
        )
        optimal_interval[mask] = np.inf
        return optimal_interval

    def simulate_policy(policy):
        (
            _,
            review_cnt_per_day,
            _,
            memorized_cnt_per_day,
            cost_per_day,
            _,
        ) = simulate(
            w=w,
            policy=policy,
            deck_size=10000,
            learn_span=365 * 10,
            loss_aversion=loss_aversion,
            s_max=S_MAX,
        )

        def moving_average(data, window_size=365 // 20):
            weights = np.ones(window_size) / window_size
            return np.convolve(data, weights, mode="valid")

        return (
            moving_average(review_cnt_per_day),
            moving_average(cost_per_day),
            moving_average(memorized_cnt_per_day),
        )

    simulation_table = []

    def plot_simulation(policy, title):
        review_cnt_per_day, cost_per_day, memorized_cnt_per_day = simulate_policy(
            policy
        )
        simulation_table.append(
            (
                title,
                review_cnt_per_day.mean(),
                cost_per_day.mean() / 60,
                memorized_cnt_per_day[-1],
            )
        )
        fig = plt.figure(figsize=(16, 8.5))
        ax = fig.add_subplot(131)
        ax.plot(review_cnt_per_day)
        ax.set_title("Review Count")
        ax = fig.add_subplot(132)
        ax.plot(cost_per_day, label=f"Total Cost: {cost_per_day.sum():.2f}")
        ax.set_title("Cost")
        ax.legend()
        ax = fig.add_subplot(133)
        ax.plot(
            memorized_cnt_per_day,
            label=f"Total Memorized: {memorized_cnt_per_day[-1]:.2f}",
        )
        ax.set_title("Memorized Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"./simulation/{title}.png")
        plt.close()

    plot_simulation(ssp_mmc_policy, "SSP-MMC")

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
                ivl_list.append(next_interval(cur_s, optimal_r))
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
        ax.set_title(f"Optimal Retention vs Stability")
        ax = fig.add_subplot(122)
        ax.plot(s_list, ivl_list, "*-", label="Optimal")
        ax.plot(s_list, s_list, "--", alpha=0.5, label="R=90%")
        for s, ivl in zip(s_list, ivl_list):
            ax.text(s + 1, ivl - 10, f"{ivl:.0f}", fontsize=10)
        ax.set_xlabel("Stability")
        ax.set_ylabel("Optimal Interval")
        ax.set_title(f"Optimal Interval vs Stability")
        ax.legend()
        fig.suptitle(f"Rating Sequence: {','.join(map(str, g_list))}")
        plt.tight_layout()
        plt.savefig(f"./plot/OR-OI-{','.join(map(str, g_list))}.png")
        plt.close()

    for rating in range(1, 5):
        plot_optimal_policy_vs_stability([rating] + [3 for _ in range(100)])

    costs = []

    r_range = np.linspace(R_MIN, R_MAX, 10)

    for r in r_range:
        print("--------------------------------")
        start = time.time()
        solver._init_state_spaces()
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
        plt.savefig(f"./plot/DR={r:.2f}.png")
        plt.close()
        plot_simulation(lambda s, d: next_interval(s, r), f"DR={r:.2f}")

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
    plt.savefig("./plot/cost_vs_retention.png")
    plt.close()

    print("--------------------------------")

    print(
        "| Schedulling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |"
    )
    print("| --- | --- | --- | --- | --- |")
    for (
        title,
        review_cnt_per_day,
        cost_per_day,
        memorized_cnt_at_end,
    ) in simulation_table:
        print(
            f"| {title} | {review_cnt_per_day:.1f} | {cost_per_day:.1f} | {memorized_cnt_at_end:.0f} | {memorized_cnt_at_end / cost_per_day:.0f} |"
        )
