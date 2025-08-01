import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from simulator import (
    DEFAULT_REVIEW_COSTS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_REVIEW_RATING_PROB,
    DEFAULT_FIRST_RATING_OFFSETS,
    DEFAULT_FIRST_SESSION_LENS,
    DEFAULT_FORGET_RATING_OFFSET,
    DEFAULT_FORGET_SESSION_LEN,
    next_interval_torch,
    next_interval,
    power_forgetting_curve,
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

S_MIN = 0.1
S_MAX = 365 * 25
S_MID = 5
SHORT_STEP = (np.log(S_MID) - np.log(S_MIN)) / 100
LONG_STEP = (np.log(S_MAX) - np.log(S_MID)) / 1900

D_MIN = 1
D_MAX = 10
D_EPS = 0.1

R_MIN = 0.70
R_MAX = 0.99
R_EPS = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARALLEL = 100

COST_MAX = 1_000_000

w = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    0.1542
]


def bellman_solver(
    n_iter,
    state_np: np.array,
    transitions_np: list[np.array],
    transition_probs_np: list[np.array],
    costs_np: np.array,
    discount_factor=0.97,
    device=DEVICE,
):
    S, D, R = costs_np[0].shape
    DTYPE = torch.float32
    with torch.inference_mode():
        state = torch.tensor(state_np, requires_grad=False, device=device, dtype=DTYPE)
        transitions = [
            torch.tensor(
                transition_np, requires_grad=False, device=device, dtype=torch.long
            )
            for transition_np in transitions_np
        ]
        transition_probs = [
            torch.tensor(
                transition_prob_np, requires_grad=False, device=device, dtype=DTYPE
            )
            for transition_prob_np in transition_probs_np
        ]
        costs = [
            torch.tensor(cost_np, requires_grad=False, device=device, dtype=DTYPE)
            for cost_np in costs_np
        ]
        it = 0
        cost_diff = 1e9
        while it < n_iter and cost_diff > 1e-5:
            it += 1

            action_value = torch.zeros(
                (S, D, R), requires_grad=False, device=device, dtype=DTYPE
            )
            for transition, transition_prob, cost in zip(
                transitions, transition_probs, costs
            ):
                d, s = transition.unbind(dim=-1)
                action_value += transition_prob * (cost + discount_factor * state[d, s])

            optimal_value, optimal_action = action_value.min(dim=-1)
            optimal_value = torch.minimum(state, optimal_value)
            cost_diff = torch.abs(optimal_value - state).sum().item()
            state = optimal_value

            if it % 100 == 0:
                print(f"it: {it}, cost diff: {cost_diff}")

    print(f"Done. it: {it}, cost diff: {cost_diff}")
    return state.cpu().numpy(), optimal_action.cpu().numpy()


def max_r_to_reach_next_stability(s, s_next, d, rating):
    hard_penalty = torch.where(rating == 2, w[15], 1)
    easy_bonus = torch.where(rating == 4, w[16], 1)
    c = np.exp(w[8]) * (11 - d) * torch.pow(s, -w[9]) * hard_penalty * easy_bonus

    # let f(r) = e^((1-r) * w[10])  - 1
    # Then s_next = s * (1 + c * f(r))
    # => f(r) = (s_next / s - 1) / c
    # => e^((1 - r) * w[10]) - 1 = (s_next / s - 1) / c
    # => 1 - r = log(((s_next / s - 1) / c + 1)) / w[10]
    return torch.maximum(
        torch.tensor(0.01, device=s.device),
        1 - torch.log(((s_next / s - 1) / c + 1)) / w[10],
    )


def next_interval_ceil(s, r, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    ivl = s / factor * (r ** (1.0 / decay) - 1.0)
    return torch.maximum(torch.tensor(1, device=s.device), torch.ceil(ivl))


def s_max_aware_next_interval(s, d, dr, decay):
    int_base = next_interval_torch(s, dr, decay)
    int_req = next_interval_ceil(
        s,
        max_r_to_reach_next_stability(s, S_MAX + 1e-3, d, torch.full_like(s, 3)),
        decay,
    )
    return torch.where(s > S_MAX, 1e9, torch.minimum(int_base, int_req))


def s_max_aware_fixed_interval(s, d, fixed_interval, decay):
    int_base = fixed_interval
    int_req = next_interval_ceil(
        s,
        max_r_to_reach_next_stability(s, S_MAX + 1e-3, d, torch.full_like(s, 3)),
        decay,
    )
    return torch.where(
        s > S_MAX, 1e9, torch.minimum(torch.tensor(int_base, device=s.device), int_req)
    )


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
        w,
    ):
        self.review_costs = review_costs
        self.first_rating_prob = first_rating_prob
        self.review_rating_prob = review_rating_prob
        self.first_rating_offsets = first_rating_offsets
        self.first_session_lens = first_session_lens
        self.forget_rating_offset = forget_rating_offset
        self.forget_session_len = forget_session_len
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
        rating = 3
        sinc = np.exp(self.w[17] * (rating - 3 + self.w[18])) * np.power(s, -self.w[19])
        return s * sinc

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
            np.ceil(
                (np.log(s[small_mask]) - np.log(self.s_min)) / self.short_step
            ).astype(int),
            0,
            len(self.s_state_small) - 1,
        )

        # Handle large values (linear scale)
        result[~small_mask] = len(self.s_state_small) + np.clip(
            np.ceil(
                (s[~small_mask] - self.s_state_small[-1] - self.long_step)
                / self.long_step
            ).astype(int),
            0,
            len(self.s_state_large) - 1,
        )

        return result

    def s2i_torch(self, s):
        result = torch.zeros_like(s, dtype=torch.int)
        small_mask = s <= self.s_mid

        # Handle small values (logarithmic scale)
        result[small_mask] = torch.clamp(
            torch.ceil(
                (torch.log(s[small_mask]) - np.log(self.s_min)) / self.short_step
            ).to(torch.int),
            0,
            len(self.s_state_small) - 1,
        )

        # Handle large values (linear scale)
        result[~small_mask] = len(self.s_state_small) + torch.clamp(
            torch.ceil(
                (s[~small_mask] - self.s_state_small[-1] - self.long_step)
                / self.long_step
            ).to(torch.int),
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

    def d2i_torch(self, d):
        return torch.clamp(
            (
                torch.floor((d - self.d_min) / (self.d_max - self.d_min) * self.d_size)
            ).to(torch.int),
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
        s_mid=S_MID,
        short_step=SHORT_STEP,
        long_step=LONG_STEP,
        d_min=D_MIN,
        d_max=D_MAX,
        d_eps=D_EPS,
        r_min=R_MIN,
        r_max=R_MAX,
        r_eps=R_EPS,
        cost_max=COST_MAX,
    ):
        # Stability state space
        self.s_min = s_min
        self.s_max = s_max
        self.s_mid = s_mid
        self.short_step = short_step
        self.long_step = long_step

        self.s_state_small = np.exp(
            np.arange(np.log(self.s_min), np.log(self.s_mid), self.short_step)
        )
        self.s_state_large = np.exp(
            np.arange(np.log(self.s_mid), np.log(self.s_max), self.long_step)
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
        self.cost_matrix = np.full((self.d_size, self.s_size), cost_max)
        self.cost_matrix[:, -1] = 0
        self.retention_matrix = np.zeros_like(self.cost_matrix)

        # Create meshgrids
        self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d = np.meshgrid(
            self.s_state, self.d_state, self.r_state
        )

    def solve(self, n_iter=100_000, verbose=True):
        """Solve the SSP-MMC problem using value iteration."""
        # Initial setup
        ivl_mesh = next_interval(
            self.s_state_mesh_3d, self.r_state_mesh_3d, -self.w[20]
        )
        self.r_state_mesh_3d = power_forgetting_curve(
            ivl_mesh, self.s_state_mesh_3d, -self.w[20]
        )
        zeros = np.zeros_like(self.s_state_mesh_3d)  # For broadcasting
        transitions = []
        transition_probs = []
        costs = []

        # Again case
        next_s_again = self.stability_short_term(
            self.stability_after_failure(
                self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d
            )
        )
        next_d_again = self.next_d(self.d_state_mesh_3d, 1)
        transitions.append(
            np.stack([self.d2i(next_d_again), self.s2i(next_s_again)], axis=-1)
        )
        transition_probs.append(1.0 - self.r_state_mesh_3d)
        costs.append(zeros + self.review_costs[0])

        # Calculate costs for each rating
        for i, (g, review_cost) in enumerate(zip([2, 3, 4], self.review_costs[1:])):
            next_s = self.stability_after_success(
                self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d, g
            )
            next_d = self.next_d(self.d_state_mesh_3d, g)
            transitions.append(np.stack([self.d2i(next_d), self.s2i(next_s)], axis=-1))
            transition_probs.append(self.r_state_mesh_3d * self.review_rating_prob[i])
            costs.append(zeros + review_cost)

        assert len(transitions) == len(transition_probs) and len(transitions) == len(
            costs
        )
        self.cost_matrix, optimal_r_indices = bellman_solver(
            n_iter, self.cost_matrix, transitions, transition_probs, costs
        )
        retention_d_indices, retention_s_indices = np.meshgrid(
            np.arange(self.d_size), np.arange(self.s_size), indexing="ij"
        )
        self.retention_matrix = self.r_state_mesh_3d[
            retention_d_indices, retention_s_indices, optimal_r_indices
        ]
        return self.cost_matrix, self.retention_matrix

    def _evaluate_policy(self, n_iter=100_000):
        """Evaluate the cost and retention for a given r_state_mesh_2d."""
        i = 0
        cost_diff = COST_MAX
        # Pre-calculate next states for again case
        next_s_again = self.stability_short_term(
            self.stability_after_failure(
                self.s_state_mesh_2d, self.d_state_mesh_2d, self.r_state_mesh_2d
            )
        )
        next_d_again = self.next_d(self.d_state_mesh_2d, 1)
        # Pre-calculate next states for each rating
        next_s_list = []
        next_d_list = []
        for g in [2, 3, 4]:
            next_s = self.stability_after_success(
                self.s_state_mesh_2d, self.d_state_mesh_2d, self.r_state_mesh_2d, g
            )
            next_d = self.next_d(self.d_state_mesh_2d, g)
            next_s_list.append(next_s)
            next_d_list.append(next_d)

        while i < n_iter and cost_diff > 1e-4 * self.s_size * self.d_size:
            cost_again = (
                self._get_cost(next_s_again, next_d_again) + self.review_costs[0]
            )

            # Calculate costs for each rating
            costs = []
            for next_s, next_d, review_cost in zip(
                next_s_list, next_d_list, self.review_costs[1:]
            ):
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

    def evaluate_r_threshold(self, r_threshold, n_iter=100_000):
        """Evaluate the cost and retention for a given r threshold."""
        self.r_state_mesh_2d = r_threshold * np.ones_like(self.cost_matrix)
        self.s_state_mesh_2d, self.d_state_mesh_2d = np.meshgrid(
            self.s_state, self.d_state
        )
        ivl_mesh = next_interval(
            self.s_state_mesh_2d, self.r_state_mesh_2d, -self.w[20]
        )
        self.r_state_mesh_2d = power_forgetting_curve(
            ivl_mesh, self.s_state_mesh_2d, -self.w[20]
        )
        return self._evaluate_policy(n_iter)

    def _get_cost(self, s, d):
        """Get cost from cost matrix for given stability and difficulty."""
        return self.cost_matrix[self.d2i(d), self.s2i(s)]

def memrise_policy(stability, difficulty, prev_interval, grade):
    """
    Vectorized version of fixed sequence policy with closest interval matching
    Special case: if prev_interval=0 (new cards), start with interval=1
    """
    device = prev_interval.device
    dtype = prev_interval.dtype

    # Define the interval sequence
    sequence = torch.tensor([1, 6, 12, 48, 96, 180], device=device, dtype=dtype)

    # Special case: new cards (prev_interval = 0) always start with 1 day
    is_new_card = prev_interval == 0

    # For existing cards, find the closest interval in the sequence
    prev_expanded = prev_interval.unsqueeze(-1)  # Shape: (..., 1)
    sequence_expanded = sequence.unsqueeze(0).expand_as(
        torch.cat([prev_expanded] * len(sequence), dim=-1)
    )  # Shape: (..., 6)

    # Calculate distances to each sequence value
    distances = torch.abs(prev_expanded - sequence_expanded)

    # Find the index of the closest sequence value
    closest_indices = torch.argmin(distances, dim=-1)

    # Calculate next indices (advance by 1, but cap at last position)
    next_indices = torch.clamp(closest_indices + 1, 0, len(sequence) - 1)

    # Get the next intervals
    next_intervals = sequence[next_indices]

    # Handle different cases:
    # 1. New cards (prev_interval=0): always start with 1 day
    # 2. Again grade: reset to 1 day
    # 3. Hard/Good/Easy: advance in sequence with s_max awareness
    result = torch.where(
        is_new_card,  # New cards
        torch.ones_like(prev_interval),  # Start with 1 day
        torch.where(
            grade == 1,  # Again
            torch.ones_like(prev_interval),  # Reset to 1 day
            s_max_aware_fixed_interval(stability, difficulty, next_intervals, -w[20])
            # Hard/Good/Easy: advance from closest sequence position
        )
    )

    return result

def create_fixed_interval_policy(interval):
    """Create a fixed interval policy that uses the full 4-parameter signature"""
    def fixed_policy(stability, difficulty, prev_interval, grade):
        return s_max_aware_fixed_interval(stability, difficulty, interval, -w[20])
    return fixed_policy

def create_dr_policy(desired_retention):
    """Create a DR policy that uses the full 4-parameter signature"""
    def dr_policy(stability, difficulty, prev_interval, grade):
        return s_max_aware_next_interval(stability, difficulty, desired_retention, -w[20])
    return dr_policy

if __name__ == "__main__":
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

    cost_matrix, retention_matrix = solver.solve()
    retention_matrix_tensor = torch.tensor(retention_matrix, device=DEVICE)
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
    interval_matrix = next_interval(s_state_mesh_2d, retention_matrix, -w[20])
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
        d_index = solver.d2i_torch(d)
        s_index = solver.s2i_torch(s)
        # Handle array inputs by checking each element
        mask = (d_index >= solver.d_size) | (s_index >= solver.s_size - 1)
        optimal_interval = torch.zeros_like(s)
        optimal_interval[~mask] = next_interval_torch(
            s[~mask], retention_matrix_tensor[d_index[~mask], s_index[~mask]], -w[20]
        )
        optimal_interval[mask] = np.inf
        return optimal_interval

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
            device=DEVICE,
            deck_size=10000,
            learn_span=365 * 5,
            s_max=S_MAX,
        )

        def moving_average(data, window_size=365 // 20):
            weights = np.ones(window_size) / window_size
            return np.apply_along_axis(
                lambda x: np.convolve(x, weights, mode="valid"), axis=-1, arr=data
            )

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
                review_cnt_per_day.mean(axis=-1).mean(axis=-1),
                cost_per_day.mean(axis=-1).mean(axis=-1) / 60,
                memorized_cnt_per_day[:, -1].mean(),
                (
                    memorized_cnt_per_day[:, -1] / (cost_per_day.mean(axis=-1) / 60)
                ).mean(),
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
        plt.savefig(f"./simulation/{title}.png")
        plt.close()

    plot_simulation(ssp_mmc_policy, "SSP-MMC")

    plot_simulation(memrise_policy, "Memrise")

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

    r_range = np.arange(R_MIN, R_MAX, 0.01)

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
    plt.savefig("./plot/cost_vs_retention.png")
    plt.close()

    for fixed_interval in [3, 7, 30]:
        fixed_policy = create_fixed_interval_policy(fixed_interval)
        plot_simulation(fixed_policy, f"IVL={fixed_interval}")

    print("--------------------------------")

    print(
        "| Scheduling Policy | Average number of reviews per day | Average number of minutes per day | Total knowledge at the end | Knowledge per minute |"
    )
    print("| --- | --- | --- | --- | --- |")
    for (
        title,
        review_cnt_per_day,
        cost_per_day,
        memorized_cnt_at_end,
        knowledge_per_minute,
    ) in simulation_table:
        print(
            f"| {title} | {review_cnt_per_day:.1f} | {cost_per_day:.1f} | {memorized_cnt_at_end:.0f} | {knowledge_per_minute:.0f} |"
        )
