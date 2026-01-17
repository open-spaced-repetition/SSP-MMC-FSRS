import numpy as np
import time
import torch

from .config import (
    COST_MAX,
    D_EPS,
    D_MAX,
    D_MIN,
    R_EPS,
    R_MAX,
    R_MIN,
    S_MAX,
    S_MIN,
    SHORT_STEP,
    LONG_STEP,
    default_device,
)
from .core import next_interval, power_forgetting_curve


def bellman_solver(
    n_iter,
    state_np: np.array,
    transitions_np: list[np.array],
    transition_probs_np: list[np.array],
    costs_np: np.array,
    discount_factor=0.97,
    device=None,
):
    if device is None:
        device = default_device()
    start = time.perf_counter()
    S, D, R = costs_np[0].shape
    dtype = torch.float32
    with torch.inference_mode():
        state = torch.tensor(state_np, requires_grad=False, device=device, dtype=dtype)
        transitions = [
            torch.tensor(
                transition_np, requires_grad=False, device=device, dtype=torch.long
            )
            for transition_np in transitions_np
        ]
        transition_probs = [
            torch.tensor(
                transition_prob_np, requires_grad=False, device=device, dtype=dtype
            )
            for transition_prob_np in transition_probs_np
        ]
        costs = [
            torch.tensor(cost_np, requires_grad=False, device=device, dtype=dtype)
            for cost_np in costs_np
        ]
        it = 0
        cost_diff = 1e9
        while it < n_iter and cost_diff > 0.1:
            it += 1

            action_value = torch.zeros((S, D, R), device=device, dtype=dtype)
            for transition, transition_prob, cost in zip(
                transitions, transition_probs, costs
            ):
                d, s = transition.unbind(dim=-1)
                action_value += transition_prob * (cost + discount_factor * state[d, s])

            optimal_value, optimal_action = action_value.min(dim=-1)
            optimal_value = torch.minimum(state, optimal_value)
            check_interval = 10 if it <= 100 else 25
            if it % check_interval == 0:
                cost_diff = torch.abs(optimal_value - state).max().item()
            state = optimal_value

    end = time.perf_counter()
    print(f"Done in {end - start:.1f} seconds. Iterations: {it}/{n_iter}, cost diff: {cost_diff}")
    return state.cpu().numpy(), optimal_action.cpu().numpy()


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

        self._init_state_spaces()

    def stability_after_success(self, s, d, r, g):
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
        rating = 3
        sinc = np.exp(self.w[17] * (rating - 3 + self.w[18])) * np.power(
            s, -self.w[19]
        )
        return s * sinc

    def init_s(self, rating):
        return np.choose(
            rating - 1,
            np.array(self.w[0:4])
            * np.exp(
                self.w[17]
                * (self.first_rating_offsets + self.first_session_lens * self.w[18])
            ),
        )

    def init_d(self, rating):
        return self.w[4] - np.exp(self.w[5] * (rating - 1)) + 1

    def init_d_with_short_term(self, rating):
        rating_offset = np.choose(rating - 1, self.first_rating_offsets)
        new_d = self.init_d(rating) - self.w[6] * rating_offset
        return np.clip(new_d, 1, 10)

    def linear_damping(self, delta_d, old_d):
        return delta_d * (10 - old_d) / 9

    def next_d(self, d, g):
        delta_d = -self.w[6] * (g - 3)
        new_d = d + self.linear_damping(delta_d, d)
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d.clip(1, 10)

    def mean_reversion(self, init, current):
        return self.w[7] * init + (1 - self.w[7]) * current

    def s2i(self, s):
        result = np.zeros_like(s, dtype=int)
        small_mask = s <= self.s_mid

        result[small_mask] = np.clip(
            np.ceil((np.log(s[small_mask]) - np.log(self.s_min)) / self.short_step).astype(
                int
            ),
            0,
            len(self.s_state_small) - 1,
        )

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

        result[small_mask] = torch.clamp(
            torch.ceil((torch.log(s[small_mask]) - np.log(self.s_min)) / self.short_step).to(
                torch.int
            ),
            0,
            len(self.s_state_small) - 1,
        )

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
        return np.clip(
            np.floor((d - self.d_min) / (self.d_max - self.d_min) * self.d_size).astype(
                int
            ),
            0,
            self.d_size - 1,
        )

    def d2i_torch(self, d):
        return torch.clamp(
            torch.floor((d - self.d_min) / (self.d_max - self.d_min) * self.d_size).to(
                torch.int
            ),
            0,
            self.d_size - 1,
        )

    def r2i(self, r):
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
        cost_max=COST_MAX,
    ):
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

        self.d_min = d_min
        self.d_max = d_max
        self.d_eps = d_eps
        self.d_size = np.ceil((self.d_max - self.d_min) / self.d_eps + 1).astype(int)
        self.d_state = np.linspace(self.d_min, self.d_max, self.d_size)

        self.r_min = r_min
        self.r_max = r_max
        self.r_eps = r_eps
        self.r_size = np.ceil((self.r_max - self.r_min) / self.r_eps + 1).astype(int)
        self.r_state = np.linspace(self.r_min, self.r_max, self.r_size)

        self.cost_matrix = np.full((self.d_size, self.s_size), cost_max)
        self.cost_matrix[:, -1] = 0
        self.retention_matrix = np.zeros_like(self.cost_matrix)

        self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d = np.meshgrid(
            self.s_state, self.d_state, self.r_state
        )

    def solve(self, hyperparams, n_iter=100_000):
        ivl_mesh = next_interval(self.s_state_mesh_3d, self.r_state_mesh_3d, -self.w[20])
        self.r_state_mesh_3d = power_forgetting_curve(
            ivl_mesh, self.s_state_mesh_3d, -self.w[20]
        )
        zeros = np.zeros_like(self.s_state_mesh_3d)
        transitions = []
        transition_probs = []
        costs = []

        next_s_again = self.stability_after_failure(
            self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d
        )
        next_d_again = self.next_d(self.d_state_mesh_3d, 1)
        transitions.append(
            np.stack([self.d2i(next_d_again), self.s2i(next_s_again)], axis=-1)
        )
        transition_probs.append(1.0 - self.r_state_mesh_3d)

        a0, a1, a2, a3, a4, a5, a6, a7, a8 = hyperparams.values()

        assert isinstance(a0, str)
        assert a0 in ["no_log", "log"]
        assert a1 > 0
        assert a2 > 0

        if a0 == "log":
            stability_ratio = np.log1p(self.s_state_mesh_3d) / np.log1p(self.s_max)
        else:
            stability_ratio = self.s_state_mesh_3d / self.s_max

        difficulty_ratio = (self.d_state_mesh_3d - 1) / 10

        stability_modifier = np.power(stability_ratio, a1)
        difficulty_modifier = np.power(difficulty_ratio, a2)

        assert np.max(stability_modifier) <= 1, f"max={np.max(stability_modifier)}"
        assert np.min(stability_modifier) >= 0, f"min={np.min(stability_modifier)}"
        assert np.max(difficulty_modifier) <= 1, f"max={np.max(difficulty_modifier)}"
        assert np.min(difficulty_modifier) >= 0, f"min={np.min(difficulty_modifier)}"

        failure_cost = self.review_costs[0] * (
            a3 + a5 * stability_modifier + a7 * difficulty_modifier
        )

        costs.append(failure_cost)

        for i, (g, review_cost) in enumerate(zip([2, 3, 4], self.review_costs[1:])):
            next_s = self.stability_after_success(
                self.s_state_mesh_3d, self.d_state_mesh_3d, self.r_state_mesh_3d, g
            )
            next_d = self.next_d(self.d_state_mesh_3d, g)
            transitions.append(np.stack([self.d2i(next_d), self.s2i(next_s)], axis=-1))
            transition_probs.append(self.r_state_mesh_3d * self.review_rating_prob[i])

            success_cost = review_cost * (
                a4 + a6 * stability_modifier + a8 * difficulty_modifier
            )

            costs.append(zeros + success_cost)

        assert len(transitions) == len(transition_probs) and len(transitions) == len(costs)

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
        i = 0
        cost_diff = COST_MAX
        next_s_again = self.stability_after_failure(
            self.s_state_mesh_2d, self.d_state_mesh_2d, self.r_state_mesh_2d
        )
        next_d_again = self.next_d(self.d_state_mesh_2d, 1)
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
            optimal_cost = np.minimum(self.cost_matrix, expected_cost)
            cost_diff = self.cost_matrix.sum() - optimal_cost.sum()
            self.cost_matrix = optimal_cost
            i += 1
        return self.cost_matrix, self.r_state_mesh_2d

    def evaluate_r_threshold(self, r_threshold, n_iter=100_000):
        self.r_state_mesh_2d = r_threshold * np.ones_like(self.cost_matrix)
        self.s_state_mesh_2d, self.d_state_mesh_2d = np.meshgrid(
            self.s_state, self.d_state
        )
        ivl_mesh = next_interval(self.s_state_mesh_2d, self.r_state_mesh_2d, -self.w[20])
        self.r_state_mesh_2d = power_forgetting_curve(
            ivl_mesh, self.s_state_mesh_2d, -self.w[20]
        )
        return self._evaluate_policy(n_iter)

    def _get_cost(self, s, d):
        return self.cost_matrix[self.d2i(d), self.s2i(s)]
