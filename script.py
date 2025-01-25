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

s_min = 0.1
s_max = 365 * 3
short_step = np.log(2) / 15
long_step = 10
# use long step when short step exceeds long step
s_mid = min(long_step / (1 - np.exp(-short_step)), s_max)
print(f"s_mid={s_mid:.2f}")
# Adaptive step size
s_state_small = np.exp(np.arange(np.log(s_min), np.log(s_mid), short_step))
s_state_large = np.arange(max(s_state_small) + long_step, s_max, long_step)
s_state = np.concatenate([s_state_small, s_state_large])
# np.set_printoptions(suppress=True, precision=3, threshold=10000)
# print(s_state)
s_size = len(s_state)

d_min = 1
d_max = 10
d_eps = 1 / 3
d_size = np.ceil((d_max - d_min) / d_eps + 1).astype(int)

r_min = 0.70
r_max = 0.97
r_eps = 0.01
r_size = np.ceil((r_max - r_min) / r_eps + 1).astype(int)

cost_matrix = np.zeros((d_size, s_size))
cost_matrix.fill(1e8)
cost_matrix[:, -1] = 0
action_matrix = np.zeros((d_size, s_size))

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


def stability_after_success(s, d, r, g):
    return s * (
        1
        + np.exp(w[8])
        * (11 - d)
        * np.power(s, -w[9])
        * (np.exp((1 - r) * w[10]) - 1)
        * (w[15] if g == 2 else 1)
        * (w[16] if g == 4 else 1)
    )


def stability_after_failure(s, d, r):
    return np.maximum(
        s_min,
        np.minimum(
            w[11]
            * np.power(d, -w[12])
            * (np.power(s + 1, w[13]) - 1)
            * np.exp((1 - r) * w[14]),
            s / np.exp(w[17] * w[18]),
        ),
    )


def stability_short_term(s):
    return s * np.exp(w[17] * (forget_rating_offset + forget_session_len * w[18]))


def init_s(rating):
    return np.choose(
        rating - 1,
        np.array(w[0:4])
        * np.exp(w[17] * (first_rating_offsets + first_session_lens * w[18])),
    )


def init_d(rating):
    return w[4] - np.exp(w[5] * (rating - 1)) + 1


def init_d_with_short_term(rating):
    rating_offset = np.choose(rating - 1, first_rating_offsets)
    new_d = init_d(rating) - w[6] * rating_offset
    return np.clip(new_d, 1, 10)


def linear_damping(delta_d, old_d):
    return delta_d * (10 - old_d) / 9


def next_d(d, g):
    delta_d = -w[6] * (g - 3)
    new_d = d + linear_damping(delta_d, d)
    new_d = mean_reversion(init_d(4), new_d)
    return new_d.clip(1, 10)


def mean_reversion(init, current):
    return w[7] * init + (1 - w[7]) * current


# stability to indexes
def s2i(s):
    # Vectorized version for array input
    result = np.zeros_like(s, dtype=int)
    small_mask = s <= s_mid

    # Handle small values (logarithmic scale)
    result[small_mask] = np.clip(
        np.floor((np.log(s[small_mask]) - np.log(s_min)) / short_step).astype(int),
        0,
        len(s_state_small) - 1,
    )

    # Handle large values (linear scale)
    result[~small_mask] = len(s_state_small) + np.clip(
        np.floor((s[~small_mask] - s_state_small[-1] - long_step) / long_step).astype(
            int
        ),
        0,
        len(s_state_large) - 1,
    )

    return result


# difficulty to indexes
def d2i(d):
    return np.clip(
        np.floor((d - d_min) / (d_max - d_min) * d_size).astype(int), 0, d_size - 1
    )


# retention to indexes
def r2i(r):
    return np.clip(
        np.floor((r - r_min) / (r_max - r_min) * r_size).astype(int), 0, r_size - 1
    )


# indexes to cost
def i2c(s, d):
    return cost_matrix[d2i(d), s2i(s)]


i = 0
cost_diff = 1000
n_iter = 10000

start = time.time()

d_state = np.linspace(d_min, d_max, d_size)
r_state = np.linspace(r_min, r_max, r_size)
print(f"Min(s_state)={min(s_state)}, max(s_state)={max(s_state)}, N={len(s_state)}")
print(f"Min(d_state)={min(d_state)}, max(d_state)={max(d_state)}, N={len(d_state)}")
print(f"Min(r_state)={min(r_state)}, max(r_state)={max(r_state)}, N={len(r_state)}")
print("")

s_state_mesh, d_state_mesh, r_state_mesh = np.meshgrid(s_state, d_state, r_state)

ivl_mesh = next_interval(s_state_mesh, r_state_mesh)
r_state_mesh = power_forgetting_curve(ivl_mesh, s_state_mesh)
retention_matrix = np.zeros_like(cost_matrix)

while i < n_iter and cost_diff > 1e-4 * s_size * d_size:
    next_stability_after_again = stability_short_term(
        stability_after_failure(s_state_mesh, d_state_mesh, r_state_mesh)
    )
    next_difficulty_after_again = next_d(d_state_mesh, 1)
    next_cost_after_again = (
        i2c(next_stability_after_again, next_difficulty_after_again)
        + review_costs[0] * loss_aversion
    )

    next_stability_after_hard = stability_after_success(
        s_state_mesh, d_state_mesh, r_state_mesh, 2
    )
    next_difficulty_after_hard = next_d(d_state_mesh, 2)
    next_cost_after_hard = (
        i2c(next_stability_after_hard, next_difficulty_after_hard) + review_costs[1]
    )

    next_stability_after_good = stability_after_success(
        s_state_mesh, d_state_mesh, r_state_mesh, 3
    )
    next_difficulty_after_good = next_d(d_state_mesh, 3)
    next_cost_after_good = (
        i2c(next_stability_after_good, next_difficulty_after_good) + review_costs[2]
    )

    next_stability_after_easy = stability_after_success(
        s_state_mesh, d_state_mesh, r_state_mesh, 4
    )
    next_difficulty_after_easy = next_d(d_state_mesh, 4)
    next_cost_after_easy = (
        i2c(next_stability_after_easy, next_difficulty_after_easy) + review_costs[3]
    )

    expected_cost = (
        r_state_mesh
        * (
            review_rating_prob[0] * next_cost_after_hard
            + review_rating_prob[1] * next_cost_after_good
            + review_rating_prob[2] * next_cost_after_easy
        )
        + (1 - r_state_mesh) * next_cost_after_again
    )
    # update cost matrix
    optimal_cost = np.minimum(cost_matrix, expected_cost.min(axis=2))
    cost_diff = np.sum(cost_matrix - optimal_cost)
    cost_matrix = optimal_cost

    last_retention_matrix = retention_matrix
    retention_matrix = r_state[np.argmin(expected_cost, axis=2)]
    r_diff = np.abs(retention_matrix - last_retention_matrix).sum()
    if i % 10 == 0:
        print(
            f"iteration {i:>5}, cost diff {cost_diff:.2f}, retention diff {r_diff:.2f}, elapsed time {time.time() - start:.1f}s"
        )
    i += 1

end = time.time()
print(f"Time: {end - start:.2f}s")
init_stabilities = init_s(np.arange(1, 5))
init_difficulties = init_d_with_short_term(np.arange(1, 5))
init_cost = cost_matrix[d2i(init_difficulties), s2i(init_stabilities)]
avg_cost = init_cost @ first_rating_prob
print(f"Average cost: {avg_cost:.2f}")
avg_retention = retention_matrix.mean()
print(f"Average retention: {avg_retention:.2f}")

s_state_mesh_2d, d_state_mesh_2d = np.meshgrid(s_state, d_state)
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
    d_index = d2i(d)
    s_index = s2i(s)
    # Handle array inputs by checking each element
    mask = (d_index >= d_size) | (s_index >= s_size - 1)
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
        s_max=s_max,
    )

    def moving_average(data, window_size=365 // 20):
        weights = np.ones(window_size) / window_size
        return np.convolve(data, weights, mode="valid")

    return (
        moving_average(review_cnt_per_day),
        moving_average(cost_per_day),
        moving_average(memorized_cnt_per_day),
    )


def plot_simulation(policy, title):
    review_cnt_per_day, cost_per_day, memorized_cnt_per_day = simulate_policy(policy)
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
        memorized_cnt_per_day, label=f"Total Memorized: {memorized_cnt_per_day[-1]:.2f}"
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
            d_index, s_index = d2i(init_difficulties[rating - 1]), s2i(
                init_stabilities[rating - 1]
            )
            cur_s = s_state[s_index]
            cur_d = d_state[d_index]
        else:
            optimal_r = retention_matrix[d_index, s_index]
            s_list.append(cur_s)
            r_list.append(optimal_r)
            ivl_list.append(next_interval(cur_s, optimal_r))
            cur_s = stability_after_success(cur_s, cur_d, optimal_r, rating)
            cur_d = next_d(cur_d, rating)
            d_index, s_index = d2i(cur_d), s2i(cur_s)

        if cur_s > s_max:
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

r_range = np.linspace(r_min, r_max, 10)

for r in r_range:
    print("--------------------------------")
    start = time.time()
    cost_matrix = np.zeros((d_size, s_size))
    cost_matrix.fill(1000)
    cost_matrix[:, -1] = 0
    s_state_mesh, d_state_mesh = np.meshgrid(s_state, d_state)
    r_state_mesh = r * np.ones((d_size, s_size))
    ivl_mesh = next_interval(s_state_mesh, r_state_mesh)
    r_state_mesh = power_forgetting_curve(ivl_mesh, s_state_mesh)

    i = 0
    cost_diff = 10000
    n_iter = 1000
    while i < n_iter and cost_diff > 1e-4 * s_size * d_size:
        next_stability_after_again = stability_short_term(
            stability_after_failure(s_state_mesh, d_state_mesh, r_state_mesh)
        )
        next_difficulty_after_again = next_d(d_state_mesh, 1)
        next_cost_after_again = (
            i2c(next_stability_after_again, next_difficulty_after_again)
            + review_costs[0] * loss_aversion
        )

        next_stability_after_hard = stability_after_success(
            s_state_mesh, d_state_mesh, r_state_mesh, 2
        )
        next_difficulty_after_hard = next_d(d_state_mesh, 2)
        next_cost_after_hard = (
            i2c(next_stability_after_hard, next_difficulty_after_hard) + review_costs[1]
        )

        next_stability_after_good = stability_after_success(
            s_state_mesh, d_state_mesh, r_state_mesh, 3
        )
        next_difficulty_after_good = next_d(d_state_mesh, 3)
        next_cost_after_good = (
            i2c(next_stability_after_good, next_difficulty_after_good) + review_costs[2]
        )

        next_stability_after_easy = stability_after_success(
            s_state_mesh, d_state_mesh, r_state_mesh, 4
        )
        next_difficulty_after_easy = next_d(d_state_mesh, 4)
        next_cost_after_easy = (
            i2c(next_stability_after_easy, next_difficulty_after_easy) + review_costs[3]
        )

        expected_cost = (
            r_state_mesh
            * (
                review_rating_prob[0] * next_cost_after_hard
                + review_rating_prob[1] * next_cost_after_good
                + review_rating_prob[2] * next_cost_after_easy
            )
            + (1 - r_state_mesh) * next_cost_after_again
        )
        # update cost matrix
        optimal_cost = np.minimum(cost_matrix, expected_cost)
        cost_diff = cost_matrix.sum() - optimal_cost.sum()
        cost_matrix = optimal_cost
        i += 1
    end = time.time()
    print(f"Time: {end - start:.2f}s, Iterations: {i}")
    init_stabilities = init_s(np.arange(1, 5))
    init_difficulties = init_d_with_short_term(np.arange(1, 5))
    init_cost = cost_matrix[d2i(init_difficulties), s2i(init_stabilities)]
    avg_cost = init_cost @ first_rating_prob
    avg_retention = r_state_mesh.mean()
    print(f"Desired Retention: {r * 100:.2f}%")
    print(f"True Retention: {avg_retention * 100:.2f}%")
    costs.append(avg_cost)
    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(s_state_mesh, d_state_mesh, cost_matrix, cmap="viridis")
    ax.set_xlabel("Stability")
    ax.set_ylabel("Difficulty")
    ax.set_zlabel("Cost")
    ax.set_title(f"Desired Retention: {r * 100:.2f}%, Avg Cost: {avg_cost:.2f}")
    ax.set_box_aspect(None, zoom=0.8)
    ax = fig.add_subplot(122, projection="3d")
    ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, r_state_mesh, cmap="viridis")
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
