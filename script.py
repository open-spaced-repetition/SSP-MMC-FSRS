import time
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

again_cost = 25
hard_cost = 14
good_cost = 10
easy_cost = 6
first_rating_prob = np.array([0.15, 0.2, 0.6, 0.05])
review_rating_prob = np.array([0.3, 0.6, 0.1])

s_min = 0.1
s_max = 365 * 3
short_step = np.log(2) / 15
long_step = 1
s_mid = long_step / (1 - np.exp(-short_step))
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

DECAY = -0.5
FACTOR = 0.9 ** (1.0 / DECAY) - 1.0


def power_forgetting_curve(t, s):
    return (1 + FACTOR * t / s) ** DECAY


def next_interval(s, r):
    ivl = s / FACTOR * (r ** (1.0 / DECAY) - 1.0)
    return np.maximum(1, np.floor(ivl))


cost_matrix = np.zeros((d_size, s_size))
cost_matrix.fill(1e8)
cost_matrix[:, -1] = 0
action_matrix = np.zeros((d_size, s_size))

w = [
    0.5701,
    1.4436,
    4.1386,
    10.9355,
    5.1443,
    1.2006,
    0.8627,
    0.0362,
    1.629,
    0.1342,
    1.0166,
    2.1174,
    0.0839,
    0.3204,
    1.4676,
    0.219,
    2.8237,
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
    return np.minimum(
        w[11]
        * np.power(d, -w[12])
        * (np.power(s + 1, w[13]) - 1)
        * np.exp((1 - r) * w[14]),
        s,
    )


def mean_reversion(init, current):
    return (w[7] * init + (1 - w[7]) * current).clip(1, 10)


def next_difficulty(d, g):
    return mean_reversion(w[4], d - w[6] * (g - 3))


# stability to index (logarithmic)
def s2i(s):
    index = np.clip(
        np.floor(
            (np.log(s) - np.log(s_min)) / (np.log(s_max) - np.log(s_min)) * s_size
        ).astype(int),
        0,
        s_size - 1,
    )
    return index


# difficulty to index
def d2i(d):
    return np.clip(
        np.floor((d - d_min) / (d_max - d_min) * d_size).astype(int), 0, d_size - 1
    )


# retention to index
def r2i(r):
    return np.clip(
        np.floor((r - r_min) / (r_max - r_min) * r_size).astype(int), 0, r_size - 1
    )


# indexes to cost


def i2c(s, d):
    return cost_matrix[d2i(d), s2i(s)]


i = 0
diff = 1000
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

while i < n_iter and diff > 1e-4 * s_size * d_size:
    next_stability_after_again = stability_after_failure(
        s_state_mesh, d_state_mesh, r_state_mesh
    )
    next_difficulty_after_again = next_difficulty(d_state_mesh, 1)
    next_cost_after_again = (
        i2c(next_stability_after_again, next_difficulty_after_again) + again_cost
    )

    next_stability_after_hard = stability_after_success(
        s_state_mesh, d_state_mesh, r_state_mesh, 2
    )
    next_difficulty_after_hard = next_difficulty(d_state_mesh, 2)
    next_cost_after_hard = (
        i2c(next_stability_after_hard, next_difficulty_after_hard) + hard_cost
    )

    next_stability_after_good = stability_after_success(
        s_state_mesh, d_state_mesh, r_state_mesh, 3
    )
    next_difficulty_after_good = next_difficulty(d_state_mesh, 3)
    next_cost_after_good = (
        i2c(next_stability_after_good, next_difficulty_after_good) + good_cost
    )

    next_stability_after_easy = stability_after_success(
        s_state_mesh, d_state_mesh, r_state_mesh, 4
    )
    next_difficulty_after_easy = next_difficulty(d_state_mesh, 4)
    next_cost_after_easy = (
        i2c(next_stability_after_easy, next_difficulty_after_easy) + easy_cost
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
    retention_matrix = r_state[np.argmin(expected_cost, axis=2)]
    diff = cost_matrix.sum() - optimal_cost.sum()
    cost_matrix = optimal_cost
    if i % 10 == 0:
        print(
            f"iteration {i:>5}, diff {diff:.2f}, elapsed time {time.time() - start:.1f}s"
        )
    i += 1

end = time.time()
print(f"Time: {end - start:.2f}s")
init_stability = np.array(w[0:4])
init_difficulty = np.array([w[4] - (3 - g) * w[5] for g in range(1, 5)])
init_cost = cost_matrix[d2i(init_difficulty), s2i(init_stability)]
avg_cost = init_cost @ first_rating_prob
print(f"Average cost: {avg_cost:.2f}")
avg_retention = retention_matrix.mean()
print(f"Average retention: {avg_retention:.2f}")

s_state_mesh_2d, d_state_mesh_2d = np.meshgrid(s_state, d_state)
fig = plt.figure(figsize=(16, 8.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, cost_matrix, cmap="viridis")
ax.set_xlabel("Stability")
ax.set_ylabel("Difficulty")
ax.set_zlabel("Cost")
ax.set_title(f"Avg Init Cost: {avg_cost:.2f}")
ax.set_box_aspect(None, zoom=0.8)
ax = fig.add_subplot(122, projection="3d")
ax.plot_surface(s_state_mesh_2d, d_state_mesh_2d, retention_matrix, cmap="viridis")
ax.set_xlabel("Stability")
ax.set_ylabel("Difficulty")
ax.set_zlabel("Retention")
ax.set_title(f"Avg Retention: {avg_retention:.2f}")
ax.set_box_aspect(None, zoom=0.8)
plt.tight_layout()
plt.savefig("./plot/SSP-MMC.png")

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
    diff = 10000
    n_iter = 1000
    while i < n_iter and diff > 1e-4 * s_size * d_size:
        next_stability_after_again = stability_after_failure(
            s_state_mesh, d_state_mesh, r_state_mesh
        )
        next_difficulty_after_again = next_difficulty(d_state_mesh, 1)
        next_cost_after_again = (
            i2c(next_stability_after_again, next_difficulty_after_again) + again_cost
        )

        next_stability_after_hard = stability_after_success(
            s_state_mesh, d_state_mesh, r_state_mesh, 2
        )
        next_difficulty_after_hard = next_difficulty(d_state_mesh, 2)
        next_cost_after_hard = (
            i2c(next_stability_after_hard, next_difficulty_after_hard) + hard_cost
        )

        next_stability_after_good = stability_after_success(
            s_state_mesh, d_state_mesh, r_state_mesh, 3
        )
        next_difficulty_after_good = next_difficulty(d_state_mesh, 3)
        next_cost_after_good = (
            i2c(next_stability_after_good, next_difficulty_after_good) + good_cost
        )

        next_stability_after_easy = stability_after_success(
            s_state_mesh, d_state_mesh, r_state_mesh, 4
        )
        next_difficulty_after_easy = next_difficulty(d_state_mesh, 4)
        next_cost_after_easy = (
            i2c(next_stability_after_easy, next_difficulty_after_easy) + easy_cost
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
        diff = cost_matrix.sum() - optimal_cost.sum()
        cost_matrix = optimal_cost
        i += 1
    end = time.time()
    print(f"Time: {end - start:.2f}s, Iterations: {i}")
    init_stability = np.array(w[0:4])
    init_difficulty = np.array([w[4] - (3 - g) * w[5] for g in range(1, 5)])
    init_cost = cost_matrix[d2i(init_difficulty), s2i(init_stability)]
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
