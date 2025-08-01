import numpy as np
import math
import torch
from tqdm import trange


def power_forgetting_curve(t, s, decay, s_max=math.inf):
    factor = 0.9 ** (1.0 / decay) - 1.0
    return np.where(s > s_max, 1, (1 + factor * t / s) ** decay)


def power_forgetting_curve_torch(t, s, decay, s_max=math.inf):
    factor = 0.9 ** (1.0 / decay) - 1.0
    return torch.where(s > s_max, 1, (1 + factor * t / s) ** decay)


def next_interval(s, r, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    ivl = s / factor * (r ** (1.0 / decay) - 1.0)
    return np.maximum(1, np.floor(ivl))


def next_interval_torch(s, r, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    ivl = s / factor * (r ** (1.0 / decay) - 1.0)
    return torch.maximum(torch.ones_like(ivl), torch.floor(ivl))


DEFAULT_LEARN_COSTS = np.array([33.79, 24.3, 13.68, 6.5])
DEFAULT_REVIEW_COSTS = np.array([23.0, 11.68, 7.33, 5.6])
DEFAULT_FIRST_RATING_PROB = np.array([0.24, 0.094, 0.495, 0.171])
DEFAULT_REVIEW_RATING_PROB = np.array([0.224, 0.631, 0.145])
DEFAULT_FIRST_RATING_OFFSETS = np.array([-0.72, -0.15, -0.01, 0.0])
DEFAULT_FIRST_SESSION_LENS = np.array([2.02, 1.28, 0.81, 0.0])
DEFAULT_FORGET_RATING_OFFSET = -0.28
DEFAULT_FORGET_SESSION_LEN = 1.05


@torch.inference_mode()
def simulate(
    parallel,
    w,
    policy,
    device,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=86400 / 4,
    learn_limit_perday=10,
    review_limit_perday=9999,
    learn_costs=DEFAULT_LEARN_COSTS,
    review_costs=DEFAULT_REVIEW_COSTS,
    first_rating_prob=DEFAULT_FIRST_RATING_PROB,
    review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
    first_rating_offset=DEFAULT_FIRST_RATING_OFFSETS,
    first_session_len=DEFAULT_FIRST_SESSION_LENS,
    forget_rating_offset=DEFAULT_FORGET_RATING_OFFSET,
    forget_session_len=DEFAULT_FORGET_SESSION_LEN,
    seed=42,
    s_max=math.inf,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    due = torch.full((parallel, deck_size), learn_span, device=device)
    difficulty = torch.full_like(due, 1e-10)
    stability = torch.full_like(due, 1e-10, dtype=torch.float64)
    retrievability = torch.full_like(due, 0)
    delta_t = torch.zeros_like(due)
    reps = torch.zeros_like(due)
    lapses = torch.zeros_like(due)
    last_date = torch.zeros_like(due)
    ivl = torch.zeros_like(due)
    cost = torch.zeros_like(due)
    ratings_np = np.random.choice([1, 2, 3, 4], deck_size, p=first_rating_prob)
    rating = torch.tensor(ratings_np, dtype=torch.int32, device=device)
    review_cnt_per_day = torch.zeros((parallel, learn_span), device=device)
    learn_cnt_per_day = torch.zeros_like(review_cnt_per_day)
    memorized_cnt_per_day = torch.zeros_like(review_cnt_per_day)
    cost_per_day = torch.zeros_like(review_cnt_per_day)
    pass_ratings_tensor = torch.tensor([2, 3, 4], device=device)
    weights_tensor = torch.tensor(review_rating_prob, device=device)
    review_costs_tensor = torch.tensor(review_costs, device=device)
    learn_costs_tensor = torch.tensor(learn_costs, device=device)
    w_4_tensor = torch.tensor(w[:4], device=device)
    first_rating_offset_tensor = torch.tensor(first_rating_offset, device=device)

    def stability_after_success(s, r, d, rating):
        hard_penalty = torch.where(rating == 2, w[15], 1)
        easy_bonus = torch.where(rating == 4, w[16], 1)
        return torch.maximum(
            torch.tensor(0.01, device=s.device),
            s
            * (
                1
                + np.exp(w[8])
                * (11 - d)
                * torch.pow(s, -w[9])
                * (torch.exp((1 - r) * w[10]) - 1)
                * hard_penalty
                * easy_bonus
            ),
        )

    def stability_after_failure(s, r, d):
        return torch.maximum(
            torch.tensor(0.01, device=s.device),
            torch.minimum(
                w[11]
                * torch.pow(d, -w[12])
                * (torch.pow(s + 1, w[13]) - 1)
                * torch.exp((1 - r) * w[14]),
                s / np.exp(w[17] * w[18]),
            ),
        )

    def stability_short_term(s):
        rating = 3
        new_s = s * np.exp(w[17] * (rating - 3 + w[18])) * torch.pow(s, -w[19])
        return new_s

    def init_d(rating):
        return w[4] - torch.exp(w[5] * (rating - 1)) + 1

    def init_d_with_short_term(rating):
        rating_offset = first_rating_offset_tensor[rating - 1]
        new_d = init_d(rating) - w[6] * rating_offset
        return torch.clamp(new_d, min=1, max=10)

    def linear_damping(delta_d, old_d):
        return delta_d * (10 - old_d) / 9

    def next_d(d, rating):
        delta_d = -w[6] * (rating - 3)
        new_d = d + linear_damping(delta_d, d)
        new_d = mean_reversion(init_d(torch.full_like(d, 4)), new_d)
        return torch.clamp(new_d, min=1, max=10)

    def mean_reversion(init, current):
        return w[7] * init + (1 - w[7]) * current

    for today in trange(learn_span, position=1, leave=False):
        has_learned = stability > 1e-10
        delta_t = torch.where(has_learned, today - last_date, delta_t)
        retrievability = torch.where(
            ~has_learned,
            retrievability,
            power_forgetting_curve_torch(delta_t, stability, -w[20], s_max),
        )
        cost.zero_()
        need_review = due <= today
        rand = torch.rand(need_review.shape, device=device)
        forget = rand > retrievability
        rating = torch.where(need_review & forget, 1, rating)
        ratings_ind_sample = torch.multinomial(
            weights_tensor, num_samples=due.numel(), replacement=True
        ).view_as(due)
        ratings_sample = pass_ratings_tensor[ratings_ind_sample]
        rating = torch.where(need_review & ~forget, ratings_sample, rating)
        cost = torch.where(~need_review, cost, review_costs_tensor[rating - 1])

        true_review = (
            need_review
            & (torch.cumsum(cost, dim=-1) <= max_cost_perday)
            & (torch.cumsum(need_review, dim=-1) <= review_limit_perday)
        )
        last_date = torch.where(true_review, today, last_date)
        lapses = lapses + (true_review & forget)
        reps = reps + (true_review & ~forget)
        stability = torch.where(
            true_review & forget,
            stability_after_failure(stability, retrievability, difficulty),
            stability,
        )
        stability = torch.where(
            true_review & forget, stability_short_term(stability), stability
        )
        stability = torch.where(
            true_review & ~forget,
            stability_after_success(stability, retrievability, difficulty, rating),
            stability,
        )
        difficulty = torch.where(true_review, next_d(difficulty, rating), difficulty)
        difficulty = torch.where(
            true_review & forget,
            torch.clamp(difficulty - (w[6] * forget_rating_offset), min=1, max=10),
            difficulty,
        )

        need_learn = stability == 1e-10
        cost = torch.where(~need_learn, cost, learn_costs_tensor[rating - 1])
        true_learn = (
            need_learn
            & (torch.cumsum(cost, dim=-1) <= max_cost_perday)
            & (torch.cumsum(need_learn, dim=-1) <= learn_limit_perday)
        )
        last_date = torch.where(true_learn, today, last_date)
        stability = torch.where(true_learn, w_4_tensor[rating - 1], stability)
        difficulty = torch.where(true_learn, init_d_with_short_term(rating), difficulty)
        ivl = torch.where(true_review | true_learn, policy(stability, difficulty), ivl)
        due = torch.where(true_review | true_learn, today + ivl, due)

        review_cnt_per_day[:, today] = true_review.sum(dim=-1)
        learn_cnt_per_day[:, today] = true_learn.sum(dim=-1)
        memorized_cnt_per_day[:, today] = retrievability.sum(dim=-1)
        cost_per_day[:, today] = (cost * (true_review | true_learn)).sum(dim=-1)

    return (
        review_cnt_per_day.cpu().numpy(),
        learn_cnt_per_day.cpu().numpy(),
        memorized_cnt_per_day.cpu().numpy(),
        cost_per_day.cpu().numpy(),
    )
