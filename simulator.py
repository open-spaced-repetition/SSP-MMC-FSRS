import random
import numpy as np
import math
from tqdm import trange


DECAY = -0.5
FACTOR = 0.9 ** (1.0 / DECAY) - 1.0


def power_forgetting_curve(t, s, s_max=math.inf):
    return np.where(s > s_max, 1, (1 + FACTOR * t / s) ** DECAY)
    # return (1 + FACTOR * t / s) ** DECAY


def next_interval(s, r):
    ivl = s / FACTOR * (r ** (1.0 / DECAY) - 1.0)
    return np.maximum(1, np.floor(ivl))


columns = [
    "difficulty",
    "stability",
    "retrievability",
    "delta_t",
    "reps",
    "lapses",
    "last_date",
    "due",
    "ivl",
    "cost",
    "cum_cost",
    "rand",
    "rating",
]
col = {key: i for i, key in enumerate(columns)}

DEFAULT_LEARN_COSTS = np.array([33.79, 24.3, 13.68, 6.5])
DEFAULT_REVIEW_COSTS = np.array([23.0, 11.68, 7.33, 5.6])
DEFAULT_FIRST_RATING_PROB = np.array([0.24, 0.094, 0.495, 0.171])
DEFAULT_REVIEW_RATING_PROB = np.array([0.224, 0.631, 0.145])
DEFAULT_FIRST_RATING_OFFSETS = np.array([-0.72, -0.15, -0.01, 0.0])
DEFAULT_FIRST_SESSION_LENS = np.array([2.02, 1.28, 0.81, 0.0])
DEFAULT_FORGET_RATING_OFFSET = -0.28
DEFAULT_FORGET_SESSION_LEN = 1.05


def simulate(
    w,
    policy,
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
    loss_aversion=2.5,
    seed=42,
    s_max=math.inf,
):
    np.random.seed(seed)
    card_table = np.zeros((len(columns), deck_size))
    card_table[col["due"]] = learn_span
    card_table[col["difficulty"]] = 1e-10
    card_table[col["stability"]] = 1e-10
    card_table[col["rating"]] = np.random.choice(
        [1, 2, 3, 4], deck_size, p=first_rating_prob
    )
    card_table[col["rating"]] = card_table[col["rating"]].astype(int)
    card_table[col["cum_cost"]] = 0

    revlogs = {}
    review_cnt_per_day = np.zeros(learn_span)
    learn_cnt_per_day = np.zeros(learn_span)
    memorized_cnt_per_day = np.zeros(learn_span)
    cost_per_day = np.zeros(learn_span)

    def stability_after_success(s, r, d, rating):
        hard_penalty = np.where(rating == 2, w[15], 1)
        easy_bonus = np.where(rating == 4, w[16], 1)
        return np.maximum(
            0.01,
            s
            * (
                1
                + np.exp(w[8])
                * (11 - d)
                * np.power(s, -w[9])
                * (np.exp((1 - r) * w[10]) - 1)
                * hard_penalty
                * easy_bonus
            ),
        )

    def stability_after_failure(s, r, d):
        return np.maximum(
            0.01,
            np.minimum(
                w[11]
                * np.power(d, -w[12])
                * (np.power(s + 1, w[13]) - 1)
                * np.exp((1 - r) * w[14]),
                s / np.exp(w[17] * w[18]),
            ),
        )

    def stability_short_term(s, init_rating=None):
        if init_rating is not None:
            rating_offset = np.choose(init_rating - 1, first_rating_offset)
            session_len = np.choose(init_rating - 1, first_session_len)
        else:
            rating_offset = forget_rating_offset
            session_len = forget_session_len
        new_s = s * np.exp(w[17] * (rating_offset + session_len * w[18]))
        return new_s

    def init_d(rating):
        return w[4] - np.exp(w[5] * (rating - 1)) + 1

    def init_d_with_short_term(rating):
        rating_offset = np.choose(rating - 1, first_rating_offset)
        new_d = init_d(rating) - w[6] * rating_offset
        return np.clip(new_d, 1, 10)

    def linear_damping(delta_d, old_d):
        return delta_d * (10 - old_d) / 9

    def next_d(d, rating):
        delta_d = -w[6] * (rating - 3)
        new_d = d + linear_damping(delta_d, d)
        new_d = mean_reversion(init_d(4), new_d)
        return np.clip(new_d, 1, 10)

    def mean_reversion(init, current):
        return w[7] * init + (1 - w[7]) * current

    for today in trange(learn_span, position=1, leave=False):
        has_learned = card_table[col["stability"]] > 1e-10
        card_table[col["delta_t"]][has_learned] = (
            today - card_table[col["last_date"]][has_learned]
        )
        card_table[col["retrievability"]][has_learned] = power_forgetting_curve(
            card_table[col["delta_t"]][has_learned],
            card_table[col["stability"]][has_learned],
            s_max,
        )
        card_table[col["cost"]] = 0
        need_review = card_table[col["due"]] <= today
        card_table[col["rand"]][need_review] = np.random.rand(np.sum(need_review))
        forget = card_table[col["rand"]] > card_table[col["retrievability"]]
        card_table[col["rating"]][need_review & forget] = 1
        card_table[col["rating"]][need_review & ~forget] = np.random.choice(
            [2, 3, 4], np.sum(need_review & ~forget), p=review_rating_prob
        )
        card_table[col["cost"]][need_review] = np.choose(
            card_table[col["rating"]][need_review].astype(int) - 1,
            review_costs,
        )
        card_table[col["cost"]][need_review & forget] *= loss_aversion
        true_review = (
            need_review
            & (np.cumsum(card_table[col["cost"]]) <= max_cost_perday)
            & (np.cumsum(need_review) <= review_limit_perday)
        )
        card_table[col["last_date"]][true_review] = today

        card_table[col["lapses"]][true_review & forget] += 1
        card_table[col["reps"]][true_review & ~forget] += 1

        card_table[col["stability"]][true_review & forget] = stability_after_failure(
            card_table[col["stability"]][true_review & forget],
            card_table[col["retrievability"]][true_review & forget],
            card_table[col["difficulty"]][true_review & forget],
        )
        card_table[col["stability"]][true_review & forget] = stability_short_term(
            card_table[col["stability"]][true_review & forget]
        )
        card_table[col["stability"]][true_review & ~forget] = stability_after_success(
            card_table[col["stability"]][true_review & ~forget],
            card_table[col["retrievability"]][true_review & ~forget],
            card_table[col["difficulty"]][true_review & ~forget],
            card_table[col["rating"]][true_review & ~forget],
        )

        card_table[col["difficulty"]][true_review] = next_d(
            card_table[col["difficulty"]][true_review],
            card_table[col["rating"]][true_review],
        )
        card_table[col["difficulty"]][true_review & forget] = np.clip(
            card_table[col["difficulty"]][true_review & forget]
            - (w[6] * forget_rating_offset),
            1,
            10,
        )

        need_learn = card_table[col["stability"]] == 1e-10
        card_table[col["cost"]][need_learn] = np.choose(
            card_table[col["rating"]][need_learn].astype(int) - 1,
            learn_costs,
        )
        true_learn = (
            need_learn
            & (np.cumsum(card_table[col["cost"]]) <= max_cost_perday)
            & (np.cumsum(need_learn) <= learn_limit_perday)
        )
        card_table[col["last_date"]][true_learn] = today
        card_table[col["stability"]][true_learn] = np.choose(
            card_table[col["rating"]][true_learn].astype(int) - 1, w[:4]
        )
        card_table[col["stability"]][true_learn] = stability_short_term(
            card_table[col["stability"]][true_learn],
            init_rating=card_table[col["rating"]][true_learn].astype(int),
        )
        card_table[col["difficulty"]][true_learn] = init_d_with_short_term(
            card_table[col["rating"]][true_learn].astype(int)
        )

        card_table[col["ivl"]][true_review | true_learn] = policy(
            card_table[col["stability"]][true_review | true_learn],
            card_table[col["difficulty"]][true_review | true_learn],
        )
        card_table[col["due"]][true_review | true_learn] = (
            today + card_table[col["ivl"]][true_review | true_learn]
        )

        revlogs[today] = {
            "card_id": np.where(true_review | true_learn)[0],
            "rating": card_table[col["rating"]][true_review | true_learn],
        }

        np.set_printoptions(edgeitems=20)
        card_table[col["cum_cost"]][true_review | true_learn] += card_table[col["cost"]][true_review | true_learn]
        review_cnt_per_day[today] = np.sum(true_review)
        learn_cnt_per_day[today] = np.sum(true_learn)
        # memorized_cnt_per_day[today] = card_table[col["retrievability"]].sum()
        reached_target = card_table[col["stability"]] > s_max
        memorized_cnt_per_day[today] = reached_target.sum()
        # cost_per_day[today] = card_table[col["cost"]][reached_target & (true_review | true_learn)].sum()
        cost_per_day[today] = card_table[col["cost"]][true_review | true_learn].sum()
        cost_reached = card_table[col["cum_cost"]][reached_target].mean()

    if len(card_table[col["cum_cost"]][reached_target]) > 0:
        z = card_table[col["cum_cost"]][reached_target]
        # print("here", z.min(), z.max())
        # np.set_printoptions(linewidth=200)
        # np.set_printoptions(edgeitems=20)
        # print(card_table[col["reps"]][reached_target])
        # print(card_table[col["lapses"]][reached_target])
        # print(card_table[col["cum_cost"]][reached_target])
        # print(card_table[col["cum_cost"]][reached_target] / card_table[col["reps"]][reached_target])
        # print(card_table[col["cost"]][reached_target])
        # print(cost_reached)
        # print(cost_per_day[today])
        # print(card_table[col["cost"]][reached_target & (true_review | true_learn)].sum())
    return (
        card_table,
        review_cnt_per_day,
        learn_cnt_per_day,
        memorized_cnt_per_day,
        cost_per_day,
        cost_reached,
        revlogs,
    )
