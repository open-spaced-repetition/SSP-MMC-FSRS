import numpy as np
import torch

from .config import DEFAULT_W, S_MAX
from .core import next_interval_torch


def max_r_to_reach_next_stability(s, s_next, d, rating, w):
    hard_penalty = torch.where(rating == 2, w[15], 1)
    easy_bonus = torch.where(rating == 4, w[16], 1)
    c = np.exp(w[8]) * (11 - d) * torch.pow(s, -w[9]) * hard_penalty * easy_bonus
    return torch.maximum(
        torch.tensor(0.01, device=s.device),
        1 - torch.log(((s_next / s - 1) / c + 1)) / w[10],
    )


def next_interval_ceil(s, r, decay):
    factor = 0.9 ** (1.0 / decay) - 1.0
    ivl = s / factor * (r ** (1.0 / decay) - 1.0)
    return torch.maximum(torch.tensor(1, device=s.device), torch.ceil(ivl))


def s_max_aware_next_interval(s, d, dr, decay, w, s_max):
    int_base = next_interval_torch(s, dr, decay)
    int_req = next_interval_ceil(
        s,
        max_r_to_reach_next_stability(s, s_max + 1e-3, d, torch.full_like(s, 3), w),
        decay,
    )
    return torch.where(s > s_max, 1e9, torch.minimum(int_base, int_req))


def s_max_aware_fixed_interval(s, d, fixed_interval, decay, w, s_max):
    int_base = fixed_interval
    int_req = next_interval_ceil(
        s,
        max_r_to_reach_next_stability(s, s_max + 1e-3, d, torch.full_like(s, 3), w),
        decay,
    )
    int_base = torch.as_tensor(int_base, device=s.device)
    return torch.where(s > s_max, 1e9, torch.minimum(int_base, int_req))


def _round_half_up_torch(value):
    return torch.floor(value + 0.5)


def _constrain_passing_interval_tensor(interval, minimum):
    rounded = _round_half_up_torch(interval)
    return torch.maximum(rounded, minimum).clamp(min=1.0)


def _anki_sm2_next_interval(
    prev_interval,
    elapsed,
    grade,
    ease,
    *,
    graduating_interval,
    easy_interval,
    easy_bonus,
    hard_interval_factor,
    ease_min,
    ease_max,
):
    ease_clamped = torch.clamp(ease, min=ease_min, max=ease_max)
    current_interval = torch.clamp(prev_interval, min=1.0)
    is_new_card = prev_interval == 0
    new_card_interval = torch.where(
        grade < 4,
        torch.full_like(prev_interval, graduating_interval),
        torch.full_like(prev_interval, easy_interval),
    )

    is_early = elapsed < current_interval
    hard_multiplier = hard_interval_factor
    easy_multiplier = easy_bonus

    if hard_multiplier <= 1.0:
        hard_minimum = torch.zeros_like(current_interval)
    else:
        hard_minimum = current_interval + 1.0
    hard_non_early = _constrain_passing_interval_tensor(
        current_interval * hard_multiplier, hard_minimum
    )
    if hard_multiplier <= 1.0:
        good_minimum = current_interval + 1.0
    else:
        good_minimum = hard_non_early + 1.0
    days_late = torch.clamp(elapsed - current_interval, min=0.0)
    good_non_early = _constrain_passing_interval_tensor(
        (current_interval + days_late / 2.0) * ease_clamped, good_minimum
    )
    easy_non_early = _constrain_passing_interval_tensor(
        (current_interval + days_late) * ease_clamped * easy_multiplier,
        good_non_early + 1.0,
    )

    half_usual = hard_multiplier / 2.0
    hard_early = _constrain_passing_interval_tensor(
        torch.maximum(elapsed * hard_multiplier, current_interval * half_usual),
        torch.zeros_like(current_interval),
    )
    good_early = _constrain_passing_interval_tensor(
        torch.maximum(elapsed * ease_clamped, current_interval),
        torch.zeros_like(current_interval),
    )
    reduced_bonus = easy_multiplier - (easy_multiplier - 1.0) / 2.0
    easy_early = _constrain_passing_interval_tensor(
        torch.maximum(elapsed * ease_clamped, current_interval) * reduced_bonus,
        torch.zeros_like(current_interval),
    )

    hard_interval = torch.where(is_early, hard_early, hard_non_early)
    good_interval = torch.where(is_early, good_early, good_non_early)
    easy_interval_val = torch.where(is_early, easy_early, easy_non_early)

    interval = torch.where(
        is_new_card,
        new_card_interval,
        torch.where(
            grade == 2,
            hard_interval,
            torch.where(grade == 4, easy_interval_val, good_interval),
        ),
    )
    interval = torch.where(grade == 1, torch.ones_like(interval), interval)

    new_ease = ease_clamped
    new_ease = new_ease + -0.2 * (grade == 1).to(new_ease.dtype)
    new_ease = new_ease + -0.15 * (grade == 2).to(new_ease.dtype)
    new_ease = new_ease + 0.15 * (grade == 4).to(new_ease.dtype)
    new_ease = torch.clamp(new_ease, min=ease_min, max=ease_max)

    return interval, new_ease


def make_anki_sm2_policy(w=None, s_max=S_MAX):
    if w is None:
        w = DEFAULT_W
    decay = -w[20]

    def anki_sm2_policy(stability, difficulty, prev_interval, grade, ease):
        graduating_interval = 1.0
        easy_interval = 4.0
        easy_bonus = 1.3
        hard_interval_factor = 1.2
        ease_min = 1.3
        ease_max = 5.5
        # No elapsed parameter in this policy API; assume on-time review.
        elapsed = prev_interval

        interval, new_ease = _anki_sm2_next_interval(
            prev_interval,
            elapsed,
            grade,
            ease,
            graduating_interval=graduating_interval,
            easy_interval=easy_interval,
            easy_bonus=easy_bonus,
            hard_interval_factor=hard_interval_factor,
            ease_min=ease_min,
            ease_max=ease_max,
        )

        final_interval = s_max_aware_fixed_interval(
            stability, difficulty, interval, decay, w, s_max
        ).round()

        return final_interval, new_ease

    return anki_sm2_policy


def make_memrise_policy(w=None, s_max=S_MAX):
    if w is None:
        w = DEFAULT_W
    decay = -w[20]

    def memrise_policy(stability, difficulty, prev_interval, grade):
        device = prev_interval.device
        dtype = prev_interval.dtype

        sequence = torch.tensor([1, 6, 12, 48, 96, 180], device=device, dtype=dtype)

        is_new_card = prev_interval == 0

        prev_expanded = prev_interval.unsqueeze(-1)
        sequence_expanded = sequence.unsqueeze(0).expand_as(
            torch.cat([prev_expanded] * len(sequence), dim=-1)
        )

        distances = torch.abs(prev_expanded - sequence_expanded)

        closest_indices = torch.argmin(distances, dim=-1)

        next_indices = torch.clamp(closest_indices + 1, 0, len(sequence) - 1)

        next_intervals = sequence[next_indices]

        result = torch.where(
            is_new_card,
            torch.ones_like(prev_interval),
            torch.where(
                grade == 1,
                torch.ones_like(prev_interval),
                s_max_aware_fixed_interval(
                    stability, difficulty, next_intervals, decay, w, s_max
                ),
            ),
        )

        return result

    return memrise_policy


def create_fixed_interval_policy(interval, w=None, s_max=S_MAX):
    if w is None:
        w = DEFAULT_W
    decay = -w[20]

    def fixed_policy(stability, difficulty, prev_interval, grade):
        return s_max_aware_fixed_interval(
            stability, difficulty, interval, decay, w, s_max
        )

    return fixed_policy


def create_dr_policy(desired_retention, w=None, s_max=S_MAX):
    if w is None:
        w = DEFAULT_W
    decay = -w[20]

    def dr_policy(stability, difficulty, prev_interval, grade):
        return s_max_aware_next_interval(
            stability, difficulty, desired_retention, decay, w, s_max
        )

    return dr_policy


anki_sm2_policy = make_anki_sm2_policy()
memrise_policy = make_memrise_policy()
