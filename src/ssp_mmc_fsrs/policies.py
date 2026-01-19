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


def make_anki_sm2_policy(w=None, s_max=S_MAX):
    if w is None:
        w = DEFAULT_W
    decay = -w[20]

    def anki_sm2_policy(stability, difficulty, prev_interval, grade, ease):
        graduating_interval = 1.0
        easy_interval = 4.0
        easy_bonus = 1.3
        hard_interval_factor = 1.2
        new_interval = 0.0
        interval_multiplier = 1.0

        new_ease = torch.where(
            grade == 1,
            ease - 0.2,
            torch.where(
                grade == 2,
                ease - 0.15,
                torch.where(grade == 4, ease + 0.15, ease),
            ),
        )
        new_ease = torch.clamp(new_ease, 1.3, 5.5)

        is_new_card = prev_interval == 0

        new_card_interval = torch.where(
            grade < 4,
            torch.full_like(prev_interval, graduating_interval),
            torch.full_like(prev_interval, easy_interval),
        )

        days_late = torch.zeros_like(prev_interval)
        elapsed = prev_interval + days_late

        existing_card_interval = torch.where(
            grade == 1,
            prev_interval * new_interval,
            torch.where(
                grade == 2,
                torch.maximum(
                    elapsed * hard_interval_factor,
                    prev_interval * hard_interval_factor / 2,
                ),
                torch.where(
                    grade == 4,
                    torch.maximum(elapsed * new_ease, prev_interval) * easy_bonus,
                    torch.maximum(elapsed * new_ease, prev_interval),
                ),
            ),
        )

        existing_card_interval = existing_card_interval * interval_multiplier

        result_interval = torch.where(
            is_new_card, new_card_interval, existing_card_interval
        )

        result_interval = torch.maximum(
            torch.ones_like(result_interval), result_interval
        )

        final_interval = s_max_aware_fixed_interval(
            stability, difficulty, result_interval, decay, w, s_max
        )

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
