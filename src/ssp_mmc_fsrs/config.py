from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_LEARN_COSTS = np.array([33.79, 24.3, 13.68, 6.5])
DEFAULT_REVIEW_COSTS = np.array([23.0, 11.68, 7.33, 5.6])
DEFAULT_FIRST_RATING_PROB = np.array([0.24, 0.094, 0.495, 0.171])
DEFAULT_REVIEW_RATING_PROB = np.array([0.224, 0.631, 0.145])
DEFAULT_FIRST_RATING_OFFSETS = np.array([0.0, 0.0, 0.0, 0.0])
DEFAULT_FIRST_SESSION_LENS = np.array([0.0, 0.0, 0.0, 0.0])
DEFAULT_FORGET_RATING_OFFSET = 0.0
DEFAULT_FORGET_SESSION_LEN = 0.0


DEFAULT_W = [
    0.4002,
    0.1813,
    0.6958,
    3.1337,
    6.6154,
    0.565,
    2.999,
    0.0759,
    1.568,
    0.248,
    0.4816,
    1.7469,
    0.0247,
    0.5986,
    1.8441,
    0.4189,
    1.81,
    0.5784,
    0.1924,
    0.0658,
    0.4444,
]


DEFAULT_SEED = 42

S_MIN = 0.1
S_MAX = 365 * 25
SHORT_STEP = np.log(2) / 20
LONG_STEP = 5

D_MIN = 1
D_MAX = 10
D_EPS = 0.1

R_MIN = 0.70
R_MAX = 0.99
R_EPS = 0.01

COST_MAX = 1_000_000
LEARN_SPAN = 365 * 5
REVIEW_LIMIT_PER_DAY = 9999
DECK_SIZE = 10_000
PARALLEL = 100

DEFAULT_SIMULATION_TYPE = "unlim_time_lim_reviews"


def resolve_simulation_limits(simulation_type: str):
    if simulation_type == "unlim_time_lim_reviews":
        return 10, 86400 / 2
    if simulation_type == "lim_time_unlim_reviews":
        return 9999, 3600
    raise ValueError(f"Unknown simulation type: {simulation_type}")


LEARN_LIMIT_PER_DAY, MAX_STUDYING_TIME_PER_DAY = resolve_simulation_limits(
    DEFAULT_SIMULATION_TYPE
)


OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"
SIMULATION_DIR = OUTPUTS_DIR / "simulation"
POLICIES_DIR = OUTPUTS_DIR / "policies"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
POLICY_CONFIGS_PATH = CHECKPOINTS_DIR / "policy_configs.json"
DR_BASELINE_PATH = CHECKPOINTS_DIR / "dr_baseline.json"
SIMULATION_RESULTS_PATH = CHECKPOINTS_DIR / "simulation_results.json"


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class StateSpaceConfig:
    s_min: float = S_MIN
    s_max: float = S_MAX
    short_step: float = SHORT_STEP
    long_step: float = LONG_STEP
    d_min: float = D_MIN
    d_max: float = D_MAX
    d_eps: float = D_EPS
    r_min: float = R_MIN
    r_max: float = R_MAX
    r_eps: float = R_EPS
    cost_max: float = COST_MAX


@dataclass(frozen=True)
class SimulationConfig:
    learn_span: int = LEARN_SPAN
    review_limit_per_day: int = REVIEW_LIMIT_PER_DAY
    learn_limit_per_day: int = LEARN_LIMIT_PER_DAY
    max_studying_time_per_day: float = MAX_STUDYING_TIME_PER_DAY
    deck_size: int = DECK_SIZE
    parallel: int = PARALLEL
    s_max: float = S_MAX
    seed: int = DEFAULT_SEED
