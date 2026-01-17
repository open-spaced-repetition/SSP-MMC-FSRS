import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import (
    DEFAULT_FIRST_RATING_OFFSETS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_FIRST_SESSION_LENS,
    DEFAULT_FORGET_RATING_OFFSET,
    DEFAULT_FORGET_SESSION_LEN,
    DEFAULT_REVIEW_COSTS,
    DEFAULT_REVIEW_RATING_PROB,
    DEFAULT_W,
    DR_BASELINE_PATH,
    POLICY_CONFIGS_PATH,
    SIMULATION_RESULTS_PATH,
    default_device,
)
from .core import next_interval_torch
from .solver import SSPMMCSolver


def _safe_slug(value):
    if not value:
        return "policy"
    result = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            result.append(ch)
        elif ch in (" ", "(", ")", "[", "]"):
            result.append("_")
        else:
            result.append("_")
    return "".join(result).strip("_").lower()


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _policy_hash(hyperparams):
    normalized = {k: _jsonify(v) for k, v in hyperparams.items()}
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha1(payload).hexdigest()[:8]


def save_policy(output_dir, title, solver, cost_matrix, retention_matrix, hyperparams):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_id = _policy_hash(hyperparams)
    base_name = f"{_safe_slug(title)}_{policy_id}"
    npz_path = output_dir / f"{base_name}.npz"
    meta_path = output_dir / f"{base_name}.json"

    np.savez_compressed(
        npz_path,
        retention_matrix=retention_matrix,
        cost_matrix=cost_matrix,
        s_state=solver.s_state,
        d_state=solver.d_state,
        r_state=solver.r_state,
    )

    meta = {
        "title": title,
        "policy_id": policy_id,
        "policy_file": npz_path.name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hyperparams": {k: _jsonify(v) for k, v in hyperparams.items()},
        "state_space": {
            "s_min": float(solver.s_min),
            "s_max": float(solver.s_max),
            "short_step": float(solver.short_step),
            "long_step": float(solver.long_step),
            "d_min": float(solver.d_min),
            "d_max": float(solver.d_max),
            "d_eps": float(solver.d_eps),
            "r_min": float(solver.r_min),
            "r_max": float(solver.r_max),
            "r_eps": float(solver.r_eps),
        },
        "w": _jsonify(np.array(solver.w)),
        "review_costs": _jsonify(solver.review_costs),
        "review_rating_prob": _jsonify(solver.review_rating_prob),
        "first_rating_prob": _jsonify(solver.first_rating_prob),
        "first_rating_offsets": _jsonify(solver.first_rating_offsets),
        "first_session_lens": _jsonify(solver.first_session_lens),
        "forget_rating_offset": float(solver.forget_rating_offset),
        "forget_session_len": float(solver.forget_session_len),
    }

    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_policy(policy_path, device=None):
    policy_path = Path(policy_path)
    meta = None
    meta_path = None
    npz_path = None

    if policy_path.suffix == ".json":
        meta_path = policy_path
        with meta_path.open("r") as f:
            meta = json.load(f)
        npz_path = meta_path.parent / meta["policy_file"]
    elif policy_path.suffix == ".npz":
        npz_path = policy_path
        meta_path = policy_path.with_suffix(".json")
        if meta_path.exists():
            with meta_path.open("r") as f:
                meta = json.load(f)
    else:
        candidate_json = policy_path.with_suffix(".json")
        candidate_npz = policy_path.with_suffix(".npz")
        if candidate_json.exists():
            meta_path = candidate_json
            with meta_path.open("r") as f:
                meta = json.load(f)
            npz_path = meta_path.parent / meta["policy_file"]
        elif candidate_npz.exists():
            npz_path = candidate_npz

    if npz_path is None or not npz_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    data = np.load(npz_path)
    retention_matrix = data["retention_matrix"]
    cost_matrix = data["cost_matrix"] if "cost_matrix" in data else None
    s_state = data["s_state"]
    d_state = data["d_state"]
    r_state = data["r_state"]

    if meta is None:
        raise ValueError(f"Missing metadata JSON for policy: {npz_path}")

    state_space = meta["state_space"]
    w_local = np.array(meta.get("w", DEFAULT_W), dtype=float)
    review_costs_local = np.array(
        meta.get("review_costs", DEFAULT_REVIEW_COSTS), dtype=float
    )
    first_rating_prob_local = np.array(
        meta.get("first_rating_prob", DEFAULT_FIRST_RATING_PROB), dtype=float
    )
    review_rating_prob_local = np.array(
        meta.get("review_rating_prob", DEFAULT_REVIEW_RATING_PROB), dtype=float
    )
    first_rating_offsets_local = np.array(
        meta.get("first_rating_offsets", DEFAULT_FIRST_RATING_OFFSETS), dtype=float
    )
    first_session_lens_local = np.array(
        meta.get("first_session_lens", DEFAULT_FIRST_SESSION_LENS), dtype=float
    )
    forget_rating_offset_local = float(
        meta.get("forget_rating_offset", DEFAULT_FORGET_RATING_OFFSET)
    )
    forget_session_len_local = float(
        meta.get("forget_session_len", DEFAULT_FORGET_SESSION_LEN)
    )

    solver = SSPMMCSolver(
        review_costs=review_costs_local,
        first_rating_prob=first_rating_prob_local,
        review_rating_prob=review_rating_prob_local,
        first_rating_offsets=first_rating_offsets_local,
        first_session_lens=first_session_lens_local,
        forget_rating_offset=forget_rating_offset_local,
        forget_session_len=forget_session_len_local,
        w=w_local,
    )
    solver._init_state_spaces(
        s_min=state_space["s_min"],
        s_max=state_space["s_max"],
        short_step=state_space["short_step"],
        long_step=state_space["long_step"],
        d_min=state_space["d_min"],
        d_max=state_space["d_max"],
        d_eps=state_space["d_eps"],
        r_min=state_space["r_min"],
        r_max=state_space["r_max"],
        r_eps=state_space["r_eps"],
    )

    if retention_matrix.shape != (solver.d_size, solver.s_size):
        raise ValueError(
            "Retention matrix shape does not match state space: "
            f"{retention_matrix.shape} vs ({solver.d_size}, {solver.s_size})"
        )

    if not np.array_equal(s_state, solver.s_state) or not np.array_equal(
        d_state, solver.d_state
    ):
        raise ValueError("State grids do not match metadata state space parameters")

    if device is None:
        device = default_device()
    retention_matrix_tensor = torch.tensor(retention_matrix, device=device)
    decay = -w_local[20]

    def ssp_mmc_policy(s, d):
        d_index = solver.d2i_torch(d)
        s_index = solver.s2i_torch(s)
        mask = (d_index >= solver.d_size) | (s_index >= solver.s_size - 1)
        optimal_interval = torch.zeros_like(s)
        optimal_interval[~mask] = next_interval_torch(
            s[~mask],
            retention_matrix_tensor[d_index[~mask], s_index[~mask]],
            decay,
        )
        optimal_interval[mask] = np.inf
        return optimal_interval

    return {
        "policy": ssp_mmc_policy,
        "solver": solver,
        "retention_matrix": retention_matrix,
        "cost_matrix": cost_matrix,
        "meta": meta,
    }


def save_policy_configs(policy_configs, path=POLICY_CONFIGS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(policy_configs, f, indent=2, sort_keys=True)


def load_policy_configs(path=POLICY_CONFIGS_PATH):
    path = Path(path)
    with path.open("r") as f:
        policy_configs = json.load(f)
    if not isinstance(policy_configs, list):
        raise ValueError(f"Policy configs must be a list: {path}")
    for entry in policy_configs:
        if not isinstance(entry, dict) or "params" not in entry:
            raise ValueError(f"Invalid policy config entry in {path}: {entry}")
    return policy_configs


def save_dr_baseline(dr_baseline, path=DR_BASELINE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(dr_baseline, f, indent=2, sort_keys=True)


def load_dr_baseline(path=DR_BASELINE_PATH):
    path = Path(path)
    with path.open("r") as f:
        dr_baseline = json.load(f)
    if not isinstance(dr_baseline, list):
        raise ValueError(f"DR baseline must be a list: {path}")
    for entry in dr_baseline:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid DR baseline entry in {path}: {entry}")
        missing = {"dr", "average_knowledge", "average_knowledge_per_hour"} - set(
            entry.keys()
        )
        if missing:
            raise ValueError(f"Missing {missing} in DR baseline entry: {entry}")
    return dr_baseline


def save_simulation_results(simulation_results, path=SIMULATION_RESULTS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(simulation_results, f, indent=2, sort_keys=True)


def load_simulation_results(path=SIMULATION_RESULTS_PATH):
    path = Path(path)
    with path.open("r") as f:
        simulation_results = json.load(f)
    if not isinstance(simulation_results, list):
        raise ValueError(f"Simulation results must be a list: {path}")
    for entry in simulation_results:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid simulation result entry in {path}: {entry}")
        missing = {
            "title",
            "reviews_average",
            "time_average",
            "memorized_average",
            "avg_accum_memorized_per_hour",
        } - set(entry.keys())
        if missing:
            raise ValueError(f"Missing {missing} in simulation result entry: {entry}")
    return simulation_results
