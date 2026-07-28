"""Shared resumable experiment framework with reproducibility metadata."""

import hashlib
import hmac
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np

from agents.tabular_utils import map_signature as _grid_signature
from core.types import (
    ApproximateEvaluationSummary,
    ValueIterationConfig,
)
from environments.room1_dp import Room1DP
from environments.room2_sarsa import ROOM2_GRID
from environments.room3_qlearning import ROOM3_GRID
from environments.room4_continuous import Room4Continuous


FINAL_DIR = os.path.join("storage", "experiments", "final")


def git_commit() -> str:
    # Stored with experiment outputs to make results traceable to a code state.
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def python_version() -> str:
    return sys.version


def package_versions() -> dict[str, str]:
    import importlib.metadata
    deps = ["numpy", "streamlit", "pytest"]
    return {d: importlib.metadata.version(d) for d in deps if _pkg_installed(d)}


def _pkg_installed(name: str) -> bool:
    import importlib.metadata
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def room1_map_signature() -> str:
    return _grid_signature(Room1DP().grid)


def room2_map_signature() -> str:
    return _grid_signature(ROOM2_GRID)


def room3_map_signature() -> str:
    return _grid_signature(ROOM3_GRID)


def room4_map_signature() -> str:
    env = Room4Continuous()
    return _grid_signature(np.zeros((10, 10), dtype=int))


def _room4_cfg_id(cfg: dict) -> str:
    start_mode = cfg.get("start_mode")
    start_suffix = ""
    if start_mode:
        start_mode_value = getattr(start_mode, "value", str(start_mode))
        if start_mode_value != "fixed":
            start_suffix = f"_sm={start_mode_value}"
    return (
        f"nt={cfg['num_tilings']}_tx={cfg['tiles_xy']}"
        f"_a={cfg['alpha']}_ps={cfg['progress_scale']}"
        f"_ed={cfg['epsilon_decay']}_ep={cfg['episodes']}"
        f"_s={cfg['seed']}{start_suffix}"
    )


def make_base_metadata(
    algorithm: str,
    room: str,
    map_sig: str,
    config: dict,
    training_seeds: list[int],
    evaluation_seeds: list[int],
    ranking_criteria: list[str],
    reward_config: dict | None = None,
    motion_config: dict | None = None,
) -> dict:
    # Shared metadata block for final experiment artifacts.  This makes JSON
    # outputs defensible and reproducible during grading.
    return {
        "schema_version": 1,
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "python_version": python_version(),
        "package_versions": package_versions(),
        "algorithm": algorithm,
        "room": room,
        "map_signature": map_sig,
        "config": config,
        "reward_config": reward_config or {},
        "motion_config": motion_config or {},
        "training_seeds": training_seeds,
        "evaluation_seeds": evaluation_seeds,
        "ranking_criteria": ranking_criteria,
        "retained_rollouts": False,
    }


def save_trial(
    filepath: str,
    data: dict,
) -> None:
    # Write through a temporary file first so interrupted experiments do not
    # leave behind partially-written JSON.
    data.setdefault("schema_version", 1)
    data.setdefault("generated_at", now_iso())
    data.setdefault("git_commit", git_commit())
    data.setdefault("python_version", python_version())
    data.setdefault("package_versions", package_versions())
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = filepath + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, filepath)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_completed_results(
    directory: str,
    suffix: str = ".json",
) -> dict[str, dict]:
    # Resumable pipelines use this to skip trials that already finished.
    results: dict[str, dict] = {}
    if not os.path.isdir(directory):
        return results
    for fn in os.listdir(directory):
        if fn.endswith(suffix) and not fn.endswith(".tmp"):
            path = os.path.join(directory, fn)
            with open(path) as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
            results[fn] = data
    return results


def trial_id_for_sarsa(alpha: float, gamma: float, decay: float, seed: int) -> str:
    return f"sarsa_a={alpha}_g={gamma}_d={decay}_s={seed}"


def trial_id_for_q(alpha: float, gamma: float, decay: float, seed: int) -> str:
    return f"q_a={alpha}_g={gamma}_d={decay}_s={seed}"


def trial_id_for_room4(cfg: dict, seed: int) -> str:
    return f"room4_{_room4_cfg_id({**cfg, 'seed': seed, 'episodes': cfg.get('episodes', 3000)})}"


# ============================================================
# Ranking helpers
# ============================================================

def rank_sarsa(results: list[dict]) -> list[dict]:
    # Primary criterion is success rate; later criteria break ties with
    # stability, speed, return, and deterministic config ordering.
    return sorted(results, key=lambda r: (
        r.get("mean_success_rate", 0.0),
        -(r.get("std_success_rate", 1.0)),
        -(r.get("mean_successful_steps", 9999) or 9999),
        r.get("mean_return", -9999),
        r.get("mean_improvement", -9999),
        str(r.get("config_tuple", "")),
    ), reverse=True)


def rank_q_learning(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda r: (
        r.get("mean_success_rate", 0.0),
        r.get("mean_key_collection_rate", 0.0),
        -(r.get("std_success_rate", 1.0)),
        -(r.get("mean_successful_steps", 9999) or 9999),
        -(r.get("mean_key_collection_step", 9999) or 9999),
        r.get("mean_return", -9999),
        str(r.get("config_tuple", "")),
    ), reverse=True)


def rank_room4(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda r: (
        r.get("fixed_training_start_success_rate", 0.0),
        -(r.get("fixed_training_start_success_rate_std", 0.0) or 0.0),
        r.get("fixed_unseen_starts_success_rate", 0.0),
        r.get("random_lower_left_success_rate", 0.0),
        r.get("random_room_success_rate", 0.0),
        -(r.get("truncation_count", 9999)),
        -(r.get("mean_successful_steps", 9999) or 9999),
        r.get("mean_return", -9999),
        -(r.get("std_return", 9999)),
        str(r.get("config_tuple", "")),
    ), reverse=True)


# ============================================================
# Benchmark utility
# ============================================================

def benchmark_config(
    algorithm: str,
    episodes: int,
    seed: int = 42,
) -> float:
    # Quick runtime estimate used by the final pipeline before launching longer
    # experiment sweeps.
    from agents.sarsa import SarsaAgent
    from agents.q_learning import QLearningAgent
    from agents.approximate_sarsa import ApproximateSarsaAgent
    from core.types import (
        ApproximateSarsaConfig,
        EpsilonScheduleConfig,
        QLearningConfig,
        SarsaConfig,
        TileCodingConfig,
    )
    from environments.room4_continuous import ContinuousRewardConfig

    t0 = time.time()
    if algorithm == "sarsa":
        factory = lambda: Room1DP()
        config = SarsaConfig(
            episodes=episodes, alpha=0.1, gamma=0.95, max_steps=200, seed=seed,
            epsilon=EpsilonScheduleConfig(kind="exponential", start=1.0, minimum=0.05, decay=0.995),
        )
        from environments.room2_sarsa import Room2SARSA
        factory = lambda: Room2SARSA(max_steps=200)
        agent = SarsaAgent(factory, config)
        agent.train()
    elif algorithm == "q_learning":
        from environments.room3_qlearning import Room3QLearning
        factory = lambda: Room3QLearning(max_steps=200)
        config = QLearningConfig(
            episodes=episodes, alpha=0.1, gamma=0.95, max_steps=200, seed=seed,
            epsilon=EpsilonScheduleConfig(kind="exponential", start=1.0, minimum=0.05, decay=0.995),
        )
        agent = QLearningAgent(factory, config)
        agent.train()
    elif algorithm == "room4":
        factory = lambda: Room4Continuous(
            max_steps=750,
            reward_config=ContinuousRewardConfig(distance_progress_scale=1.0),
        )
        config = ApproximateSarsaConfig(
            episodes=episodes, alpha=0.1, gamma=0.99, max_steps=750, seed=seed,
            epsilon=EpsilonScheduleConfig(kind="exponential", start=1.0, minimum=0.02, decay=0.997),
            tile_coding=TileCodingConfig(num_tilings=8, tiles_x=10, tiles_y=10, include_velocity=True),
        )
        agent = ApproximateSarsaAgent(factory, config)
        agent.train()
    return time.time() - t0
