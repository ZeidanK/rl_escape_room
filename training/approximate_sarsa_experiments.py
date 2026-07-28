"""Experiment runners for Room 4 approximate SARSA sweeps and confirmations."""

import json
import os
from collections.abc import Sequence

import numpy as np

from core.types import (
    ApproximateEvaluationSummary,
    ApproximateSarsaConfig,
    EpsilonScheduleConfig,
    StartMode,
    TileCodingConfig,
)
from agents.approximate_sarsa import ApproximateSarsaAgent, evaluate_approximate_policy
from environments.room4_continuous import Room4Continuous, ContinuousRewardConfig, Room4MotionConfig


# Room 4 experiment runner for tile-coding Approximate SARSA.  The search is
# split into stages because full continuous-control training is slower than
# the tabular rooms.
def _make_env_factory(
    start_mode: StartMode = StartMode.RANDOM_LOWER_LEFT,
    distance_progress_scale: float = 1.0,
):
    def factory():
        return Room4Continuous(
            motion_config=Room4MotionConfig(),
            reward_config=ContinuousRewardConfig(distance_progress_scale=distance_progress_scale),
            max_steps=750,
            start_mode=start_mode,
        )
    return factory


def run_screening_stage_a(
    *,
    n_episodes: int = 500,
    eval_episodes: int = 50,
    seed: int = 42,
) -> list[dict]:
    """
    Stage A: One factor at a time around defaults.
    Defaults: 8 tilings, 10x10 tiles, alpha=0.10, progress_scale=1.0
    """
    # One-factor-at-a-time screening helps identify useful ranges before
    # testing combinations.
    results = []
    params_to_test = {
        "num_tilings": [4, 8, 16],
        "tiles_xy": [8, 10, 16],
        "alpha": [0.05, 0.10, 0.20],
        "progress_scale": [0.0, 0.5, 1.0],
        "epsilon_decay": [0.995, 0.997, 0.999],
    }

    defaults = {
        "num_tilings": 8,
        "tiles_xy": 10,
        "alpha": 0.10,
        "progress_scale": 1.0,
        "epsilon_decay": 0.997,
    }

    for param_name, values in params_to_test.items():
        for val in values:
            nt = defaults["num_tilings"] if param_name != "num_tilings" else val
            tx = defaults["tiles_xy"] if param_name not in ("tiles_xy",) else val
            ty = tx
            alp = defaults["alpha"] if param_name != "alpha" else val
            ps = defaults["progress_scale"] if param_name != "progress_scale" else val
            ed = defaults["epsilon_decay"] if param_name != "epsilon_decay" else val

            config = ApproximateSarsaConfig(
                episodes=n_episodes,
                alpha=alp,
                gamma=0.99,
                max_steps=750,
                seed=seed,
                epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.02, decay=ed),
                tile_coding=TileCodingConfig(num_tilings=nt, tiles_x=tx, tiles_y=ty, include_velocity=True),
                start_mode=StartMode.RANDOM_LOWER_LEFT,
            )
            factory = _make_env_factory(start_mode=StartMode.RANDOM_LOWER_LEFT, distance_progress_scale=ps)
            agent = ApproximateSarsaAgent(factory, config)
            result = agent.train()
            # Fixed start evaluation
            eval_fixed = evaluate_approximate_policy(
                factory, result.weights, config.tile_coding, Room4MotionConfig(),
                n_episodes=eval_episodes, start_mode=StartMode.FIXED,
            )
            eval_random = evaluate_approximate_policy(
                factory, result.weights, config.tile_coding, Room4MotionConfig(),
                n_episodes=eval_episodes, start_mode=StartMode.RANDOM_LOWER_LEFT,
            )
            results.append({
                "config": {
                    "num_tilings": nt,
                    "tiles_xy": tx,
                    "alpha": alp,
                    "progress_scale": ps,
                    "epsilon_decay": ed,
                },
                "fixed_sr": eval_fixed.success_rate,
                "random_sr": eval_random.success_rate,
                "fixed_return": eval_fixed.mean_return,
                "random_return": eval_random.mean_return,
                "fixed_steps": eval_fixed.mean_steps,
                "truncated": eval_fixed.truncated_count + eval_random.truncated_count,
            })

    return results


def run_screening_stage_b(
    stage_a_results: list[dict],
    *,
    n_episodes: int = 1000,
    eval_episodes: int = 50,
) -> list[dict]:
    """
    Stage B: Combine best 2 values per factor based on Stage A fixed-start SR.
    """
    # Stage B narrows the search to combinations of the strongest Stage A
    # values, reducing total training time.
    best_of = {}
    for param in ["num_tilings", "tiles_xy", "alpha", "progress_scale", "epsilon_decay"]:
        scored = {}
        for r in stage_a_results:
            key = r["config"][param]
            if key not in scored:
                scored[key] = []
            scored[key].append(r["fixed_sr"])
        means = {k: np.mean(v) for k, v in scored.items()}
        sorted_params = sorted(means.keys(), key=lambda k: means[k], reverse=True)
        best_of[param] = sorted_params[:2]

    results = []
    vals = list(best_of.values())
    from itertools import product
    for combo in product(*vals):
        nt, tx, alp, ps, ed = combo
        ty = tx
        config = ApproximateSarsaConfig(
            episodes=n_episodes,
            alpha=alp,
            gamma=0.99,
            max_steps=750,
            seed=42,
            epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.02, decay=ed),
            tile_coding=TileCodingConfig(num_tilings=nt, tiles_x=tx, tiles_y=ty, include_velocity=True),
            start_mode=StartMode.RANDOM_LOWER_LEFT,
        )
        factory = _make_env_factory(distance_progress_scale=ps)
        agent = ApproximateSarsaAgent(factory, config)
        result = agent.train()
        eval_fixed = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=eval_episodes, start_mode=StartMode.FIXED,
        )
        eval_random = evaluate_approximate_policy(
            factory, result.weights, config.tile_coding, Room4MotionConfig(),
            n_episodes=eval_episodes, start_mode=StartMode.RANDOM_LOWER_LEFT,
        )
        results.append({
            "config": {
                "num_tilings": nt, "tiles_xy": tx, "alpha": alp,
                "progress_scale": ps, "epsilon_decay": ed,
            },
            "fixed_sr": eval_fixed.success_rate,
            "random_sr": eval_random.success_rate,
            "fixed_return": eval_fixed.mean_return,
            "random_return": eval_random.mean_return,
            "fixed_steps": eval_fixed.mean_steps,
            "truncated": eval_fixed.truncated_count + eval_random.truncated_count,
        })

    return results


def run_confirmation_experiments(
    configs: Sequence[dict],
    *,
    n_episodes: int = 3000,
    eval_episodes: int = 100,
    seeds: Sequence[int] = (42, 43, 44),
) -> list[dict]:
    results = []
    for cfg in configs:
        per_seed = []
        for seed in seeds:
            config = ApproximateSarsaConfig(
                episodes=n_episodes,
                alpha=cfg["alpha"],
                gamma=0.99,
                max_steps=750,
                seed=seed,
                epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.02, decay=cfg.get("epsilon_decay", 0.997)),
                tile_coding=TileCodingConfig(
                    num_tilings=cfg["num_tilings"],
                    tiles_x=cfg["tiles_xy"],
                    tiles_y=cfg["tiles_xy"],
                    include_velocity=True,
                ),
                start_mode=StartMode.RANDOM_LOWER_LEFT,
            )
            factory = _make_env_factory(distance_progress_scale=cfg.get("progress_scale", 1.0))
            agent = ApproximateSarsaAgent(factory, config)
            result = agent.train()
            for scat in ["fixed", "unseen", "random"]:
                sm = StartMode.FIXED if scat == "fixed" else (
                    StartMode.FIXED if scat == "unseen" else StartMode.RANDOM_LOWER_LEFT
                )
                # We can use FIXED with different start positions for unseen
                # For simplicity, use FIXED for fixed, RANDOM_LOWER_LEFT for random
            eval_fixed = evaluate_approximate_policy(
                factory, result.weights, config.tile_coding, Room4MotionConfig(),
                n_episodes=eval_episodes, start_mode=StartMode.FIXED,
            )
            per_seed.append({
                "seed": seed,
                "fixed_sr": eval_fixed.success_rate,
                "fixed_return": eval_fixed.mean_return,
                "fixed_steps": eval_fixed.mean_steps,
                "truncated": eval_fixed.truncated_count,
            })

        fixed_srs = [ps["fixed_sr"] for ps in per_seed]
        fixed_rets = [ps["fixed_return"] for ps in per_seed]
        results.append({
            "config": cfg,
            "seeds": seeds,
            "fixed_sr_mean": float(np.mean(fixed_srs)),
            "fixed_sr_std": float(np.std(fixed_srs)),
            "fixed_return_mean": float(np.mean(fixed_rets)),
            "fixed_return_std": float(np.std(fixed_rets)),
            "per_seed": per_seed,
        })

    return results


def save_experiments(data, filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return filepath
