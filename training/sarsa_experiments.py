"""Experiment runners for Room 2 SARSA sweeps and confirmations."""

import json
import os
import time
from datetime import datetime

import numpy as np

from core.types import EpsilonScheduleConfig, SarsaConfig, SlipConfig
from environments.room2_sarsa import ROOM2_MAP, Room2SARSA
from agents.sarsa import SarsaAgent, evaluate_sarsa_policy


# Room 2 experiment runner.  Screening tries many hyperparameter combinations;
# confirmation repeats candidates with more seeds/episodes.
STORAGE_DIR = os.path.join("storage", "experiments", "room2_sarsa")

ALPHA_VALUES = [0.05, 0.10, 0.30, 0.50]
GAMMA_VALUES = [0.90, 0.95, 0.99]
DECAY_VALUES = [0.990, 0.995, 0.999]

SCREENING_SEEDS = [42]
CONFIRMATION_SEEDS = [42, 123, 256]


def create_factory(slip_config: SlipConfig | None = None):
    def _factory():
        return Room2SARSA(max_steps=200, slip_config=slip_config or SlipConfig())
    return _factory


def _build_config(alpha, gamma, decay, episodes, seed) -> SarsaConfig:
    return SarsaConfig(
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        max_steps=200,
        seed=seed,
        epsilon=EpsilonScheduleConfig(
            kind="exponential",
            start=1.0,
            minimum=0.05,
            decay=decay,
        ),
    )


def run_screening_experiments(
    episodes: int = 2_000,
) -> list[dict]:
    # Fast pass over alpha/gamma/epsilon-decay to identify promising settings.
    os.makedirs(STORAGE_DIR, exist_ok=True)
    factory = create_factory()
    results: list[dict] = []

    for alpha in ALPHA_VALUES:
        for gamma in GAMMA_VALUES:
            for decay in DECAY_VALUES:
                trial_records: list[dict] = []
                for seed in SCREENING_SEEDS:
                    config = _build_config(alpha, gamma, decay, episodes, seed)
                    agent = SarsaAgent(factory, config)
                    t0 = time.time()
                    train_result = agent.train()
                    duration = time.time() - t0

                    eval_summary = evaluate_sarsa_policy(
                        factory, train_result.q_values, n_episodes=100,
                    )

                    final_metrics = train_result.metrics[-1]
                    rolling = train_result.metrics[-min(100, max(10, episodes // 20)):]
                    final_rolling_success = sum(1 for m in rolling if m.success) / len(rolling)
                    initial_rolling = train_result.metrics[:len(rolling)]
                    initial_rolling_success = sum(1 for m in initial_rolling if m.success) / len(initial_rolling)

                    trial_records.append({
                        "seed": seed,
                        "final_eval_success_rate": eval_summary.success_rate,
                        "final_eval_mean_return": eval_summary.mean_return,
                        "final_eval_mean_steps": eval_summary.mean_steps,
                        "mean_successful_steps": eval_summary.mean_successful_steps,
                        "train_duration_sec": duration,
                        "train_final_rolling_success": final_rolling_success,
                        "train_improvement": final_rolling_success - initial_rolling_success,
                        "final_epsilon": train_result.final_epsilon,
                    })

                # Aggregate across seeds
                sr_vals = [r["final_eval_success_rate"] for r in trial_records]
                ms_vals = [r["mean_successful_steps"] for r in trial_records if r["mean_successful_steps"] is not None]
                mr_vals = [r["final_eval_mean_return"] for r in trial_records]
                duration_vals = [r["train_duration_sec"] for r in trial_records]
                impr_vals = [r["train_improvement"] for r in trial_records]

                results.append({
                    "experiment_type": "screening",
                    "alpha": alpha,
                    "gamma": gamma,
                    "decay": decay,
                    "n_seeds": len(trial_records),
                    "mean_success_rate": float(np.mean(sr_vals)) if sr_vals else 0.0,
                    "std_success_rate": float(np.std(sr_vals)) if len(sr_vals) > 1 else 0.0,
                    "mean_successful_steps": float(np.mean(ms_vals)) if ms_vals else None,
                    "mean_return": float(np.mean(mr_vals)) if mr_vals else 0.0,
                    "mean_duration_sec": float(np.mean(duration_vals)),
                    "mean_improvement": float(np.mean(impr_vals)),
                    "trials": trial_records,
                })

    results.sort(key=lambda r: (
        r["mean_success_rate"],
        -r["std_success_rate"],
        -(r["mean_successful_steps"] if r["mean_successful_steps"] is not None else 9999),
        r["mean_return"],
    ), reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(STORAGE_DIR, f"sarsa_screening_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_confirmation_experiments(
    episodes: int = 5_000,
) -> list[dict]:
    # Slower pass with more episodes and seeds to reduce dependence on one
    # lucky random training run.
    os.makedirs(STORAGE_DIR, exist_ok=True)
    factory = create_factory()
    results: list[dict] = []

    for alpha in ALPHA_VALUES:
        for gamma in GAMMA_VALUES:
            for decay in DECAY_VALUES:
                trial_records: list[dict] = []
                for seed in CONFIRMATION_SEEDS:
                    config = _build_config(alpha, gamma, decay, episodes, seed)
                    agent = SarsaAgent(factory, config)
                    t0 = time.time()
                    train_result = agent.train()
                    duration = time.time() - t0

                    eval_summary = evaluate_sarsa_policy(
                        factory, train_result.q_values, n_episodes=100,
                    )

                    trial_records.append({
                        "seed": seed,
                        "final_eval_success_rate": eval_summary.success_rate,
                        "final_eval_mean_return": eval_summary.mean_return,
                        "final_eval_mean_steps": eval_summary.mean_steps,
                        "mean_successful_steps": eval_summary.mean_successful_steps,
                        "train_duration_sec": duration,
                        "final_epsilon": train_result.final_epsilon,
                    })

                sr_vals = [r["final_eval_success_rate"] for r in trial_records]
                ms_vals = [r["mean_successful_steps"] for r in trial_records if r["mean_successful_steps"] is not None]
                mr_vals = [r["final_eval_mean_return"] for r in trial_records]

                results.append({
                    "experiment_type": "confirmation",
                    "alpha": alpha,
                    "gamma": gamma,
                    "decay": decay,
                    "n_seeds": len(trial_records),
                    "mean_success_rate": float(np.mean(sr_vals)) if sr_vals else 0.0,
                    "std_success_rate": float(np.std(sr_vals)) if len(sr_vals) > 1 else 0.0,
                    "mean_successful_steps": float(np.mean(ms_vals)) if ms_vals else None,
                    "mean_return": float(np.mean(mr_vals)) if mr_vals else 0.0,
                    "mean_duration_sec": float(np.mean(duration)),
                    "trials": trial_records,
                })

    results.sort(key=lambda r: (
        r["mean_success_rate"],
        -r["std_success_rate"],
        -(r["mean_successful_steps"] if r["mean_successful_steps"] is not None else 9999),
        r["mean_return"],
    ), reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(STORAGE_DIR, f"sarsa_confirmation_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results
