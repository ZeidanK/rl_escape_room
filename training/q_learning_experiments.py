import json
import os
import time
from datetime import datetime

import numpy as np

from core.types import EpsilonScheduleConfig, QLearningConfig, SlipConfig
from environments.room3_qlearning import Room3QLearning
from agents.q_learning import QLearningAgent, evaluate_q_learning_policy


STORAGE_DIR = os.path.join("storage", "experiments", "room3_q_learning")
SCREENING_DIR = os.path.join(STORAGE_DIR, "screening")
CONFIRMATION_DIR = os.path.join(STORAGE_DIR, "confirmation")

ALPHA_VALUES = [0.05, 0.10, 0.30, 0.50]
GAMMA_VALUES = [0.90, 0.95, 0.99]
DECAY_VALUES = [0.990, 0.995, 0.999]

SCREENING_SEEDS = [42]
CONFIRMATION_SEEDS = [42, 123, 256]


def _make_room3(slip_config: SlipConfig | None = None):
    def _factory():
        return Room3QLearning(max_steps=200, slip_config=slip_config or SlipConfig())
    return _factory


def _build_config(alpha, gamma, decay, episodes, seed) -> QLearningConfig:
    return QLearningConfig(
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
    episodes: int = 1_000,
) -> list[dict]:
    os.makedirs(SCREENING_DIR, exist_ok=True)
    factory = _make_room3()
    results: list[dict] = []

    for alpha in ALPHA_VALUES:
        for gamma in GAMMA_VALUES:
            for decay in DECAY_VALUES:
                trial_records: list[dict] = []
                for seed in SCREENING_SEEDS:
                    config = _build_config(alpha, gamma, decay, episodes, seed)
                    agent = QLearningAgent(factory, config)
                    t0 = time.time()
                    train_result = agent.train()
                    duration = time.time() - t0

                    eval_summary = evaluate_q_learning_policy(
                        factory, train_result.q_values, n_episodes=50,
                    )

                    final_metrics = train_result.metrics[-1]
                    rolling = train_result.metrics[-min(100, max(10, episodes // 20)):]
                    final_rolling_success = sum(1 for m in rolling if m.success) / len(rolling)
                    initial_rolling = train_result.metrics[:len(rolling)]
                    initial_rolling_success = sum(1 for m in initial_rolling if m.success) / len(initial_rolling)

                    trial_records.append({
                        "seed": seed,
                        "final_eval_success_rate": eval_summary.success_rate,
                        "final_eval_key_collection_rate": eval_summary.key_collection_rate,
                        "final_eval_mean_return": eval_summary.mean_return,
                        "final_eval_mean_steps": eval_summary.mean_steps,
                        "mean_successful_steps": eval_summary.mean_successful_steps,
                        "train_duration_sec": duration,
                        "train_final_rolling_success": final_rolling_success,
                        "train_improvement": final_rolling_success - initial_rolling_success,
                        "final_epsilon": train_result.final_epsilon,
                    })

                sr_vals = [r["final_eval_success_rate"] for r in trial_records]
                kcr_vals = [r["final_eval_key_collection_rate"] for r in trial_records]
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
                    "mean_key_collection_rate": float(np.mean(kcr_vals)) if kcr_vals else 0.0,
                    "mean_successful_steps": float(np.mean(ms_vals)) if ms_vals else None,
                    "mean_return": float(np.mean(mr_vals)) if mr_vals else 0.0,
                    "mean_duration_sec": float(np.mean(duration_vals)),
                    "mean_improvement": float(np.mean(impr_vals)),
                    "trials": trial_records,
                })

    results.sort(key=lambda r: (
        r["mean_success_rate"],
        r["mean_key_collection_rate"],
        -r["std_success_rate"],
        -(r["mean_successful_steps"] if r["mean_successful_steps"] is not None else 9999),
    ), reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SCREENING_DIR, f"q_learning_screening_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Q-Learning Screening — Top 5 Configs")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Alpha':<8} {'Gamma':<8} {'Decay':<8} {'SR':<8} {'KCR':<8} {'SR Std':<8} {'Steps':<8}")
    print(f"{'-'*80}")
    for rank, r in enumerate(results[:5], 1):
        ms = r["mean_successful_steps"]
        steps_str = f"{ms:.1f}" if ms is not None else "N/A"
        print(f"{rank:<6} {r['alpha']:<8.2f} {r['gamma']:<8.2f} {r['decay']:<8.3f} {r['mean_success_rate']:<8.3f} {r['mean_key_collection_rate']:<8.3f} {r['std_success_rate']:<8.3f} {steps_str:<8}")
    print(f"{'='*80}\n")

    return results


def run_confirmation_experiments(
    top_k: int = 5,
    episodes: int = 5_000,
) -> list[dict]:
    screening_results = run_screening_experiments()
    top_configs = screening_results[:top_k]

    os.makedirs(CONFIRMATION_DIR, exist_ok=True)
    factory = _make_room3()
    results: list[dict] = []

    for cfg in top_configs:
        alpha = cfg["alpha"]
        gamma = cfg["gamma"]
        decay = cfg["decay"]
        trial_records: list[dict] = []
        for seed in CONFIRMATION_SEEDS:
            config = _build_config(alpha, gamma, decay, episodes, seed)
            agent = QLearningAgent(factory, config)
            t0 = time.time()
            train_result = agent.train()
            duration = time.time() - t0

            eval_summary = evaluate_q_learning_policy(
                factory, train_result.q_values, n_episodes=100,
            )

            trial_records.append({
                "seed": seed,
                "final_eval_success_rate": eval_summary.success_rate,
                "final_eval_key_collection_rate": eval_summary.key_collection_rate,
                "final_eval_mean_return": eval_summary.mean_return,
                "final_eval_mean_steps": eval_summary.mean_steps,
                "mean_successful_steps": eval_summary.mean_successful_steps,
                "train_duration_sec": duration,
                "final_epsilon": train_result.final_epsilon,
            })

        sr_vals = [r["final_eval_success_rate"] for r in trial_records]
        kcr_vals = [r["final_eval_key_collection_rate"] for r in trial_records]
        ms_vals = [r["mean_successful_steps"] for r in trial_records if r["mean_successful_steps"] is not None]
        mr_vals = [r["final_eval_mean_return"] for r in trial_records]
        duration_vals = [r["train_duration_sec"] for r in trial_records]

        results.append({
            "experiment_type": "confirmation",
            "alpha": alpha,
            "gamma": gamma,
            "decay": decay,
            "n_seeds": len(trial_records),
            "mean_success_rate": float(np.mean(sr_vals)) if sr_vals else 0.0,
            "std_success_rate": float(np.std(sr_vals)) if len(sr_vals) > 1 else 0.0,
            "mean_key_collection_rate": float(np.mean(kcr_vals)) if kcr_vals else 0.0,
            "mean_successful_steps": float(np.mean(ms_vals)) if ms_vals else None,
            "mean_return": float(np.mean(mr_vals)) if mr_vals else 0.0,
            "mean_duration_sec": float(np.mean(duration_vals)),
            "trials": trial_records,
        })

    default_trial_records: list[dict] = []
    for seed in CONFIRMATION_SEEDS:
        config = QLearningConfig(episodes=episodes, seed=seed)
        agent = QLearningAgent(factory, config)
        t0 = time.time()
        train_result = agent.train()
        duration = time.time() - t0

        eval_summary = evaluate_q_learning_policy(
            factory, train_result.q_values, n_episodes=100,
        )

        default_trial_records.append({
            "seed": seed,
            "final_eval_success_rate": eval_summary.success_rate,
            "final_eval_key_collection_rate": eval_summary.key_collection_rate,
            "final_eval_mean_return": eval_summary.mean_return,
            "final_eval_mean_steps": eval_summary.mean_steps,
            "mean_successful_steps": eval_summary.mean_successful_steps,
            "train_duration_sec": duration,
            "final_epsilon": train_result.final_epsilon,
        })

    sr_vals = [r["final_eval_success_rate"] for r in default_trial_records]
    kcr_vals = [r["final_eval_key_collection_rate"] for r in default_trial_records]
    ms_vals = [r["mean_successful_steps"] for r in default_trial_records if r["mean_successful_steps"] is not None]
    mr_vals = [r["final_eval_mean_return"] for r in default_trial_records]
    duration_vals = [r["train_duration_sec"] for r in default_trial_records]

    results.append({
        "experiment_type": "confirmation_default",
        "alpha": 0.10,
        "gamma": 0.95,
        "decay": 0.995,
        "n_seeds": len(default_trial_records),
        "mean_success_rate": float(np.mean(sr_vals)) if sr_vals else 0.0,
        "std_success_rate": float(np.std(sr_vals)) if len(sr_vals) > 1 else 0.0,
        "mean_key_collection_rate": float(np.mean(kcr_vals)) if kcr_vals else 0.0,
        "mean_successful_steps": float(np.mean(ms_vals)) if ms_vals else None,
        "mean_return": float(np.mean(mr_vals)) if mr_vals else 0.0,
        "mean_duration_sec": float(np.mean(duration_vals)),
        "trials": default_trial_records,
        "is_default": True,
    })

    results.sort(key=lambda r: (
        r["mean_success_rate"],
        r["mean_key_collection_rate"],
        -r["std_success_rate"],
        -(r["mean_successful_steps"] if r["mean_successful_steps"] is not None else 9999),
    ), reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(CONFIRMATION_DIR, f"q_learning_confirmation_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results
