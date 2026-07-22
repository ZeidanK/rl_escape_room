"""Controlled SARSA-versus-Q-Learning comparison on Room 2 benchmark."""

import json
import os
from dataclasses import dataclass, field

import numpy as np

from agents.q_learning import (
    QLearningAgent,
    evaluate_q_learning_policy,
    rollout_q_learning_policy,
)
from agents.sarsa import (
    SarsaAgent,
    evaluate_sarsa_policy,
    rollout_sarsa_policy,
)
from agents.tabular_utils import (
    epsilon_for_episode,
    extract_deterministic_greedy_policy,
)
from core.types import (
    EpsilonScheduleConfig,
    QLearningConfig,
    QLearningEvaluationSummary,
    RewardConfig,
    SarsaConfig,
    SarsaEvaluationSummary,
    SlipConfig,
)
from environments.room2_sarsa import ROOM2_GRID, Room2SARSA


@dataclass(frozen=True)
class ComparisonResult:
    algorithm: str
    config_label: str
    seed: int
    success_rate: float
    mean_return: float
    std_return: float
    mean_steps: float
    mean_successful_steps: float | None
    total_collisions: int
    total_slipped_actions: int
    total_traps: int
    eval_seeds_used: int


@dataclass(frozen=True)
class TunedComparisonResult:
    algorithm: str
    config: dict
    success_rate_mean: float
    success_rate_std: float
    mean_return_mean: float
    mean_return_std: float
    mean_steps_mean: float
    total_collisions: int
    total_slipped: int
    total_traps: int
    per_seed: list[ComparisonResult]


def _room2_factory():
    return Room2SARSA(max_steps=200, reward_config=RewardConfig(), slip_config=SlipConfig(0.8, 0.1, 0.1))


def run_matched_comparison(
    *,
    alpha: float = 0.10,
    gamma: float = 0.95,
    episodes: int = 2000,
    epsilon_decay: float = 0.995,
    training_seeds: list[int] | None = None,
    eval_seeds: range | None = None,
    max_steps: int = 200,
) -> tuple[list[ComparisonResult], list[ComparisonResult]]:
    """Comparison A: identical hyperparameters for both algorithms."""
    if training_seeds is None:
        training_seeds = [0, 1, 2, 3, 4]
    if eval_seeds is None:
        eval_seeds = range(100)

    eps_config = EpsilonScheduleConfig(
        kind="exponential",
        start=1.0,
        minimum=0.05,
        decay=epsilon_decay,
    )

    sarsa_results: list[ComparisonResult] = []
    q_results: list[ComparisonResult] = []

    for seed in training_seeds:
        # SARSA
        sarsa_agent = SarsaAgent(
            _room2_factory,
            SarsaConfig(
                episodes=episodes, alpha=alpha, gamma=gamma,
                max_steps=max_steps, seed=seed, epsilon=eps_config,
            ),
        )
        sarsa_result = sarsa_agent.train()
        sarsa_eval = evaluate_sarsa_policy(
            _room2_factory, sarsa_result.q_values,
            n_episodes=len(eval_seeds), seeds=eval_seeds,
        )
        sarsa_results.append(_to_comparison("SARSA", f"α={alpha},γ={gamma},ε={epsilon_decay}", seed, sarsa_eval))

        # Q-Learning on Room 2
        q_agent = QLearningAgent(
            _room2_factory,
            QLearningConfig(
                episodes=episodes, alpha=alpha, gamma=gamma,
                max_steps=max_steps, seed=seed, epsilon=eps_config,
            ),
        )
        q_result = q_agent.train()
        q_eval = evaluate_q_learning_policy(
            _room2_factory, q_result.q_values,
            n_episodes=len(eval_seeds), seeds=eval_seeds,
        )
        q_results.append(_to_comparison("Q-Learning", f"α={alpha},γ={gamma},ε={epsilon_decay}", seed, q_eval))

    return sarsa_results, q_results


def _to_comparison(algo: str, label: str, seed: int, eval_result) -> ComparisonResult:
    return ComparisonResult(
        algorithm=algo,
        config_label=label,
        seed=seed,
        success_rate=eval_result.success_rate,
        mean_return=eval_result.mean_return,
        std_return=eval_result.std_return,
        mean_steps=eval_result.mean_steps,
        mean_successful_steps=getattr(eval_result, 'mean_successful_steps', None),
        total_collisions=eval_result.total_collisions,
        total_slipped_actions=eval_result.total_slipped_actions,
        total_traps=eval_result.total_traps,
        eval_seeds_used=eval_result.episodes,
    )


def run_tuned_comparison(
    *,
    sarsa_configs: list[dict] | None = None,
    q_configs: list[dict] | None = None,
    training_seeds: list[int] | None = None,
    eval_seeds: range | None = None,
    episodes: int = 5000,
    max_steps: int = 200,
) -> list[TunedComparisonResult]:
    """Comparison B: each algorithm's best-tuned config."""
    if training_seeds is None:
        training_seeds = [0, 1, 2]
    if eval_seeds is None:
        eval_seeds = range(100)

    if sarsa_configs is None:
        sarsa_configs = [
            {"alpha": 0.10, "gamma": 0.95, "epsilon_decay": 0.995},
            {"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.999},
        ]
    if q_configs is None:
        q_configs = [
            {"alpha": 0.10, "gamma": 0.95, "epsilon_decay": 0.995},
            {"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.999},
        ]

    results: list[TunedComparisonResult] = []

    for cfg in sarsa_configs:
        per_seed: list[ComparisonResult] = []
        eps_config = EpsilonScheduleConfig(
            kind="exponential", start=1.0, minimum=0.05,
            decay=cfg["epsilon_decay"],
        )
        for seed in training_seeds:
            agent = SarsaAgent(
                _room2_factory,
                SarsaConfig(
                    episodes=episodes, alpha=cfg["alpha"], gamma=cfg["gamma"],
                    max_steps=max_steps, seed=seed, epsilon=eps_config,
                ),
            )
            result = agent.train()
            ev = evaluate_sarsa_policy(
                _room2_factory, result.q_values,
                n_episodes=len(eval_seeds), seeds=eval_seeds,
            )
            per_seed.append(_to_comparison("SARSA", f"α={cfg['alpha']},γ={cfg['gamma']},ε={cfg['epsilon_decay']}", seed, ev))
        results.append(_aggregate("SARSA", cfg, per_seed))

    for cfg in q_configs:
        per_seed: list[ComparisonResult] = []
        eps_config = EpsilonScheduleConfig(
            kind="exponential", start=1.0, minimum=0.05,
            decay=cfg["epsilon_decay"],
        )
        for seed in training_seeds:
            agent = QLearningAgent(
                _room2_factory,
                QLearningConfig(
                    episodes=episodes, alpha=cfg["alpha"], gamma=cfg["gamma"],
                    max_steps=max_steps, seed=seed, epsilon=eps_config,
                ),
            )
            result = agent.train()
            ev = evaluate_q_learning_policy(
                _room2_factory, result.q_values,
                n_episodes=len(eval_seeds), seeds=eval_seeds,
            )
            per_seed.append(_to_comparison("Q-Learning", f"α={cfg['alpha']},γ={cfg['gamma']},ε={cfg['epsilon_decay']}", seed, ev))
        results.append(_aggregate("Q-Learning", cfg, per_seed))

    return results


def _aggregate(algo: str, config: dict, per_seed: list[ComparisonResult]) -> TunedComparisonResult:
    rates = [r.success_rate for r in per_seed]
    returns = [r.mean_return for r in per_seed]
    steps = [r.mean_steps for r in per_seed]
    colls = sum(r.total_collisions for r in per_seed)
    slips = sum(r.total_slipped_actions for r in per_seed)
    traps = sum(r.total_traps for r in per_seed)

    return TunedComparisonResult(
        algorithm=algo,
        config=config,
        success_rate_mean=float(np.mean(rates)),
        success_rate_std=float(np.std(rates)),
        mean_return_mean=float(np.mean(returns)),
        mean_return_std=float(np.std(returns)),
        mean_steps_mean=float(np.mean(steps)),
        total_collisions=colls,
        total_slipped=slips,
        total_traps=traps,
        per_seed=per_seed,
    )


def save_comparison(
    matched: tuple[list[ComparisonResult], list[ComparisonResult]],
    tuned: list[TunedComparisonResult],
    directory: str = "storage/comparisons",
) -> str:
    os.makedirs(directory, exist_ok=True)

    sarsa_m, q_m = matched
    data = _build_comparison_data(matched, tuned)

    path = os.path.join(directory, "sarsa_vs_q_learning.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def save_comparison_to_final(
    matched: tuple[list[ComparisonResult], list[ComparisonResult]],
    tuned: list[TunedComparisonResult],
) -> str:
    from training.experiment_utils import FINAL_DIR, now_iso, git_commit, room2_map_signature
    os.makedirs(FINAL_DIR, exist_ok=True)

    sarsa_m, q_m = matched
    data = _build_comparison_data(matched, tuned)

    # Compute paired differences
    paired_diffs = []
    for s, q in zip(sarsa_m, q_m):
        paired_diffs.append({
            "seed": s.seed,
            "diff_success_rate": q.success_rate - s.success_rate,
            "diff_mean_return": q.mean_return - s.mean_return,
            "diff_mean_steps": q.mean_steps - s.mean_steps,
        })
    diff_srs = [d["diff_success_rate"] for d in paired_diffs]
    diff_rets = [d["diff_mean_return"] for d in paired_diffs]
    data["matched_comparison"]["paired_differences"] = paired_diffs
    data["matched_comparison"]["paired_diff_mean_success_rate"] = float(np.mean(diff_srs))
    data["matched_comparison"]["paired_diff_std_success_rate"] = float(np.std(diff_srs))
    data["matched_comparison"]["paired_diff_mean_return"] = float(np.mean(diff_rets))
    data["matched_comparison"]["paired_diff_std_return"] = float(np.std(diff_rets))

    data["_metadata"] = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "git_commit": git_commit(),
        "sarsa_vs_q_learning_comparison": True,
        "match_config": {
            "alpha": 0.10, "gamma": 0.95,
            "epsilon_kind": "exponential", "epsilon_start": 1.0,
            "epsilon_minimum": 0.05, "epsilon_decay": 0.995,
            "episodes": 2000, "max_steps": 200,
            "training_seeds": [0, 1, 2, 3, 4],
            "eval_seeds": list(range(100)),
        },
        "map_signature": room2_map_signature(),
    }

    path = os.path.join(FINAL_DIR, "sarsa_vs_q_learning_matched.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # Also save final comparison
    comp_path = os.path.join("storage", "comparisons", "sarsa_vs_q_learning.json")
    with open(comp_path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def _build_comparison_data(
    matched: tuple[list[ComparisonResult], list[ComparisonResult]],
    tuned: list[TunedComparisonResult],
) -> dict:
    sarsa_m, q_m = matched
    return {
        "matched_comparison": {
            "sarsa": [vars(r) for r in sarsa_m],
            "q_learning": [vars(r) for r in q_m],
        },
        "tuned_comparison": [
            {
                "algorithm": r.algorithm,
                "config": r.config,
                "success_rate_mean": r.success_rate_mean,
                "success_rate_std": r.success_rate_std,
                "mean_return_mean": r.mean_return_mean,
                "mean_return_std": r.mean_return_std,
                "mean_steps_mean": r.mean_steps_mean,
                "total_collisions": r.total_collisions,
                "total_slipped": r.total_slipped,
                "total_traps": r.total_traps,
                "per_seed": [vars(ps) for ps in r.per_seed],
            }
            for r in tuned
        ],
    }


def print_summary(
    matched: tuple[list[ComparisonResult], list[ComparisonResult]],
    tuned: list[TunedComparisonResult],
) -> None:
    sarsa_m, q_m = matched

    print("=" * 70)
    print("Comparison A — Matched Parameters")
    print("=" * 70)
    print(f"{'Metric':<35} {'SARSA':>15} {'Q-Learning':>15}")
    print("-" * 70)

    s_sr = [r.success_rate for r in sarsa_m]
    q_sr = [r.success_rate for r in q_m]
    s_ret = [r.mean_return for r in sarsa_m]
    q_ret = [r.mean_return for r in q_m]
    s_steps = [r.mean_steps for r in sarsa_m]
    q_steps = [r.mean_steps for r in q_m]

    print(f"{'Success rate':<35} {np.mean(s_sr):>14.2%} {np.mean(q_sr):>15.2%}")
    print(f"{'Success rate std':<35} {np.std(s_sr):>14.4f} {np.std(q_sr):>15.4f}")
    print(f"{'Mean return':<35} {np.mean(s_ret):>14.2f} {np.mean(q_ret):>15.2f}")
    print(f"{'Mean steps':<35} {np.mean(s_steps):>14.1f} {np.mean(q_steps):>15.1f}")
    print(f"{'Traps':<35} {sum(r.total_traps for r in sarsa_m):>14} {sum(r.total_traps for r in q_m):>15}")

    print()
    print("=" * 70)
    print("Comparison B — Tuned Models")
    print("=" * 70)
    for r in tuned:
        print(f"  {r.algorithm} {r.config}: SR={r.success_rate_mean:.2%}±{r.success_rate_std:.2%}, "
              f"Return={r.mean_return_mean:.1f}±{r.mean_return_std:.1f}, "
              f"Steps={r.mean_steps_mean:.1f}, Traps={r.total_traps}")
