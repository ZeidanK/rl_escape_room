"""Orchestrator for final multi-seed experiments with resumability."""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from training.experiment_utils import (
    FINAL_DIR,
    benchmark_config,
    git_commit,
    load_completed_results,
    make_base_metadata,
    now_iso,
    package_versions,
    python_version,
    rank_room4,
    rank_sarsa,
    rank_q_learning,
    room2_map_signature,
    room3_map_signature,
    room4_map_signature,
    save_trial,
    trial_id_for_q,
    trial_id_for_room4,
    trial_id_for_sarsa,
)

# ============================================================
# Benchmarking
# ============================================================

def run_benchmark() -> dict:
    print("=" * 60)
    print("  BENCHMARK: Estimating runtime")
    print("=" * 60)
    bench = {}
    for algo, eps in [("sarsa", 2000), ("q_learning", 2000), ("room4", 250), ("room4", 1500)]:
        label = f"{algo}_{eps}"
        print(f"  Benchmarking {label}...", end=" ", flush=True)
        t = benchmark_config(algo, eps)
        print(f"{t:.1f}s")
        bench[label] = round(t, 2)
    print()
    return bench


def print_estimates(bench: dict):
    print("=" * 60)
    print("  RUNTIME ESTIMATES")
    print("=" * 60)
    # Room 2: 36 screening + top 5 x 5 confirmation
    sarsa_screen = bench.get("sarsa_2000", 5.0) * 36
    sarsa_confirm = bench.get("sarsa_2000", 5.0) * 2.5 * 5 * 5
    print(f"  Room 2 SARSA screening:  ~{sarsa_screen:.0f}s")
    print(f"  Room 2 SARSA confirmation: ~{sarsa_confirm:.0f}s")
    # Room 3: same
    q_screen = bench.get("q_learning_2000", 5.0) * 36
    q_confirm = bench.get("q_learning_2000", 5.0) * 2.5 * 5 * 5
    print(f"  Room 3 Q-Learning screening: ~{q_screen:.0f}s")
    print(f"  Room 3 Q-Learning confirmation: ~{q_confirm:.0f}s")
    # Room 4: 15 Stage A + up to 32 Stage B + 5 x 5 x 3000 confirmation
    r4_250 = bench.get("room4_250", 10.0)
    r4_1500 = bench.get("room4_1500", 60.0)
    r4_stage_a = r4_250 * 15
    r4_stage_b = r4_250 * 2 * 32
    r4_confirm = r4_1500 * 5 * 5
    print(f"  Room 4 Stage A:  ~{r4_stage_a:.0f}s")
    print(f"  Room 4 Stage B:  ~{r4_stage_b:.0f}s")
    print(f"  Room 4 confirmation: ~{r4_confirm:.0f}s")
    total = sarsa_screen + sarsa_confirm + q_screen + q_confirm + r4_stage_a + r4_stage_b + r4_confirm
    print(f"  TOTAL (excl Room 1 + comparison): ~{total:.0f}s (~{total/3600:.1f}h)")
    print()


# ============================================================
# Room 1 — Value Iteration
# ============================================================

def run_room1():
    from training.dp_experiments import run_room1_experiments
    print("=" * 60)
    print("  ROOM 1: Value Iteration")
    print("=" * 60)
    results = run_room1_experiments()
    os.makedirs(FINAL_DIR, exist_ok=True)
    filepath = os.path.join(FINAL_DIR, "room1_value_iteration.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved {len(results)} configs to {filepath}")
    if results:
        best = results[0]
        print(f"  Best: gamma={best['gamma']}, tol={best['tolerance']}, "
              f"slip={best['slip_config']}, SR={best['success_rate']:.2%}")
    print()


# ============================================================
# Room 2 — SARSA
# ============================================================

SARSA_FINAL_DIR = os.path.join(FINAL_DIR, "room2_sarsa")

def _sarsa_trial_id(cfg: dict, seed: int) -> str:
    return trial_id_for_sarsa(cfg["alpha"], cfg["gamma"], cfg["decay"], seed)


def _run_sarsa_screening():
    from agents.sarsa import SarsaAgent, evaluate_sarsa_policy
    from core.types import EpsilonScheduleConfig, SarsaConfig
    from environments.room2_sarsa import ROOM2_GRID, Room2SARSA

    ALPHA_VALUES = [0.05, 0.10, 0.30, 0.50]
    GAMMA_VALUES = [0.90, 0.95, 0.99]
    DECAY_VALUES = [0.990, 0.995, 0.999]

    os.makedirs(SARSA_FINAL_DIR, exist_ok=True)
    completed = load_completed_results(SARSA_FINAL_DIR)
    results: list[dict] = []

    for alpha in ALPHA_VALUES:
        for gamma in GAMMA_VALUES:
            for decay in DECAY_VALUES:
                seed = 42
                tid = _sarsa_trial_id({"alpha": alpha, "gamma": gamma, "decay": decay}, seed)
                if any(tid in fn for fn in completed):
                    print(f"  [SKIP] {tid}")
                    continue
                t0 = time.time()
                config = SarsaConfig(
                    episodes=2000, alpha=alpha, gamma=gamma,
                    max_steps=200, seed=seed,
                    epsilon=EpsilonScheduleConfig(
                        kind="exponential", start=1.0, minimum=0.05, decay=decay,
                    ),
                )
                factory = lambda: Room2SARSA(max_steps=200)
                agent = SarsaAgent(factory, config)
                result = agent.train()
                duration = time.time() - t0
                ev = evaluate_sarsa_policy(
                    factory, result.q_values, n_episodes=100,
                )
                trial = {
                    "trial_id": tid, "stage": "screening",
                    "algorithm": "SARSA", "room": "Room2SARSA",
                    "config": {"alpha": alpha, "gamma": gamma, "decay": decay,
                               "episodes": 2000, "seed": seed},
                    "metrics": {
                        "success_rate": ev.success_rate,
                        "mean_return": ev.mean_return,
                        "std_return": ev.std_return,
                        "mean_steps": ev.mean_steps,
                        "mean_successful_steps": ev.mean_successful_steps,
                        "total_collisions": ev.total_collisions,
                        "total_slipped_actions": ev.total_slipped_actions,
                        "total_traps": ev.total_traps,
                    },
                    "runtime_seconds": duration,
                }
                save_trial(os.path.join(SARSA_FINAL_DIR, f"{tid}.json"), trial)
                results.append(trial)
                print(f"  [OK]   {tid} SR={ev.success_rate:.2%} {duration:.1f}s")
    return results


def _run_sarsa_confirmation(top_configs: list[dict]):
    from agents.sarsa import SarsaAgent, evaluate_sarsa_policy
    from core.types import EpsilonScheduleConfig, SarsaConfig
    from environments.room2_sarsa import Room2SARSA

    TRAINING_SEEDS = [0, 1, 2, 3, 4]
    EVAL_SEEDS = list(range(100))

    os.makedirs(SARSA_FINAL_DIR, exist_ok=True)
    completed = load_completed_results(SARSA_FINAL_DIR)
    results: list[dict] = []

    for cfg in top_configs:
        seed_results: list[dict] = []
        for seed in TRAINING_SEEDS:
            tid = _sarsa_trial_id(cfg, seed) + "_conf"
            if any(tid in fn for fn in completed):
                print(f"  [SKIP] {tid}")
                fn = [fn for fn in completed if tid in fn][0]
                seed_results.append(completed[fn])
                continue
            t0 = time.time()
            config = SarsaConfig(
                episodes=5000, alpha=cfg["alpha"], gamma=cfg["gamma"],
                max_steps=200, seed=seed,
                epsilon=EpsilonScheduleConfig(
                    kind="exponential", start=1.0, minimum=0.05, decay=cfg["decay"],
                ),
            )
            factory = lambda: Room2SARSA(max_steps=200)
            agent = SarsaAgent(factory, config)
            result = agent.train()
            duration = time.time() - t0
            ev = evaluate_sarsa_policy(
                factory, result.q_values,
                n_episodes=len(EVAL_SEEDS), seeds=EVAL_SEEDS,
            )
            trial = {
                "trial_id": tid, "stage": "confirmation",
                "algorithm": "SARSA", "room": "Room2SARSA",
                "config": {"alpha": cfg["alpha"], "gamma": cfg["gamma"], "decay": cfg["decay"],
                           "episodes": 5000, "seed": seed},
                "metrics": {
                    "success_rate": ev.success_rate,
                    "mean_return": ev.mean_return,
                    "std_return": ev.std_return,
                    "mean_steps": ev.mean_steps,
                    "mean_successful_steps": ev.mean_successful_steps,
                    "total_collisions": ev.total_collisions,
                    "total_slipped_actions": ev.total_slipped_actions,
                    "total_traps": ev.total_traps,
                },
                "runtime_seconds": duration,
            }
            save_trial(os.path.join(SARSA_FINAL_DIR, f"{tid}.json"), trial)
            seed_results.append(trial)
            print(f"  [OK]   {tid} SR={ev.success_rate:.2%} {duration:.1f}s")

        srs = [r["metrics"]["success_rate"] for r in seed_results]
        mss = [r["metrics"]["mean_successful_steps"] for r in seed_results if r["metrics"]["mean_successful_steps"]]
        mrs = [r["metrics"]["mean_return"] for r in seed_results]
        results.append({
            "config_tuple": (cfg["alpha"], cfg["gamma"], cfg["decay"]),
            "config": cfg,
            "n_seeds": len(TRAINING_SEEDS),
            "mean_success_rate": float(np.mean(srs)),
            "std_success_rate": float(np.std(srs)),
            "mean_successful_steps": float(np.mean(mss)) if mss else None,
            "mean_return": float(np.mean(mrs)),
            "per_seed": seed_results,
        })
    return rank_sarsa(results)


def run_room2():
    print("=" * 60)
    print("  ROOM 2: SARSA")
    print("=" * 60)
    screening = _run_sarsa_screening()
    if screening:
        ranked = rank_sarsa([
            {"mean_success_rate": t["metrics"]["success_rate"],
             "std_success_rate": 0.0,
             "mean_successful_steps": t["metrics"]["mean_successful_steps"],
             "mean_return": t["metrics"]["mean_return"],
             "mean_improvement": 0.0,
             "config_tuple": (t["config"]["alpha"], t["config"]["gamma"], t["config"]["decay"]),
             "config": t["config"]}
            for t in screening
        ])
    else:
        # Reload from disk
        all_fns = sorted(load_completed_results(SARSA_FINAL_DIR).values(),
                         key=lambda x: x.get("trial_id", ""))
        screening = [v for v in all_fns if v.get("stage") == "screening"]
        ranked = rank_sarsa([
            {"mean_success_rate": t["metrics"]["success_rate"],
             "std_success_rate": 0.0,
             "mean_successful_steps": t["metrics"]["mean_successful_steps"],
             "mean_return": t["metrics"]["mean_return"],
             "mean_improvement": 0.0,
             "config_tuple": (t["config"]["alpha"], t["config"]["gamma"], t["config"]["decay"]),
             "config": t["config"]}
            for t in screening
        ])

    top5 = ranked[:5]
    print(f"  Top 5 from screening: {[c['config_tuple'] for c in top5]}")
    confirmation = _run_sarsa_confirmation([c["config"] for c in top5])

    # Save aggregate result
    meta = make_base_metadata(
        algorithm="SARSA", room="Room2SARSA", map_sig=room2_map_signature(),
        config={"alpha_values": [0.05, 0.10, 0.30, 0.50],
                "gamma_values": [0.90, 0.95, 0.99],
                "decay_values": [0.990, 0.995, 0.999],
                "screening_episodes": 2000, "confirmation_episodes": 5000},
        training_seeds=[0, 1, 2, 3, 4],
        evaluation_seeds=list(range(100)),
        ranking_criteria=[
            "mean_success_rate",
            "-std_success_rate",
            "-mean_successful_steps",
            "mean_return",
            "mean_improvement",
            "config_tuple",
        ],
    )
    aggregate = {**meta, "screening_results": screening, "confirmation_results": confirmation}
    save_trial(os.path.join(FINAL_DIR, "room2_sarsa_confirmation.json"), aggregate)
    if confirmation:
        best = confirmation[0]
        print(f"  Best: alpha={best['config']['alpha']}, gamma={best['config']['gamma']}, "
              f"decay={best['config']['decay']}, SR={best['mean_success_rate']:.2%}")
    print()


# ============================================================
# Room 3 — Q-Learning
# ============================================================

Q_FINAL_DIR = os.path.join(FINAL_DIR, "room3_q_learning")

def _q_trial_id(cfg: dict, seed: int) -> str:
    return trial_id_for_q(cfg["alpha"], cfg["gamma"], cfg["decay"], seed)


def _run_q_screening():
    from agents.q_learning import QLearningAgent, evaluate_q_learning_policy
    from core.types import EpsilonScheduleConfig, QLearningConfig
    from environments.room3_qlearning import Room3QLearning

    ALPHA_VALUES = [0.05, 0.10, 0.30, 0.50]
    GAMMA_VALUES = [0.90, 0.95, 0.99]
    DECAY_VALUES = [0.990, 0.995, 0.999]

    os.makedirs(Q_FINAL_DIR, exist_ok=True)
    completed = load_completed_results(Q_FINAL_DIR)
    results: list[dict] = []

    for alpha in ALPHA_VALUES:
        for gamma in GAMMA_VALUES:
            for decay in DECAY_VALUES:
                seed = 42
                tid = _q_trial_id({"alpha": alpha, "gamma": gamma, "decay": decay}, seed)
                if any(tid in fn for fn in completed):
                    print(f"  [SKIP] {tid}")
                    continue
                t0 = time.time()
                config = QLearningConfig(
                    episodes=1000, alpha=alpha, gamma=gamma,
                    max_steps=200, seed=seed,
                    epsilon=EpsilonScheduleConfig(
                        kind="exponential", start=1.0, minimum=0.05, decay=decay,
                    ),
                )
                factory = lambda: Room3QLearning(max_steps=200)
                agent = QLearningAgent(factory, config)
                result = agent.train()
                duration = time.time() - t0
                ev = evaluate_q_learning_policy(
                    factory, result.q_values, n_episodes=100,
                )
                trial = {
                    "trial_id": tid, "stage": "screening",
                    "algorithm": "Q-Learning", "room": "Room3QLearning",
                    "config": {"alpha": alpha, "gamma": gamma, "decay": decay,
                               "episodes": 1000, "seed": seed},
                    "metrics": {
                        "success_rate": ev.success_rate,
                        "mean_return": ev.mean_return,
                        "std_return": ev.std_return,
                        "mean_steps": ev.mean_steps,
                        "mean_successful_steps": ev.mean_successful_steps,
                        "key_collection_rate": ev.key_collection_rate,
                        "mean_key_collection_step": ev.mean_key_collection_step,
                        "total_locked_exit_attempts": ev.total_locked_exit_attempts,
                        "total_collisions": ev.total_collisions,
                        "total_slipped_actions": ev.total_slipped_actions,
                        "total_traps": ev.total_traps,
                    },
                    "runtime_seconds": duration,
                }
                save_trial(os.path.join(Q_FINAL_DIR, f"{tid}.json"), trial)
                results.append(trial)
                print(f"  [OK]   {tid} SR={ev.success_rate:.2%} key={ev.key_collection_rate:.2%} {duration:.1f}s")
    return results


def _run_q_confirmation(top_configs: list[dict]):
    from agents.q_learning import QLearningAgent, evaluate_q_learning_policy
    from core.types import EpsilonScheduleConfig, QLearningConfig
    from environments.room3_qlearning import Room3QLearning

    TRAINING_SEEDS = [0, 1, 2, 3, 4]
    EVAL_SEEDS = list(range(100))

    os.makedirs(Q_FINAL_DIR, exist_ok=True)
    completed = load_completed_results(Q_FINAL_DIR)
    results: list[dict] = []

    for cfg in top_configs:
        seed_results: list[dict] = []
        for seed in TRAINING_SEEDS:
            tid = _q_trial_id(cfg, seed) + "_conf"
            if any(tid in fn for fn in completed):
                print(f"  [SKIP] {tid}")
                fn = [fn for fn in completed if tid in fn][0]
                seed_results.append(completed[fn])
                continue
            t0 = time.time()
            config = QLearningConfig(
                episodes=5000, alpha=cfg["alpha"], gamma=cfg["gamma"],
                max_steps=200, seed=seed,
                epsilon=EpsilonScheduleConfig(
                    kind="exponential", start=1.0, minimum=0.05, decay=cfg["decay"],
                ),
            )
            factory = lambda: Room3QLearning(max_steps=200)
            agent = QLearningAgent(factory, config)
            result = agent.train()
            duration = time.time() - t0
            ev = evaluate_q_learning_policy(
                factory, result.q_values,
                n_episodes=len(EVAL_SEEDS), seeds=EVAL_SEEDS,
            )
            trial = {
                "trial_id": tid, "stage": "confirmation",
                "algorithm": "Q-Learning", "room": "Room3QLearning",
                "config": {"alpha": cfg["alpha"], "gamma": cfg["gamma"], "decay": cfg["decay"],
                           "episodes": 5000, "seed": seed},
                "metrics": {
                    "success_rate": ev.success_rate,
                    "mean_return": ev.mean_return,
                    "std_return": ev.std_return,
                    "mean_steps": ev.mean_steps,
                    "mean_successful_steps": ev.mean_successful_steps,
                    "key_collection_rate": ev.key_collection_rate,
                    "mean_key_collection_step": ev.mean_key_collection_step,
                    "total_locked_exit_attempts": ev.total_locked_exit_attempts,
                    "total_collisions": ev.total_collisions,
                    "total_slipped_actions": ev.total_slipped_actions,
                    "total_traps": ev.total_traps,
                },
                "runtime_seconds": duration,
            }
            save_trial(os.path.join(Q_FINAL_DIR, f"{tid}.json"), trial)
            seed_results.append(trial)
            print(f"  [OK]   {tid} SR={ev.success_rate:.2%} key={ev.key_collection_rate:.2%} {duration:.1f}s")

        srs = [r["metrics"]["success_rate"] for r in seed_results]
        krs = [r["metrics"]["key_collection_rate"] for r in seed_results]
        mks = [r["metrics"]["mean_key_collection_step"] for r in seed_results if r["metrics"]["mean_key_collection_step"]]
        mss = [r["metrics"]["mean_successful_steps"] for r in seed_results if r["metrics"]["mean_successful_steps"]]
        mrs = [r["metrics"]["mean_return"] for r in seed_results]
        results.append({
            "config_tuple": (cfg["alpha"], cfg["gamma"], cfg["decay"]),
            "config": cfg,
            "n_seeds": len(TRAINING_SEEDS),
            "mean_success_rate": float(np.mean(srs)),
            "std_success_rate": float(np.std(srs)),
            "mean_key_collection_rate": float(np.mean(krs)),
            "mean_key_collection_step": float(np.mean(mks)) if mks else None,
            "mean_successful_steps": float(np.mean(mss)) if mss else None,
            "mean_return": float(np.mean(mrs)),
            "per_seed": seed_results,
        })
    return rank_q_learning(results)


def run_room3():
    print("=" * 60)
    print("  ROOM 3: Q-Learning")
    print("=" * 60)
    screening = _run_q_screening()
    if screening:
        ranked = rank_q_learning([
            {"mean_success_rate": t["metrics"]["success_rate"],
             "mean_key_collection_rate": t["metrics"]["key_collection_rate"],
             "std_success_rate": 0.0,
             "mean_successful_steps": t["metrics"]["mean_successful_steps"],
             "mean_key_collection_step": t["metrics"]["mean_key_collection_step"],
             "mean_return": t["metrics"]["mean_return"],
             "config_tuple": (t["config"]["alpha"], t["config"]["gamma"], t["config"]["decay"]),
             "config": t["config"]}
            for t in screening
        ])
    else:
        all_fns = sorted(load_completed_results(Q_FINAL_DIR).values(),
                         key=lambda x: x.get("trial_id", ""))
        screening = [v for v in all_fns if v.get("stage") == "screening"]
        ranked = rank_q_learning([
            {"mean_success_rate": t["metrics"]["success_rate"],
             "mean_key_collection_rate": t["metrics"]["key_collection_rate"],
             "std_success_rate": 0.0,
             "mean_successful_steps": t["metrics"]["mean_successful_steps"],
             "mean_key_collection_step": t["metrics"]["mean_key_collection_step"],
             "mean_return": t["metrics"]["mean_return"],
             "config_tuple": (t["config"]["alpha"], t["config"]["gamma"], t["config"]["decay"]),
             "config": t["config"]}
            for t in screening
        ])

    top5 = ranked[:5]
    print(f"  Top 5 from screening: {[c['config_tuple'] for c in top5]}")
    confirmation = _run_q_confirmation([c["config"] for c in top5])

    meta = make_base_metadata(
        algorithm="Q-Learning", room="Room3QLearning", map_sig=room3_map_signature(),
        config={"alpha_values": [0.05, 0.10, 0.30, 0.50],
                "gamma_values": [0.90, 0.95, 0.99],
                "decay_values": [0.990, 0.995, 0.999],
                "screening_episodes": 1000, "confirmation_episodes": 5000},
        training_seeds=[0, 1, 2, 3, 4],
        evaluation_seeds=list(range(100)),
        ranking_criteria=[
            "mean_success_rate", "mean_key_collection_rate",
            "-std_success_rate", "-mean_successful_steps",
            "-mean_key_collection_step", "mean_return", "config_tuple",
        ],
    )
    aggregate = {**meta, "screening_results": screening, "confirmation_results": confirmation}
    save_trial(os.path.join(FINAL_DIR, "room3_q_learning_confirmation.json"), aggregate)
    if confirmation:
        best = confirmation[0]
        print(f"  Best: alpha={best['config']['alpha']}, gamma={best['config']['gamma']}, "
              f"decay={best['config']['decay']}, SR={best['mean_success_rate']:.2%}")
    print()


# ============================================================
# Room 4 — Approximate SARSA
# ============================================================

ROOM4_FINAL_DIR = os.path.join(FINAL_DIR, "room4_approximate_sarsa")


def _room4_factory(dp_scale=1.0):
    from core.types import StartMode
    from environments.room4_continuous import (
        ContinuousRewardConfig, Room4Continuous, Room4MotionConfig,
    )
    motion = Room4MotionConfig()
    rewards = ContinuousRewardConfig(distance_progress_scale=dp_scale)
    return lambda: Room4Continuous(
        motion_config=motion, reward_config=rewards,
        max_steps=750, start_mode=StartMode.FIXED,
    )


def run_room4():
    from agents.approximate_sarsa import (
        ApproximateSarsaAgent,
        evaluate_approximate_policy_all_categories,
    )
    from core.types import (
        ApproximateSarsaConfig,
        EpsilonScheduleConfig,
        TileCodingConfig,
    )
    from environments.room4_continuous import ContinuousRewardConfig, Room4MotionConfig

    print("=" * 60)
    print("  ROOM 4: Approximate SARSA")
    print("=" * 60)
    os.makedirs(ROOM4_FINAL_DIR, exist_ok=True)
    completed = load_completed_results(ROOM4_FINAL_DIR)

    params_to_test = {
        "num_tilings": [4, 8, 16],
        "tiles_xy": [8, 10, 16],
        "alpha": [0.05, 0.10, 0.20],
        "progress_scale": [0.0, 0.5, 1.0],
        "epsilon_decay": [0.995, 0.997, 0.999],
    }
    defaults = {
        "num_tilings": 8, "tiles_xy": 10, "alpha": 0.10,
        "progress_scale": 1.0, "epsilon_decay": 0.997,
    }

    # Stage A: one factor at a time
    stage_a_results = []
    print("  --- Stage A: One factor at a time ---")
    for param_name, values in params_to_test.items():
        for val in values:
            cfg = {
                "num_tilings": defaults["num_tilings"] if param_name != "num_tilings" else val,
                "tiles_xy": defaults["tiles_xy"] if param_name != "tiles_xy" else val,
                "alpha": defaults["alpha"] if param_name != "alpha" else val,
                "progress_scale": defaults["progress_scale"] if param_name != "progress_scale" else val,
                "epsilon_decay": defaults["epsilon_decay"] if param_name != "epsilon_decay" else val,
                "episodes": 250, "seed": 42,
            }
            tid = trial_id_for_room4(cfg, cfg["seed"])
            if any(tid in fn for fn in completed):
                print(f"  [SKIP] {tid}")
                continue
            t0 = time.time()
            config = ApproximateSarsaConfig(
                episodes=250, alpha=cfg["alpha"], gamma=0.99,
                max_steps=750, seed=42,
                epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.02, decay=cfg["epsilon_decay"]),
                tile_coding=TileCodingConfig(
                    num_tilings=cfg["num_tilings"], tiles_x=cfg["tiles_xy"],
                    tiles_y=cfg["tiles_xy"], include_velocity=True,
                ),
            )
            factory = _room4_factory(cfg["progress_scale"])
            agent = ApproximateSarsaAgent(factory, config)
            result = agent.train()
            duration = time.time() - t0
            cats = evaluate_approximate_policy_all_categories(
                factory, result.weights, config.tile_coding, Room4MotionConfig(),
                n_episodes=25,
            )
            trial = {
                "trial_id": tid, "stage": "stage_a",
                "algorithm": "Approximate SARSA", "room": "Room4Continuous",
                "config": cfg,
                "categories": {k: _cat_to_dict(v) for k, v in cats.items()},
                "runtime_seconds": duration,
            }
            save_trial(os.path.join(ROOM4_FINAL_DIR, f"{tid}.json"), trial)
            stage_a_results.append(trial)
            print(f"  [OK]   {tid} {duration:.1f}s")

    # Stage B: combine best 2 per factor
    print("  --- Stage B: Combine best 2 per factor ---")
    best_of = {}
    # compute best from stage_a_results or reloaded
    all_stage_a = stage_a_results or [
        v for v in load_completed_results(ROOM4_FINAL_DIR).values()
        if v.get("stage") == "stage_a"
    ]
    for param in ["num_tilings", "tiles_xy", "alpha", "progress_scale", "epsilon_decay"]:
        scored: dict = {}
        for r in all_stage_a:
            key = r["config"][param]
            sr = r.get("categories", {}).get("fixed_training_start", {}).get("success_rate", 0.0)
            if key not in scored:
                scored[key] = []
            scored[key].append(sr)
        means = {k: float(np.mean(v)) for k, v in scored.items()}
        sorted_params = sorted(means.keys(), key=lambda k: means[k], reverse=True)
        best_of[param] = sorted_params[:2]
        print(f"    {param}: best 2 = {best_of[param]}")

    from itertools import product
    from environments.room4_continuous import Room4MotionConfig as R4Config

    stage_b_results = []
    for combo in product(*list(best_of.values())):
        nt, tx, alp, ps, ed = combo
        cfg = {
            "num_tilings": nt, "tiles_xy": tx, "alpha": alp,
            "progress_scale": ps, "epsilon_decay": ed,
            "episodes": 500, "seed": 42,
        }
        tid = trial_id_for_room4(cfg, cfg["seed"])
        if any(tid in fn for fn in completed):
            print(f"  [SKIP] {tid}")
            continue
        t0 = time.time()
        config = ApproximateSarsaConfig(
            episodes=500, alpha=alp, gamma=0.99,
            max_steps=750, seed=42,
            epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.02, decay=ed),
            tile_coding=TileCodingConfig(
                num_tilings=nt, tiles_x=tx, tiles_y=tx, include_velocity=True,
            ),
        )
        factory = _room4_factory(ps)
        agent = ApproximateSarsaAgent(factory, config)
        result = agent.train()
        duration = time.time() - t0
        cats = evaluate_approximate_policy_all_categories(
            factory, result.weights, config.tile_coding, R4Config(),
            n_episodes=25,
        )
        trial = {
            "trial_id": tid, "stage": "stage_b",
            "algorithm": "Approximate SARSA", "room": "Room4Continuous",
            "config": cfg,
            "categories": {k: _cat_to_dict(v) for k, v in cats.items()},
            "runtime_seconds": duration,
        }
        save_trial(os.path.join(ROOM4_FINAL_DIR, f"{tid}.json"), trial)
        stage_b_results.append(trial)
        print(f"  [OK]   {tid} {duration:.1f}s")

    # Pick top 5 from stage B for confirmation
    all_stage_b = stage_b_results or [
        v for v in load_completed_results(ROOM4_FINAL_DIR).values()
        if v.get("stage") == "stage_b"
    ]
    ranked_b = rank_room4([
        {"fixed_training_start_success_rate": r.get("categories", {}).get("fixed_training_start", {}).get("success_rate", 0.0),
         "fixed_unseen_starts_success_rate": r.get("categories", {}).get("fixed_unseen_starts", {}).get("success_rate", 0.0),
         "random_lower_left_success_rate": r.get("categories", {}).get("random_lower_left", {}).get("success_rate", 0.0),
         "random_room_success_rate": r.get("categories", {}).get("random_room", {}).get("success_rate", 0.0),
         "truncation_count": r.get("categories", {}).get("fixed_training_start", {}).get("truncated_count", 9999),
         "mean_successful_steps": r.get("categories", {}).get("fixed_training_start", {}).get("mean_successful_steps", None),
         "mean_return": r.get("categories", {}).get("fixed_training_start", {}).get("mean_return", -9999),
         "std_return": r.get("categories", {}).get("fixed_training_start", {}).get("std_return", 9999),
         "config_tuple": (r["config"]["num_tilings"], r["config"]["tiles_xy"],
                          r["config"]["alpha"], r["config"]["progress_scale"],
                          r["config"]["epsilon_decay"]),
         "config": r["config"]}
        for r in all_stage_b
    ])
    top5 = ranked_b[:5]
    print(f"  Top 5 from Stage B: {[c['config_tuple'] for c in top5]}")

    # Confirmation
    print("  --- Confirmation ---")
    TRAINING_SEEDS = [0, 1, 2, 3, 4]
    confirmation_results = []
    for cfg_dict in [c["config"] for c in top5]:
        seed_results = []
        for seed in TRAINING_SEEDS:
            trial_cfg = {**cfg_dict, "episodes": 1500, "seed": seed}
            tid = trial_id_for_room4(trial_cfg, seed) + "_conf"
            if any(tid in fn for fn in completed):
                print(f"  [SKIP] {tid}")
                continue
            t0 = time.time()
            config = ApproximateSarsaConfig(
                episodes=1500, alpha=trial_cfg["alpha"], gamma=0.99,
                max_steps=750, seed=seed,
                epsilon=EpsilonScheduleConfig(start=1.0, minimum=0.02, decay=trial_cfg["epsilon_decay"]),
                tile_coding=TileCodingConfig(
                    num_tilings=trial_cfg["num_tilings"], tiles_x=trial_cfg["tiles_xy"],
                    tiles_y=trial_cfg["tiles_xy"], include_velocity=True,
                ),
            )
            factory = _room4_factory(trial_cfg["progress_scale"])
            agent = ApproximateSarsaAgent(factory, config)
            result = agent.train()
            duration = time.time() - t0
            cats = evaluate_approximate_policy_all_categories(
                factory, result.weights, config.tile_coding, R4Config(),
                n_episodes=25,
            )
            trial = {
                "trial_id": tid, "stage": "confirmation",
                "algorithm": "Approximate SARSA", "room": "Room4Continuous",
                "config": trial_cfg,
                "categories": {k: _cat_to_dict(v) for k, v in cats.items()},
                "runtime_seconds": duration,
            }
            save_trial(os.path.join(ROOM4_FINAL_DIR, f"{tid}.json"), trial)
            seed_results.append(trial)
            print(f"  [OK]   {tid} {duration:.1f}s")

        if seed_results:
            ft_srs = [r["categories"]["fixed_training_start"]["success_rate"] for r in seed_results]
            fu_srs = [r["categories"]["fixed_unseen_starts"]["success_rate"] for r in seed_results]
            rl_srs = [r["categories"]["random_lower_left"]["success_rate"] for r in seed_results]
            rr_srs = [r["categories"]["random_room"]["success_rate"] for r in seed_results]
            confirmation_results.append({
                "config_tuple": (cfg_dict["num_tilings"], cfg_dict["tiles_xy"],
                                 cfg_dict["alpha"], cfg_dict["progress_scale"],
                                 cfg_dict["epsilon_decay"]),
                "config": cfg_dict,
                "n_seeds": len(TRAINING_SEEDS),
                "fixed_training_start_success_rate": float(np.mean(ft_srs)),
                "fixed_unseen_starts_success_rate": float(np.mean(fu_srs)),
                "random_lower_left_success_rate": float(np.mean(rl_srs)),
                "random_room_success_rate": float(np.mean(rr_srs)),
                "per_seed": seed_results,
            })

    confirmation_results = rank_room4(confirmation_results)
    meta = make_base_metadata(
        algorithm="Approximate SARSA", room="Room4Continuous",
        map_sig=room4_map_signature(),
        config={
            "params_tested": {k: v for k, v in params_to_test.items()},
            "defaults": defaults,
            "stage_a_episodes": 250, "stage_b_episodes": 500,
            "confirmation_episodes": 1500,
        },
        training_seeds=TRAINING_SEEDS,
        evaluation_seeds=list(range(1, 26)),
        ranking_criteria=[
            "fixed_training_start_success_rate",
            "fixed_unseen_starts_success_rate",
            "random_lower_left_success_rate",
            "random_room_success_rate",
            "-truncation_count", "-mean_successful_steps",
            "mean_return", "-std_return", "config_tuple",
        ],
        motion_config={"room_width_m": 10.0, "room_height_m": 10.0,
                        "time_step_s": 0.02, "exit_center": (9.5, 9.5),
                        "exit_radius_m": 0.35},
    )
    aggregate = {
        **meta,
        "stage_a_results": all_stage_a,
        "stage_b_results": all_stage_b,
        "confirmation_results": confirmation_results,
    }
    save_trial(os.path.join(FINAL_DIR, "room4_approximate_sarsa_confirmation.json"), aggregate)
    if confirmation_results:
        best = confirmation_results[0]
        print(f"  Best config: {best['config_tuple']}")
        print(f"    Fixed SR: {best['fixed_training_start_success_rate']:.2%}")
        print(f"    Unseen SR: {best['fixed_unseen_starts_success_rate']:.2%}")
    print()


def _cat_to_dict(summary) -> dict:
    return {
        "n_episodes": summary.n_episodes,
        "success_rate": summary.success_rate,
        "mean_return": summary.mean_return,
        "std_return": summary.std_return,
        "mean_steps": summary.mean_steps,
        "mean_successful_steps": summary.mean_successful_steps,
        "truncated_count": summary.truncated_count,
        "total_collisions": summary.total_collisions,
        "mean_distance_travelled_m": summary.mean_distance_travelled_m,
    }


# ============================================================
# SARSA vs Q-Learning Comparison
# ============================================================

def run_comparison():
    print("=" * 60)
    print("  SARSA vs Q-Learning Comparison")
    print("=" * 60)
    from training.algorithm_comparison import (
        run_matched_comparison,
        run_tuned_comparison,
        save_comparison_to_final,
    )

    # Matched — seeds 0-4, identical config
    matched = run_matched_comparison(
        alpha=0.10, gamma=0.95, episodes=2000,
        epsilon_decay=0.995,
        training_seeds=[0, 1, 2, 3, 4],
        eval_seeds=range(100),
        max_steps=200,
    )

    # Tuned — best from confirmation results
    sarsa_file = os.path.join(FINAL_DIR, "room2_sarsa_confirmation.json")
    q_file = os.path.join(FINAL_DIR, "room3_q_learning_confirmation.json")

    sarsa_best = {"alpha": 0.10, "gamma": 0.95, "epsilon_decay": 0.995}
    q_best = {"alpha": 0.10, "gamma": 0.95, "epsilon_decay": 0.995}

    if os.path.exists(sarsa_file):
        with open(sarsa_file) as f:
            sarsa_data = json.load(f)
        if sarsa_data.get("confirmation_results"):
            best_cfg = sarsa_data["confirmation_results"][0]["config"]
            sarsa_best = {"alpha": best_cfg["alpha"], "gamma": best_cfg["gamma"],
                          "epsilon_decay": best_cfg["decay"]}
    if os.path.exists(q_file):
        with open(q_file) as f:
            q_data = json.load(f)
        if q_data.get("confirmation_results"):
            best_cfg = q_data["confirmation_results"][0]["config"]
            q_best = {"alpha": best_cfg["alpha"], "gamma": best_cfg["gamma"],
                      "epsilon_decay": best_cfg["decay"]}

    tuned = run_tuned_comparison(
        sarsa_configs=[sarsa_best],
        q_configs=[q_best],
        training_seeds=[0, 1, 2, 3, 4],
        eval_seeds=range(100),
        episodes=5000,
        max_steps=200,
    )

    # Save to final
    from training.algorithm_comparison import save_comparison
    save_comparison_to_final(matched, tuned)

    # Print summary
    from training.algorithm_comparison import print_summary
    print_summary(matched, tuned)
    print()


# ============================================================
# Summary CSV
# ============================================================

def generate_summary_csv():
    import csv
    rows = []
    commit = git_commit()

    # Room 1
    r1_file = os.path.join(FINAL_DIR, "room1_value_iteration.json")
    if os.path.exists(r1_file):
        with open(r1_file) as f:
            try:
                r1 = json.load(f)
                if r1:
                    best = r1[0]
                    rows.append({
                        "room": "Room 1", "algorithm": "Value Iteration",
                        "best_config_id": f"gamma={best['gamma']},tol={best['tolerance']},slip={best['slip_config']}",
                        "training_episodes": "N/A", "training_seed_count": "N/A",
                        "evaluation_count": 100,
                        "success_rate_mean": best["success_rate"],
                        "success_rate_std": 0.0,
                        "mean_return": best["mean_return"],
                        "mean_successful_steps": best.get("mean_successful_steps") or "N/A",
                        "key_collection_rate": "N/A",
                        "fixed_unseen_success_rate": "N/A",
                        "random_room_success_rate": "N/A",
                        "runtime_seconds": "N/A",
                        "result_file": "room1_value_iteration.json",
                        "git_commit": commit,
                    })
            except Exception:
                pass

    # Room 2
    r2_file = os.path.join(FINAL_DIR, "room2_sarsa_confirmation.json")
    if os.path.exists(r2_file):
        with open(r2_file) as f:
            try:
                r2 = json.load(f)
                if r2.get("confirmation_results"):
                    best = r2["confirmation_results"][0]
                    config_meta = r2.get("config", {})
                    rows.append({
                        "room": "Room 2", "algorithm": "SARSA",
                        "best_config_id": f"alpha={best['config']['alpha']},gamma={best['config']['gamma']},decay={best['config']['decay']}",
                        "training_episodes": config_meta.get("confirmation_episodes", best["config"].get("episodes", 5000)),
                        "training_seed_count": len(r2.get("training_seeds", [])) or best["n_seeds"],
                        "evaluation_count": len(r2.get("evaluation_seeds", [])) or 100,
                        "success_rate_mean": best["mean_success_rate"],
                        "success_rate_std": best["std_success_rate"],
                        "mean_return": best["mean_return"],
                        "mean_successful_steps": best.get("mean_successful_steps") or "N/A",
                        "key_collection_rate": "N/A",
                        "fixed_unseen_success_rate": "N/A",
                        "random_room_success_rate": "N/A",
                        "runtime_seconds": "N/A",
                        "result_file": "room2_sarsa_confirmation.json",
                        "git_commit": r2.get("git_commit", commit),
                    })
            except Exception:
                pass

    # Room 3
    r3_file = os.path.join(FINAL_DIR, "room3_q_learning_confirmation.json")
    if os.path.exists(r3_file):
        with open(r3_file) as f:
            try:
                r3 = json.load(f)
                if r3.get("confirmation_results"):
                    best = r3["confirmation_results"][0]
                    config_meta = r3.get("config", {})
                    rows.append({
                        "room": "Room 3", "algorithm": "Q-Learning",
                        "best_config_id": f"alpha={best['config']['alpha']},gamma={best['config']['gamma']},decay={best['config']['decay']}",
                        "training_episodes": config_meta.get("confirmation_episodes", best["config"].get("episodes", 5000)),
                        "training_seed_count": len(r3.get("training_seeds", [])) or best["n_seeds"],
                        "evaluation_count": len(r3.get("evaluation_seeds", [])) or 100,
                        "success_rate_mean": best["mean_success_rate"],
                        "success_rate_std": best["std_success_rate"],
                        "mean_return": best["mean_return"],
                        "mean_successful_steps": best.get("mean_successful_steps") or "N/A",
                        "key_collection_rate": best.get("mean_key_collection_rate") or "N/A",
                        "fixed_unseen_success_rate": "N/A",
                        "random_room_success_rate": "N/A",
                        "runtime_seconds": "N/A",
                        "result_file": "room3_q_learning_confirmation.json",
                        "git_commit": r3.get("git_commit", commit),
                    })
            except Exception:
                pass

    # Room 4
    r4_file = os.path.join(FINAL_DIR, "room4_approximate_sarsa_confirmation.json")
    if os.path.exists(r4_file):
        with open(r4_file) as f:
            try:
                r4 = json.load(f)
                if r4.get("confirmation_results"):
                    best = r4["confirmation_results"][0]
                    config_meta = r4.get("config", {})
                    rows.append({
                        "room": "Room 4", "algorithm": "Approximate SARSA",
                        "best_config_id": f"nt={best['config']['num_tilings']},tx={best['config']['tiles_xy']},alpha={best['config']['alpha']},ps={best['config']['progress_scale']},ed={best['config']['epsilon_decay']}",
                        "training_episodes": config_meta.get("confirmation_episodes", best["config"].get("episodes", 1500)),
                        "training_seed_count": len(r4.get("training_seeds", [])) or best["n_seeds"],
                        "evaluation_count": len(r4.get("evaluation_seeds", [])) or "N/A",
                        "success_rate_mean": best["fixed_training_start_success_rate"],
                        "success_rate_std": 0.0,
                        "mean_return": "N/A",
                        "mean_successful_steps": "N/A",
                        "key_collection_rate": "N/A",
                        "fixed_unseen_success_rate": best["fixed_unseen_starts_success_rate"],
                        "random_room_success_rate": best["random_room_success_rate"],
                        "runtime_seconds": "N/A",
                        "result_file": "room4_approximate_sarsa_confirmation.json",
                        "git_commit": r4.get("git_commit", commit),
                    })
            except Exception:
                pass

    csv_path = os.path.join(FINAL_DIR, "final_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "room", "algorithm", "best_config_id", "training_episodes",
        "training_seed_count", "evaluation_count", "success_rate_mean",
        "success_rate_std", "mean_return", "mean_successful_steps",
        "key_collection_rate", "fixed_unseen_success_rate",
        "random_room_success_rate", "runtime_seconds", "result_file", "git_commit",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary CSV saved to {csv_path}")


# ============================================================
# Main pipeline
# ============================================================

def run_all(benchmark_only: bool = False):
    print(f"Final Experiment Pipeline — {datetime.now().isoformat()}")
    print(f"Git commit: {git_commit()}")
    print()

    bench = run_benchmark()
    print_estimates(bench)
    if benchmark_only:
        return

    run_room1()
    run_room2()
    run_room3()
    run_room4()
    run_comparison()
    generate_summary_csv()

    print("=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    benchmark = "--benchmark-only" in sys.argv
    run_all(benchmark_only=benchmark)
