import copy
import json
import os
from datetime import datetime

from core.types import SlipConfig, ValueIterationConfig
from environments.room1_dp import Room1DP
from agents.dynamic_programming import (
    ValueIterationAgent,
    evaluate_policy,
)


STORAGE_DIR = os.path.join("storage", "experiments", "room1_dp")


GAMMA_VALUES = [0.80, 0.90, 0.95, 0.99]
TOLERANCE_VALUES = [1e-2, 1e-4, 1e-6]
SLIP_CONFIGS: list[tuple[str, SlipConfig]] = [
    ("deterministic", SlipConfig(1.0, 0.0, 0.0)),
    ("default", SlipConfig(0.8, 0.1, 0.1)),
    ("high-slip", SlipConfig(0.6, 0.2, 0.2)),
]


def create_fresh_env(*, slip_config: SlipConfig) -> Room1DP:
    return Room1DP(slip_config=slip_config, max_steps=200)


def run_room1_experiments() -> list[dict]:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    results: list[dict] = []
    eval_seeds = range(100)

    for slip_name, slip_cfg in SLIP_CONFIGS:
        for gamma in GAMMA_VALUES:
            for tolerance in TOLERANCE_VALUES:
                env = create_fresh_env(slip_config=slip_cfg)
                config = ValueIterationConfig(
                    gamma=gamma,
                    tolerance=tolerance,
                    max_iterations=10_000,
                )
                agent = ValueIterationAgent(env, config)
                vi_result = agent.solve()

                # Evaluate using fresh env
                eval_env = create_fresh_env(slip_config=slip_cfg)
                summary = evaluate_policy(
                    eval_env,
                    vi_result.policy,
                    n_episodes=100,
                    seeds=eval_seeds,
                )

                record = {
                    "gamma": gamma,
                    "tolerance": tolerance,
                    "slip_config": slip_name,
                    "intended_prob": slip_cfg.intended_probability,
                    "left_prob": slip_cfg.left_probability,
                    "right_prob": slip_cfg.right_probability,
                    "converged": vi_result.converged,
                    "iterations": vi_result.iterations,
                    "final_delta": vi_result.final_delta,
                    "start_state_value": vi_result.start_state_value,
                    "success_rate": summary.success_rate,
                    "mean_return": summary.mean_return,
                    "mean_steps": summary.mean_steps,
                    "mean_successful_steps": summary.mean_successful_steps,
                    "std_return": summary.std_return,
                    "std_steps": summary.std_steps,
                }
                results.append(record)

    # Sort by ranking criteria: converged > success_rate > -mean_successful_steps > mean_return > iterations
    results.sort(key=lambda r: (
        r["converged"] is True,
        r["success_rate"] if r["converged"] else -1.0,
        -(r["mean_successful_steps"] if r["mean_successful_steps"] is not None else 9999),
        r["mean_return"] if r["converged"] else -9999,
        -r["iterations"],
    ), reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"value_iteration_{timestamp}.json"
    filepath = os.path.join(STORAGE_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results
