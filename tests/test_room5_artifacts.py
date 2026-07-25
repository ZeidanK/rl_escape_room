"""Regression checks for committed optional Room 5 evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOM5_ARTIFACT = Path("storage/experiments/final/room5_dqn_confirmation.json")


def _assert_finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_room5_final_artifact_is_real_seeded_evidence():
    data = json.loads(ROOM5_ARTIFACT.read_text(encoding="utf-8"))

    _assert_finite(data)
    assert data["schema_version"] == 1
    assert data["room"] == "Room 5 - Dynamic Obstacles"
    assert data["algorithm"] == "NumPy DQN"
    assert data["deployment_considered"] is False
    assert isinstance(data["runtime_seconds"], float)
    assert data["runtime_seconds"] > 0
    assert data["ranking_rule"]

    assert len(data["screening_configs"]) >= 3
    assert data["confirmation_episodes"] > data["screening_episodes"]
    assert data["confirmation_seeds"] == [0, 1, 2, 3, 4]

    seed_results = data["confirmation"]["seed_results"]
    assert len(seed_results) == 5
    for seed_result in seed_results:
        assert seed_result["training"]["finite_weights"] is True
        assert seed_result["training"]["runtime_seconds"] > 0
        for key in ["fixed_layout_evaluation", "random_layout_evaluation", "unseen_layout_evaluation"]:
            evaluation = seed_result[key]
            assert evaluation["episodes"] == 12
            assert 0.0 <= evaluation["success_rate"] <= 1.0
            assert evaluation["rollouts"]

    aggregate = data["confirmation"]["aggregate"]
    assert aggregate["random_success_rate_mean"] > 0.0
    assert aggregate["unseen_success_rate_mean"] > 0.0
    assert aggregate["training_runtime_seconds_sum"] > 0.0

    replays = data["replay_trajectories"]
    assert len(replays) >= 3
    assert all(replay["trajectory"] for replay in replays)
