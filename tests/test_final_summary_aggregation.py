"""Tests for final-pipeline summary aggregation helpers."""

import pytest

from training.run_final_pipeline import (
    _room4_category_success_std,
    _room5_eval_success_std,
    _room5_evaluation_count_label,
)


def test_room4_fixed_success_std_uses_per_seed_values():
    result = {
        "per_seed": [
            {"categories": {"fixed_training_start": {"success_rate": value}}}
            for value in [1.0, 0.0, 1.0, 1.0, 0.0]
        ]
    }

    assert _room4_category_success_std(result, "fixed_training_start") == pytest.approx(
        0.489897948556636
    )


def test_room5_random_success_std_and_count_label_use_confirmation_seeds():
    artifact = {
        "confirmation_seeds": [0, 1, 2, 3, 4],
        "confirmation": {
            "seed_results": [
                {"random_layout_evaluation": {"success_rate": value, "episodes": 12}}
                for value in [
                    1.0 / 6.0,
                    11.0 / 12.0,
                    2.0 / 3.0,
                    7.0 / 12.0,
                    1.0,
                ]
            ]
        },
    }

    assert _room5_eval_success_std(artifact, "random_layout_evaluation") == pytest.approx(
        0.293446947694317
    )
    assert _room5_evaluation_count_label(artifact) == "5x12"
