"""Expose visualization helpers used by Streamlit analysis views."""

# Convenience exports for visualization helpers used by app.py and game views.
from visualization.dp_visualization import (
    build_policy_symbols,
    build_value_matrix,
    render_trajectory_overlay,
)
from visualization.sarsa_visualization import (
    build_greedy_policy_symbols,
    build_q_value_tables,
    build_training_dataframe,
    render_sarsa_trajectory_overlay,
)
from visualization.q_learning_visualization import (
    build_q_learning_training_dataframe,
    build_room3_policy_symbols,
    build_room3_q_value_table,
    render_q_learning_trajectory_overlay,
)
from visualization.approximate_sarsa_visualization import (
    build_action_field,
    build_training_dataframe as build_approx_training_dataframe,
    build_value_surface,
    render_continuous_trajectory,
)
