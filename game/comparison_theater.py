"""Side-by-side SARSA vs Q-Learning comparison theater."""

import streamlit as st
import numpy as np
import os
import json
import hashlib


def render_comparison_theater():
    # Interactive Streamlit view for running or displaying the SARSA vs
    # Q-Learning comparison.
    st.markdown("## SARSA vs Q-Learning — Visual Race")
    st.markdown(
        '<div style="color:#90a4ae;font-size:0.9em;">'
        "Same map &middot; Same rewards &middot; Same slip probabilities &middot; "
        "Same seeds &middot; Same training budget"
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar controls for standalone comparison
    with st.sidebar:
        st.markdown("### Comparison Settings")
        comp_episodes = st.number_input("Episodes", min_value=100, max_value=10000, value=2000, step=500,
                                         key="comp_episodes",
                                         help="Training episodes per algorithm per seed.")
        comp_alpha = st.slider("Alpha", 0.01, 1.0, 0.10, step=0.01, key="comp_alpha",
                               help="Learning rate for both algorithms.")
        comp_gamma = st.slider("Gamma", 0.50, 0.99, 0.95, step=0.01, key="comp_gamma",
                               help="Discount factor for both algorithms.")
        comp_decay = st.slider("Epsilon Decay", 0.9, 1.0, 0.995, step=0.001, key="comp_decay",
                               help="Exponential epsilon decay rate per episode.")
        comp_seeds = st.number_input("Training Seeds", min_value=1, max_value=10, value=5, step=1,
                                      key="comp_seeds",
                                      help="Number of random seeds to average over. More = more reliable but slower.")
        comp_eval_ep = st.number_input("Eval Episodes per Model", min_value=10, max_value=500, value=100, step=10,
                                        key="comp_eval_ep",
                                        help="Evaluation episodes per trained model per seed.")

        if st.button("Run Comparison", type="primary", key="comp_run"):
            _run_comparison(comp_episodes, comp_alpha, comp_gamma, comp_decay, comp_seeds, comp_eval_ep)

    # Load and display existing results
    comp_dir = os.path.join("storage", "comparisons")
    comp_file = os.path.join(comp_dir, "sarsa_vs_q_learning.json")
    
    # Check if we have results in session state from a fresh run
    if "comp_matched" in st.session_state and "comp_tuned" in st.session_state:
        matched = st.session_state.comp_matched
        tuned = st.session_state.comp_tuned
    elif os.path.exists(comp_file):
        with open(comp_file) as f:
            data = json.load(f)
        matched = data.get("matched_comparison", {})
        tuned = data.get("tuned_comparison", {})
    else:
        matched = {}
        tuned = {}

    if matched and matched.get("sarsa") and matched.get("q_learning"):
        _render_comparison_results(matched, tuned)
    else:
        st.info("No comparison data found. Configure settings in the sidebar and click **Run Comparison**.")


def _run_comparison(episodes, alpha, gamma, decay, seeds, eval_ep):
    """Run the comparison and store results in session state."""
    # The comparison can be slow, so results are saved both to session state and
    # disk for later viewing.
    from training.algorithm_comparison import run_matched_comparison, run_tuned_comparison, save_comparison
    
    with st.spinner("Running matched comparison..."):
        matched = run_matched_comparison(
            alpha=alpha, gamma=gamma,
            episodes=episodes, epsilon_decay=decay,
            training_seeds=list(range(seeds)),
            eval_seeds=range(eval_ep),
        )
        st.session_state.comp_matched = matched

    with st.spinner("Running tuned comparison..."):
        tuned = run_tuned_comparison(
            sarsa_configs=[
                {"alpha": alpha, "gamma": gamma, "epsilon_decay": decay},
                {"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.999},
            ],
            q_configs=[
                {"alpha": alpha, "gamma": gamma, "epsilon_decay": decay},
                {"alpha": 0.05, "gamma": 0.95, "epsilon_decay": 0.999},
            ],
            training_seeds=list(range(min(3, seeds))),
            eval_seeds=range(eval_ep),
            episodes=episodes,
        )
        st.session_state.comp_tuned = tuned

    save_comparison(matched, tuned)
    st.session_state.comp_matched = matched
    st.session_state.comp_tuned = tuned
    st.success("Comparison complete!")
    st.rerun()


def _render_comparison_results(matched, tuned):
    """Render the comparison results."""
    # Display paired seed results side-by-side before showing aggregate metrics.
    sarsa_list = matched.get("sarsa", [])
    q_list = matched.get("q_learning", [])
    
    if not sarsa_list or not q_list:
        st.warning("Incomplete comparison data.")
        return

    n = min(len(sarsa_list), len(q_list))
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<div style="text-align:center;padding:10px;background:rgba(239,83,80,0.1);'
            'border:1px solid #ef5350;border-radius:10px;">'
            "<h3 style='color:#ef5350;margin:0;'>SARSA</h3></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="text-align:center;padding:10px;background:rgba(66,165,245,0.1);'
            'border:1px solid #42a5f5;border-radius:10px;">'
            "<h3 style='color:#42a5f5;margin:0;'>Q-Learning</h3></div>",
            unsafe_allow_html=True,
        )

    for i in range(n):
        s = sarsa_list[i]
        q = q_list[i]
        with st.container():
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;margin:4px;">'
                    f"<strong>Seed {s['seed']}</strong><br>"
                    f"SR: {s.get('success_rate', 0):.1%}<br>"
                    f"Return: {s.get('mean_return', 0):.1f}<br>"
                    f"Steps: {s.get('mean_steps', 0):.1f}<br>"
                    f"Traps: {s.get('total_traps', 0)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:10px;margin:4px;">'
                    f"<strong>Seed {q['seed']}</strong><br>"
                    f"SR: {q.get('success_rate', 0):.1%}<br>"
                    f"Return: {q.get('mean_return', 0):.1f}<br>"
                    f"Steps: {q.get('mean_steps', 0):.1f}<br>"
                    f"Traps: {q.get('total_traps', 0)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Aggregate
    s_sr = [r.get("success_rate", 0) for r in sarsa_list]
    q_sr = [r.get("success_rate", 0) for r in q_list]
    s_ret = [r.get("mean_return", 0) for r in sarsa_list]
    q_ret = [r.get("mean_return", 0) for r in q_list]

    st.markdown("### Aggregate Results")
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("SARSA Mean SR", f"{np.mean(s_sr):.1%}")
    ac2.metric("Q-Learning Mean SR", f"{np.mean(q_sr):.1%}")
    ac3.metric("Mean Diff (Q - S)", f"{np.mean(np.array(q_sr) - np.array(s_sr)):.1%}")

    ac1.metric("SARSA Mean Return", f"{np.mean(s_ret):.1f}")
    ac2.metric("Q-Learning Mean Return", f"{np.mean(q_ret):.1f}")
    ac3.metric("Mean Return Diff", f"{np.mean(np.array(q_ret) - np.array(s_ret)):.1f}")

    # Tuned comparison if available
    if tuned:
        st.markdown("### Tuned Model Comparison")
        st.dataframe(tuned, use_container_width=True)


def _velocity_arrow(vx: int, vy: int) -> str:
    """Convert velocity to arrow symbol."""
    if vx == 0 and vy == 1:
        return "\u2191"  # UP
    elif vx == 0 and vy == -1:
        return "\u2193"  # DOWN
    elif vx == 1 and vy == 0:
        return "\u2192"  # RIGHT
    elif vx == -1 and vy == 0:
        return "\u2190"  # LEFT
    elif vx == 1 and vy == 1:
        return "\u2197"  # UP-RIGHT
    elif vx == 1 and vy == -1:
        return "\u2198"  # DOWN-RIGHT
    elif vx == -1 and vy == 1:
        return "\u2196"  # UP-LEFT
    elif vx == -1 and vy == -1:
        return "\u2199"  # DOWN-LEFT
    return "*"
