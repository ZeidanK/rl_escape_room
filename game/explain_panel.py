"""Explain-this-decision panel — shows Q-values and rationale for each action."""

import html as html_mod

from core.types import Action


ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTION_SYMBOLS = {a: s for a, s in zip(Action, ["\u2191", "\u2192", "\u2193", "\u2190"])}


def render_explain_panel(
    q_values: dict[str, float] | None = None,
    *,
    selected_action: Action | str | None = None,
    algorithm: str = "Value Iteration",
    explanation: str | None = None,
) -> str:
    # Builds a compact HTML panel.  Escaping user-visible values keeps the
    # generated markup safe even when labels come from metadata.
    html = '<div class="explain-panel">'
    html += '<h4>Why this action?</h4>'

    if q_values:
        html += '<table>'
        html += "<tr><th>Action</th><th>Q-Value</th></tr>"
        selected_name = Action(selected_action).name if isinstance(selected_action, Action) else str(selected_action).upper() if selected_action else None

        sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        for name, val in sorted_actions:
            cls = ' class="selected"' if name == selected_name else ""
            arrow = ACTION_SYMBOLS.get(Action[name], "") if name in Action.__members__ else ""
            html += f"<tr{cls}><td>{html_mod.escape(arrow)} {html_mod.escape(name)}</td><td>{val:.2f}</td></tr>"
        html += "</table>"

    if explanation:
        html += f'<p style="color:#b0bec5;font-size:0.85em;margin-top:8px;">{html_mod.escape(explanation)}</p>'

    html += "</div>"
    return html


def get_algorithm_explanation(algorithm_key: str) -> str:
    # Short plain-English explanations shown beside Q-values in the game view.
    explanations = {
        "vi": "The action maximizes expected reward using the known transition model (Dynamic Programming). "
               "Value Iteration computes the optimal policy by repeatedly applying the Bellman optimality operator.",
        "sarsa": "The selected action follows the current epsilon-greedy policy. "
                  "SARSA is on-policy: it learns the value of the policy it currently follows, "
                  "making it risk-aware in stochastic environments.",
        "q_learning": "The update target uses the maximum next-state Q-value (off-policy). "
                       "Q-Learning directly approximates the optimal action-value function "
                       "independent of the behaviour policy.",
        "approximate": "The action is selected by a linear function approximator with tile coding. "
                        "The Q-value is computed as the dot product of the weight vector and "
                        "the binary feature vector from overlapping tilings.",
    }
    return explanations.get(algorithm_key, "")
