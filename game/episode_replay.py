"""Episode playback — builds replay data from rollout results."""

from core.types import Action, RolloutResult
from game.html_rendering import normalize_html
from game.models import ReplayStep, ReplayState


def build_replay_from_rollout(
    rollout: RolloutResult,
    room_id: str,
    stage_label: str = "Final",
    snapshot_epsilon: float | None = None,
) -> ReplayState:
    # Convert algorithm rollout objects into a stable UI playback model.  The
    # replay object is small enough to keep in Streamlit session state.
    steps: list[ReplayStep] = []
    cum_reward = 0.0
    for i, ts in enumerate(rollout.steps):
        cum_reward += ts.reward
        steps.append(ReplayStep(
            step_index=i,
            state=ts.state,
            action=ts.requested_action,
            effective_action=ts.effective_action,
            reward=ts.reward,
            next_state=ts.next_state,
            slipped=ts.slipped,
            collision=ts.collision,
            event=ts.event,
            terminated=ts.terminated,
            truncated=ts.truncated,
            cumulative_reward=cum_reward,
            epsilon_at_time=snapshot_epsilon,
        ))
    return ReplayState(
        room_id=room_id,
        steps=tuple(steps),
        current_index=0,
        playing=False,
        speed=1.0,
        total_steps=rollout.total_steps,
        total_reward=rollout.total_reward,
        success=rollout.success,
        stage_label=stage_label,
    )


def render_replay_bar(
    replay: ReplayState,
    replay_key: str = "replay",
) -> str:
    # This returns HTML only; button controls live in game_view_common.py.
    total = len(replay.steps)
    cur = replay.current_index
    pct = int(100 * cur / max(1, total - 1)) if total > 1 else 0

    return normalize_html(
        f'<div class="replay-bar">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span class="replay-stage">{replay.stage_label}</span>'
        f'<span style="color:#90a4ae;font-size:0.85em;">Step {cur + 1} / {total}</span>'
        f'<div style="flex:1;min-width:80px;height:4px;background:#333;border-radius:2px;margin:0 8px;">'
        f'<div style="width:{pct}%;height:100%;background:#4fc3f7;border-radius:2px;transition:width 0.15s;"></div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def get_current_step(replay: ReplayState) -> ReplayStep | None:
    # Defensive helper so views can ask for the current step even before a
    # replay exists or after an index reset.
    if not replay.steps or replay.current_index >= len(replay.steps):
        return None
    return replay.steps[replay.current_index]
