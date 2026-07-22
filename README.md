# Reinforcement Learning Escape Room

A Python-based reinforcement learning escape-room application with four rooms, each using a different RL method.

## Rooms

| Room | Name | Algorithm | Environment |
|------|------|-----------|-------------|
| 1 | Ice Maze | Value Iteration (DP) | Known 10×10 stochastic grid |
| 2 | Laser Corridor | SARSA | Unknown 10×10 stochastic grid |
| 3 | Key Vault | Q-Learning | Unknown 10×10 grid with key mechanic |
| 4 | Momentum Chamber | Semi-gradient SARSA + tile coding | Continuous 10×10 metre room |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Manual Grid Mode

Use the Streamlit app to manually explore Rooms 1–3:

1. Select a room from the dropdown.
2. Enter a seed and click **Reset**.
3. Click **UP / RIGHT / DOWN / LEFT** to move the agent.
4. Observe step count, rewards, slip events, collisions, and terminal states.

No training or RL algorithms are included in this mode.

## Cell Symbols

| Symbol | Cell Type |
|--------|-----------|
| `.` | Empty (walkable) |
| `#` | Wall (impassable) |
| `S` | Start |
| `E` | Exit |
| `I` | Slippery (stochastic transitions) |
| `T` | Trap (additive penalty, traversable) |
| `K` | Key |
| `L` | Locked exit |
| `A` | Agent (render only) |

## Movement Rules

- Actions: `UP (0)`, `RIGHT (1)`, `DOWN (2)`, `LEFT (3)`.
- Candidate cell is calculated via `ACTION_DELTAS`.
- If candidate is a wall or outside grid → agent stays in place + wall penalty added.
- Otherwise → agent moves to candidate cell.

## Slippery-cell Semantics

- Stochastic transitions apply **only** when the agent is currently standing on a `SLIPPERY` cell.
- Default probabilities: intended=0.80, left=0.10, right=0.10.
- Configurable via `SlipConfig` dataclass.
- `info["slipped"]` is `True` when the effective action differs from the requested action.

## Additive Rewards

Rewards are additive. Each action starts with `step_penalty`, then event penalties are added:

| Event | Formula | Default |
|-------|---------|--------:|
| Ordinary step | `step_penalty` | -1.0 |
| Wall collision | `step_penalty + wall_penalty` | -4.0 |
| Trap | `step_penalty + trap_penalty` | -21.0 |
| Exit | `step_penalty + exit_reward` | +99.0 |
| Timeout | `step_penalty + step_limit_penalty` | -31.0 |
| Key collection | `step_penalty + key_reward` | +9.0 |
| Locked exit (no key) | `step_penalty + locked_exit_penalty` | -6.0 |

## Terminated vs Truncated

| Outcome | `terminated` | `truncated` | `info["event"]` |
|---------|:-----------:|:----------:|:--------------:|
| Exit reached | `True` | `False` | `"exit"` |
| Max steps exceeded | `False` | `True` | `"timeout"` |

Calling `step()` after either outcome raises `RuntimeError`. Call `reset()` first.

## Project Structure

```
rl_escape_room/
├── app.py                 # Streamlit manual grid demonstrator
├── config/                # Configuration dataclasses and room specs
├── core/                  # Shared types (RewardConfig, SlipConfig, StepResult, enums)
├── environments/          # Room environments with parser, hooks, KnownModel
├── agents/                # RL algorithm stubs (implemented Phase 3+)
├── training/              # Training pipeline stubs (implemented Phase 3+)
├── visualization/         # Renderer stubs (implemented Phase 3+)
├── storage/               # Saved models, metrics, episodes
├── tests/                 # Comprehensive environment tests (40 tests)
└── docs/                  # Design documents
```

## Running Tests

```bash
pytest -v
```

## Design Documents

- `docs/ROOM_SPECS.md` — Detailed specifications for each room.
- `docs/DECISIONS.md` — Design decisions and rationale.
- `PHASE_1_AGENT_BRIEF.md` — Original Phase 1 requirements.

## Team

| Member | Primary Responsibility |
|--------|----------------------|
| Member A | Shared grid, Dynamic Programming, Testing |
| Member B | SARSA, Q-Learning, Experiments |
| Member C | Room 4, Streamlit, Deployment |
| Everyone | Documentation, Defense |
