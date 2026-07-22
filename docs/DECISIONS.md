# Design Decisions

## Theme: Research Facility Escape

The four rooms form a narrative of escaping a research facility. This theme provides a consistent visual identity and makes the reward structure intuitive (reaching exits, avoiding traps, collecting keys).

## Source Requirements vs Implementation Choices

| # | Source Requirement | Implementation Decision | Rationale |
|---|-------------------|------------------------|-----------|
| 1 | Four+ rooms with different tasks | Four rooms: Ice Maze, Laser Corridor, Key Vault, Momentum Chamber | Covers all specified algorithms |
| 2 | Each room has a final state | Each room has exactly one exit cell/region | Clear terminal condition |
| 3 | Faster completion = higher reward | Negative step penalty as primary mechanism; optional time bonus | Simplicity; bonus defaults to 0.0 |
| 4 | Rooms 1-3 use 10×10 grids | All three use the same `GridEnvironment` base | Code reuse, consistent testing |
| 5 | Room 1 uses DP with slippery cells | Value Iteration on known stochastic grid | Standard DP approach for known MDPs |
| 6 | Room 2 uses SARSA with slippery cells | SARSA with epsilon-greedy on stochastic grid | On-policy learns risk-aware behaviour |
| 7 | Room 3 uses Q-Learning | Q-Learning with key-collection state extension | Off-policy suits deterministic key mechanic |
| 8 | Room 4 non-grid, uses X,Y,Vx,Vy | Continuous state with 5 discrete velocity actions | Matches brief specification |
| 9 | Room 4 is 10×10 metres | Continuous position clipped to [0,10] | Direct from requirement |
| 10 | Room 4 dt = 0.02s | dt parameter on `Room4Continuous` | Direct from requirement |
| 11 | Vx, Vy are -1, 0, 1 | Five actions map directly to velocity vectors | Simple, matches brief |
| 12 | Training parameters controllable | `RLConfig` dataclass with all hyperparameters | Centralized, no scattered constants |
| 13 | Learning-progress graphs | Streamlit plots via `TrainingPlots` (Phase 2+) | Visual feedback during training |
| 14 | Episode replay | `EpisodeReplay` class (Phase 2+) | Debugging and presentation |
| 15 | Python | Pure Python with numpy, matplotlib, streamlit | Direct from requirement |
| 16 | Streamlit preferred | Single `app.py` with multipage-style navigation | Simpler than multi-page apps |
| 17 | GitHub submission | Git repo with `.gitignore` | Standard practice |

## Architecture Decisions

### Environment-Agent Separation
- Environment classes handle state, transitions, rewards, terminal conditions, rendering.
- Agent classes handle policies, value functions, exploration, updates.
- Algorithms never contain room-map logic.

### Common Environment API
`reset(seed=None)`, `step(action)`, `render()` on all environments.
Room 1 adds `get_transition_distribution()` via `KnownModelGridEnvironment`.

### Random Seed Handling
`numpy.random.Generator` via `np.random.default_rng(seed)`.
No global random state. Every environment and trainer accepts a seed.

### Configuration
Dataclasses for reward settings, training settings, room metadata.
No constants scattered across files.

### Reward Configurator
Single `RewardConfig` dataclass with all reward values.
Environments accept it as a constructor parameter.
Rewards are additive: `total = step_penalty + event_penalty`.

### State Representation
Rooms 1-2: `(row, col)` tuples via `_encode_state()` hook.
Room 3: `(row, col, has_key)` tuple via `_encode_state()` override.
Room 4: `(X, Y, Vx, Vy)` numpy array.

### Velocity Interpretation for Room 4
Actions directly select the next velocity vector rather than applying forces.
This avoids continuous control complexity in early phases.

### Exit Detection (Room 4)
Circular region instead of exact coordinate match.
Default centre `(9.5, 9.5)`, radius `0.3m`.

## Phase 2 Decisions

### Terminated vs Truncated
- `terminated` is only `True` on successful exit (or locked-exit with key in Room 3).
- `truncated` is only `True` when `step_count >= max_steps` without success.
- `step()` raises `RuntimeError` after either outcome; `reset()` is required.

### Slippery-cell Semantics
- Stochastic transitions apply only when the agent is currently standing on a `SLIPPERY` cell.
- Default probabilities: intended=0.80, left=0.10, right=0.10.
- Configurable via `SlipConfig` dataclass (validated to sum to 1.0).

### Additive Reward Composition
Every action starts with `step_penalty`. Event penalties are added:
- Ordinary step: `-1`
- Wall collision: `-1 + -3 = -4`
- Trap: `-1 + -20 = -21`
- Exit: `-1 + 100 = 99`

### Trap Non-Terminal Default
Traps are traversable and apply an additive penalty but do not terminate the episode.

### Known-Model Interface Separation
- `GridEnvironment` is the base for unknown-model rooms.
- `KnownModelGridEnvironment` adds `get_transition_distribution()` for Room 1.
- The method is pure (no state mutation) and excludes timeout from the DP model.

### Goal-count Validation
Each room validates its own terminal cell type via `_terminal_cell_types()`:
- Rooms 1-2 check for exactly one `EXIT`.
- Room 3 checks for `EXIT` and `LOCKED_EXIT`.

### Key Reward Issued Once
Room 3 sets `_key_collected = True` on first key pickup and clears the grid cell, preventing double rewards.

## Room 4 Movement Assumptions

1. Velocity is set instantaneously by the action (no acceleration).
2. Position update is Euler integration: `pos += vel * dt`.
3. Boundary collision zeroes only the blocked velocity component.
4. Collision penalty is applied per-boundary-hit.
5. Exit is a circle to avoid floating-point exact-match issues.
