# Design Decisions

## Theme: Research Facility Escape

The rooms form a narrative of escaping a research facility. This theme provides a consistent visual identity and makes the reward structure intuitive (reaching exits, avoiding traps, collecting keys, and avoiding obstacles).

## Source Requirements vs Implementation Choices

| # | Source Requirement | Implementation Decision | Rationale |
|---|-------------------|------------------------|-----------|
| 1 | Four+ rooms with different tasks | Four required rooms plus optional Room 5: Dynamic Obstacles | Covers the assignment and adds bonus DQN work |
| 2 | Each room has a final state | Each room has exactly one exit cell/region | Clear terminal condition |
| 3 | Faster completion = higher reward | Negative step penalty as primary mechanism; optional time bonus | Simplicity; bonus defaults to 0.0 |
| 4 | Rooms 1-3 use 10×10 grids | All three use the same `GridEnvironment` base | Code reuse, consistent testing |
| 5 | Room 1 uses DP with slippery cells | Value Iteration on known stochastic grid | Standard DP approach for known MDPs |
| 6 | Room 2 uses SARSA with slippery cells | SARSA with epsilon-greedy on stochastic grid | On-policy learns risk-aware behaviour |
| 7 | Room 3 uses Q-Learning | Q-Learning with key-collection state extension | Off-policy suits deterministic key mechanic |
| 8 | Room 4 non-grid, uses X,Y,Vx,Vy | Continuous state with 9 discrete velocity actions | Represents all velocity pairs in {-1,0,1}^2 |
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

### Grid Restored on Reset
`GridEnvironment.reset()` now restores `self._grid` from `self._original_grid`, ensuring rooms that mutate the grid (e.g. Room 3's key collection) are fully reset.

## Phase 3 Decisions (Value Iteration)

### Terminal Value Convention
Terminal states (exit cells) have `V = 0` and `policy = None`.
The Bellman update for terminal outcomes uses only the reward (no bootstrap):
```
contribution = probability × reward   (no γV(s') term)
```

### Synchronous Bellman Sweeps
All state values are updated from the previous iteration's values, then assigned atomically. This matches standard textbook value iteration.

### Tie-breaking
When multiple actions have equal Q-values (within `tie_tolerance`), the first action in enum order is selected: UP(0) → RIGHT(1) → DOWN(2) → LEFT(3).

### Pure Value Iteration
`solve()` never calls `step()`, never mutates the environment, and never samples the RNG. All transitions are computed via `get_transition_distribution()`.

### Gamma < 1.0
`ValueIterationConfig` validates `gamma` must be in `[0, 1)`. This ensures convergence guarantees.

### Immutable Results
`ValueIterationResult.values` and `.policy` use `types.MappingProxyType` (read-only views).
`RolloutResult` and `ValueIterationResult` are frozen dataclasses.

### Dedicated Result Types
- `RolloutResult` — single policy rollout trajectory with step-level data.
- `PolicyEvaluationSummary` — aggregated metrics over multiple seeded episodes.

### General State/Action APIs
`states`, `actions`, and `is_terminal_state()` are defined on `GridEnvironment` (the base class), not only on `KnownModelGridEnvironment`.

### Separate Cache Keys in Streamlit
Solve, rollout, and evaluation results are cached under independent keys in `st.session_state`, so changing rollout parameters does not invalidate solve results.

### Fresh Environment Instances for Evaluation
Each evaluation episode gets a `copy.deepcopy()` of the base environment to prevent interference between seeded runs.

### Experiment Ranking
Hyperparameter experiments are ranked by: converged > success_rate > -mean_successful_steps > mean_return > iterations.

### Transition Distribution Collision Parity
`get_transition_distribution()` now mirrors `step()` for wall and boundary
collisions: both return `step_penalty + wall_penalty` and leave the state
unchanged. `tests/test_dynamic_programming.py` includes explicit model-vs-step
parity checks for both boundary and wall collisions.

## Phase 4 Decisions (SARSA)

### Truncation Convention
When an episode is truncated (timeout), the target for the last transition is the reward only — no bootstrap. This follows the standard truncated-episode convention in Sutton & Barto and avoids overestimating Q-values with stale next-state values.

### Seeded Random Tie-Breaking
- **Evaluation (display)**: `extract_greedy_policy()` uses deterministic enum-order tie-breaking (UP → RIGHT → DOWN → LEFT) for stable, inspectable policy visualization.
- **Rollout/action selection**: `select_action()` uses seeded uniform random tie-breaking among tied maxima to break symmetry during both training and evaluation rollouts. This documented asymmetry is intentional: deterministic display, stochastic rollouts.

### RNG Stream Management
`SeedSequence(config.seed).spawn(4)` creates four independent RNG streams:
1. **Env stream** — environment randomness (slippery transitions).
2. **Policy stream** — epsilon-greedy exploration and tie-breaking.
3. **Snapshot stream** — snapshot evaluation rollouts.
4. **Bookkeeping stream** — unused (reserved for future metrics).

No `np.random` global functions are used anywhere.

### Snapshot Design
- Snapshots are captured after the specified episode **completes** (post-train iteration).
- One-based numbering: snapshot episode `1` is taken after episode 1 finishes.
- Default milestones: `[1, total//4, total//2, 3*total//4, total]`.
- Each snapshot copies the Q-table immutably (nested `MappingProxyType` for dict, tuple-wrapped arrays) and runs an evaluation-style rollout (epsilon=0, same Q-table, separate RNG).
- Snapshots do not affect training Q-values; a separate policy RNG is used.

### Model Persistence Format
- **JSON metadata** (`<stem>.json`): versioned (current v=1), map signature (SHA-256 hex[:16]), grid shape, state/action counts, config, reward/slip config, training summary.
- **NPZ arrays** (`<stem>.npz`): `states` (N×2 int grid positions), `values` (N×4 float Q-values).
- **Load validation**: map signature, state count, action count, and finite value checks produce clear `ValueError` on mismatch.

### Experiment Ranking
Hyperparameter experiments are ranked by: converged > success_rate > -mean_successful_steps > mean_return > final_epsilon. Same ranking criteria as Phase 3 but with final_epsilon as the final tiebreaker.

### SARSA Update Formula
- Terminal state (exit reached): `target = reward`
- Truncated (timeout): `target = reward` (no bootstrap)
- Non-terminal: `target = reward + γ × Q(s', a')` where `a'` is the action selected by the current epsilon-greedy policy at `s'` (on-policy).

## Phase 5 Decisions (Q-Learning)

### Augmented State Space
Room 3 uses `(row, column, has_key)` — the Cartesian product of all non-wall physical positions × `{False, True}`. This includes some unreachable combinations (e.g. `(key_position, True)` after the key is collected) but is simple, predictable, and easy to debug. Documented as a deliberate tradeoff.

### Key-Before-Exit Rule
- The locked exit without the key is **not** terminal; the agent receives a penalty and the episode continues.
- The locked exit with the key is terminal (success).
- The key reward is issued exactly once per episode; the grid cell is cleared to prevent double rewards.

### Off-Policy Update (Q-Learning vs SARSA)
The critical distinction is in the target computation:

- **SARSA**: `target = reward + γ × Q(s', a')` — uses the **next behaviour action** `a'`.
- **Q-Learning**: `target = reward + γ × maxₐ' Q(s', a')` — uses the **greedy maximum** over next-state actions, independent of the behaviour policy.

Tests explicitly prove this difference by constructing scenarios where the behaviour action at `s'` differs from the max, and verifying Q-Learning converges to the max-based target.

### Truncation Convention (Same as Phase 4)
Truncated episodes do not bootstrap: `target = reward` only. This avoids overestimating Q-values with stale next-state values on timeout.

### Rollout and Evaluation
- `rollout_q_learning_policy()` is generic — it works with any `GridEnvironment` factory and any hashable state type. This enables the common-benchmark comparison with Room 2's 2-element `Position` states.
- `evaluate_q_learning_policy()` uses epsilon=0 by default and supports `retain_rollouts` / `max_retained_rollouts` for experiment efficiency.
- Key and locked-exit events are tracked via `TrajectoryStep.event` fields during evaluation.

### Q-Learning Persistence
Separate format from SARSA:
- `"schema_version"`, `"algorithm": "Q-Learning"`, `"room": "Room3QLearning"`, `"state_schema": ["row", "column", "has_key"]`.
- States stored as `(row, column, has_key)` in `.npz` with 3 columns.
- Maximum validation: schema version, algorithm name, room name, map signature, state schema, state count, action count, finite values.

### Controlled Comparison Methodology
- **Benchmark**: Room 2 Laser Corridor map for both algorithms (same traps, slippery cells, rewards, slip config, max steps).
- **Comparison A (matched)**: Identical alpha, gamma, epsilon decay, episode count, training seeds (0–4), evaluation seeds (0–99). Reports mean paired difference.
- **Comparison B (tuned)**: Each algorithm's best confirmed configuration on the same benchmark. Reports mean and std across seeds.
- No claim of universal superiority; comparison is descriptive.

### Shared Tabular Utilities
Generic helpers extracted from `agents/sarsa.py` into `agents/tabular_utils.py`:
- `epsilon_for_episode`, `select_epsilon_greedy_action`, `extract_deterministic_greedy_policy`
- `freeze_q_table`, `validate_q_table`, `default_snapshot_episodes`, `map_signature`

Backward-compatible aliases remain in `agents/sarsa.py`. All SARSA tests remain green.

## Room 4 — Movement and Action Semantics

1. **Nine velocity actions**: STOP (0,0), 4 cardinal (N/E/S/W), 4 diagonal (NE/SE/SW/NW). Each action directly sets the velocity vector. All allowed `(Vx,Vy) ∈ {-1,0,1}²` are represented.
2. **Euler integration**: `pos_next = pos + vel * dt` with `dt = 0.02 s`.
3. **Diagonal magnitude**: Diagonal velocity (1,1) has magnitude √2 ≈ 1.414; time to traverse the room diagonally is approximately 450 steps.
4. **Boundary handling**: Position is clipped to `[0,10]²`. If a candidate position is outside, the blocked velocity component is zeroed and a collision penalty is applied.
5. **Exit**: Circle with radius 0.35 m centred at (9.5, 9.5). Distance threshold avoids floating-point exact-match issues.

## Room 4 — Tile Coding

1. **Deterministic indexing**: Feature index uses a closed-form formula with tiling × position × velocity components. No hash map or randomness.
2. **Velocity encoding**: Vx ∈ {-1,0,1} → {0,1,2}; Vy ∈ {-1,0,1} → {0,1,2}. Always included in the feature index (required by brief).
3. **Tiling offset**: Each tiling shifts its X/Y bin boundaries by `t * 7` modulo `tiles_x × tiles_y` to produce distinct partitionings.
4. **Feature count**: `num_tilings × tiles_x × tiles_y × 3 × 3`. Default: 8×10×10×9 = 7,200.

## Room 4 — Reward Design

1. **Distance-progress shaping**: `scale × (d_before − d_after)` is an implementation aid, not an assignment requirement. It can be zeroed out.
2. **Step penalty**: -0.01 per step (much smaller than grid rooms because episodes are hundreds of steps long).
3. **Timeout penalty**: -25.0 to discourage aimless wandering.
4. **Reward components exposed**: All reward terms are reported individually in `info` dict (`step_penalty`, `boundary_penalty`, `exit_reward`, `timeout_penalty`, `progress_reward`).

## Room 4 — Semi-Gradient SARSA

1. **Truncation convention**: Timeout episodes do **not** bootstrap, matching Phases 4/5.
2. **Step-size normalization**: `alpha / num_tilings` so the effective update magnitude is independent of the number of tilings.
3. **True on-policy**: The next action `a′` is selected from the current epsilon-greedy policy before computing the target.
4. **Weights read-only**: Public weight copies are `np.ndarray` with `writeable=False`. Persisted weights are validated on load.

## Room 4 — Generalization Evaluation

1. **Fixed start**: Always (0.5, 0.5, 0, 0).
2. **Unseen starts** (optional): Fixed grid of positions outside the lower-left training region.
3. **Random starts**: Configurable via `StartMode` with explicit seed sets for reproducibility.
4. **Start validation**: Custom and random starts are validated to be outside the exit radius.

## Room 5 — Optional Dynamic Obstacles

1. **Bonus scope**: Room 5 is additive and is not counted as mandatory basic
   compliance. Existing SARSA-vs-Q-Learning comparison claims remain based on
   the shared Room 2 benchmark.
2. **Geometry**: Obstacles are axis-aligned squares with exact width 0.5m.
   Width validation rejects any other value so the assignment-facing contract
   cannot drift.
3. **Layouts**: Random layouts are generated from layout seeds with configurable
   obstacle-count ranges. A fixed validation layout is available for stable
   screenshots and comparable evaluation.
4. **Visibility**: Obstacle observations use center-to-center distance. The
   DQN sees only the nearest K=4 visible obstacle records and padded slots for
   missing obstacles.
5. **Reward shape**: Distance-progress shaping helps learning but remains
   explicit in the reward config and step `info`, alongside obstacle, boundary,
   exit, and timeout components.
6. **DQN implementation**: A small NumPy network was used instead of PyTorch to
   keep dependencies lightweight. It includes replay, target-network copies,
   epsilon-greedy behavior, mini-batch TD updates, snapshots, and seeded
   reproducibility.
7. **Persistence**: Room 5 models are JSON metadata plus `.npz` weights with
   observation schema, environment config, seeds, finite-value validation, and
   a SHA-256 checksum.
