# Room Specifications

## Room 1 — Ice Maze (Dynamic Programming)

| Property | Value |
|----------|-------|
| Algorithm | Value Iteration (Dynamic Programming) |
| Grid | 10×10, known environment |
| State | `(row, column)` — 100 discrete states |
| Actions | `UP(0), RIGHT(1), DOWN(2), LEFT(3)` |
| Slippery cells | Yes — stochastic transitions on ice tiles |
| Terminal condition | Reaching the exit cell |
| Design | Short risky path near ice; long safe path available |

### Grid Legend
- `.` Empty — walkable
- `#` Wall — impassable
- `S` Start — agent starting position
- `E` Exit — terminal, ends episode
- `I` Slippery — stochastic transitions
- `T` Trap — heavy penalty (not used in Room 1)

### Default Grid
```
##########
S....I...#
#.###.##.#
#.#I.....#
#.#.####.#
#....I...#
#####.##.#
#...I....#
#......E.#
##########
```

### Rewards
| Event | Formula | Default |
|-------|---------|--------:|
| Normal step | `step_penalty` | -1.0 |
| Reach exit | `step_penalty + exit_reward` | +99.0 |
| Hit wall/boundary | `step_penalty + wall_penalty` | -4.0 |

### Value Iteration Default Configuration

| Parameter | Default | Values |
|-----------|---------|--------|
| Gamma | 0.95 | 0.80, 0.90, 0.95, 0.99 |
| Tolerance | 1e-6 | 1e-2, 1e-4, 1e-6 |
| Max iterations | 10,000 | 100–50,000 |
| Tie tolerance | 1e-12 | fixed |
| Slip (intended/left/right) | 0.80/0.10/0.10 | det(1/0/0), default, high(0.6/0.2/0.2) |

### Experiment Results (Room 1 — Ice Maze)

Best configuration is determined by: converged > success_rate > -mean_successful_steps > mean_return > iterations.
Results are saved to `storage/experiments/room1_dp/` as JSON.

---

## Room 2 — Laser Corridor (SARSA)

| Property | Value |
|----------|-------|
| Algorithm | SARSA (on-policy TD control) |
| Grid | 10×10, unknown environment |
| State | `(row, column)` — 100 discrete states |
| Actions | `UP(0), RIGHT(1), DOWN(2), LEFT(3)` |
| Slippery cells | Yes |
| Traps | Yes — agent learns risk-averse behaviour via on-policy learning |
| Terminal condition | Reaching the exit cell |

### Grid Legend
Same as Room 1, with trap cells (`T`) placed near risky shortcuts.

### Default Grid (Final)
```
##########
#SI......#
#.##.###.#
#.#T..I#.#
#....#...#
####.#.#.#
#I.....#.#
#.####.#.#
#...I....E
##########
```

### Room 2 Map Analysis
| Cell type | Count | Locations |
|-----------|-------|-----------|
| Slippery (I) | 3 | (1,2), (3,6), (6,1), (9,4), plus note below |
| Trap (T) | 1 | (3,3) |
| Empty | 22 | Various |
| Wall | 73 | Borders + interior |

Note: The grid shows 4 I cells: (1,2), (3,6), (6,1), (8,4). Some may be duplicates in the counting.

### Rewards
| Event | Formula | Default |
|-------|---------|--------:|
| Normal step | `step_penalty` | -1.0 |
| Reach exit | `step_penalty + exit_reward` | +99.0 |
| Hit wall/boundary | `step_penalty + wall_penalty` | -4.0 |
| Enter trap | `step_penalty + trap_penalty` | -21.0 |

### SARSA Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Alpha | 0.10 | Learning rate (0.05, 0.10, 0.30, 0.50) |
| Gamma | 0.95 | Discount factor (0.90, 0.95, 0.99) |
| Max steps | 200 | Episode truncation limit |
| Episodes | 2,000 / 5,000 | Screening / confirmation |
| Seed | 42 | Random seed |
| Epsilon start | 1.0 | Initial exploration rate |
| Epsilon minimum | 0.01 | Floor on exploration |
| Epsilon decay | 0.995 | Multiplicative decay (also 0.990, 0.999) |
| Decay kind | EXPONENTIAL | CONSTANT, EXPONENTIAL, or LINEAR |
| Linear decay episodes | Episodes // 2 | (used only for LINEAR kind) |

### Experiment Design
- **Screening**: 1 seed, 2,000 episodes, all 36 configs (4α × 3γ × 3 decay).
- **Confirmation**: 3 seeds, 5,000 episodes, top 3 configs + baseline.
- **Ranking**: converged > success_rate > -mean_successful_steps > mean_return > final_epsilon.
- Results saved to `storage/experiments/room2_sarsa/`.

### Model Persistence
Saved as two files:
- `<stem>.json`: metadata (version, map signature, config, rewards, slip config, training summary).
- `<stem>.npz`: Q-table arrays (`states` N×2, `values` N×4).

Load validates map signature, state count, action count, and finite values.

---

## Room 3 — Key Vault (Q-Learning)

| Property | Value |
|----------|-------|
| Algorithm | Q-Learning (off-policy TD control) |
| Grid | 10×10, unknown environment |
| State | `(row, column, has_key)` - 92 discrete states |
| Actions | `UP(0), RIGHT(1), DOWN(2), LEFT(3)` |
| Special mechanic | Key must be collected before exit becomes available |
| Terminal condition | Reaching the exit cell while `has_key == True` |
| Non-terminal | Reaching the exit cell without the key (episode continues) |

Room 3 has 46 non-wall physical positions, so `(row, column, has_key)` yields
46 x 2 = 92 tabular states.

### Grid Legend
- Same as Room 1, plus:
- `K` Key — collectable; sets `has_key = True`
- `L` Locked Exit — acts as exit only when `has_key == True`

### Default Grid
```
##########
#S..#...K#
#.#.#.##.#
#.#....#.#
#...##.#.#
###.#I...#
#......#.#
#.####.#.#
#...I....L
##########
```

### State-Space Design

Correct count: 46 non-wall physical positions x 2 key flags = 92 states.
Room 3 uses an augmented state `(row, column, has_key)` - the Cartesian product of all 46 non-wall physical positions x `{False, True}`. This yields exactly 92 states. Some combinations may be unreachable (e.g. `(key_position, True)` after the key is collected), but this is an acceptable tradeoff for simplicity in a tabular project.

### Q-Learning Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Alpha | 0.10 | Learning rate (0.05, 0.10, 0.30, 0.50) |
| Gamma | 0.95 | Discount factor (0.90, 0.95, 0.99) |
| Max steps | 500 | Episode truncation limit |
| Episodes | 1,000 / 5,000 | Screening / confirmation |
| Seed | 42 | Random seed |
| Epsilon start | 1.0 | Initial exploration rate |
| Epsilon minimum | 0.05 | Floor on exploration |
| Epsilon decay | 0.995 | Multiplicative decay (also 0.990, 0.999) |
| Decay kind | EXPONENTIAL | CONSTANT, EXPONENTIAL, or LINEAR |
| Linear decay episodes | 4,000 | (used only for LINEAR kind) |

### Experiment Design
- **Screening**: 1 seed (42), 1,000 episodes, 36 configs (4α × 3γ × 3 decay), 50 eval episodes.
- **Confirmation**: Top 5 configs × 3 seeds × 5,000 episodes × 100 eval episodes.
- **Ranking**: success_rate > key_collection_rate > -success_rate_std > -mean_successful_steps.
- Results saved to `storage/experiments/room3_q_learning/`.

### Model Persistence
- JSON metadata: `schema_version`, `algorithm`, `room`, `state_schema`, `map_signature`, config, rewards, slip config, training summary.
- NPZ arrays: `states` (N×3: row, column, has_key), `values` (N×4).
- Load validates: schema version, algorithm name, room name, map signature, state schema, state count, action count, finite values.

### SARSA Comparison

A controlled comparison of SARSA and Q-Learning is run on the **Room 2 Laser Corridor** benchmark so both algorithms encounter the same:
- Slippery transitions, trap penalties, risky shortcut, safer route, rewards, and maximum episode length.
- **Comparison A**: Identical alpha=0.10, gamma=0.95, epsilon decay=0.995, 2,000 episodes, 5 paired training seeds (0–4), 100 evaluation seeds (0–99).
- **Comparison B**: Each algorithm's best-tuned configuration on the same benchmark.

Results stored in `storage/comparisons/`.

---

## Room 4 — Momentum Chamber (Function Approximation)

| Property | Value |
|---|---|---|
| Algorithm | Semi-gradient SARSA with tile coding (Phase 6) |
| Environment | Continuous 10×10 metre room |
| State | `(X, Y, Vx, Vy)` — continuous position, integer velocity |
| Velocity domain | Vx, Vy ∈ {-1, 0, 1} |
| Actions | 9 velocity vectors: STOP, N, NE, E, SE, S, SW, W, NW |
| Action mapping | `VELOCITY_BY_ACTION` in `core/types.py` |
| Time step | 0.02 seconds |
| Default max steps | 750 (theoretical diagonal minimum ≈ 450) |
| Exit centre | `(9.5, 9.5)` |
| Exit region | Circle at centre `(9.5, 9.5)` with radius `0.35` m |
| Boundary rule | Clip to `[0, 10]`, zero blocked velocity component, apply collision penalty |

### Configuration

#### Motion (Room4MotionConfig)

| Parameter | Default | Description |
|---|---|---|
| `room_width_m` | 10.0 | Room width in metres |
| `room_height_m` | 10.0 | Room height in metres |
| `time_step_s` | 0.02 | Simulation time step |
| `start_position` | (0.5, 0.5) | Fixed-start position |
| `start_velocity` | (0, 0) | Fixed-start velocity |
| `exit_center` | (9.5, 9.5) | Exit circle centre |
| `exit_radius_m` | 0.35 | Exit circle radius |

#### Rewards (ContinuousRewardConfig)

| Parameter | Default | Description |
|---|---|---|
| `step` | -0.01 | Per-step penalty |
| `exit` | 100.0 | Exit reward |
| `boundary_collision` | -1.0 | Boundary collision penalty |
| `timeout` | -25.0 | Timeout penalty |
| `distance_progress_scale` | 1.0 | Scale for distance-to-exit progress shaping |

Rewards are additive. All components exposed in `info` dict (`step_penalty`, `boundary_penalty`, `exit_reward`, `timeout_penalty`, `progress_reward`).

#### Tile Coding (TileCodingConfig)

| Parameter | Default | Description |
|---|---|---|
| `num_tilings` | 8 | Number of overlapping tilings |
| `tiles_x` | 10 | Number of X bins per tiling |
| `tiles_y` | 10 | Number of Y bins per tiling |
| `include_velocity` | True | Whether velocity categories are included |

Feature count: `num_tilings × tiles_x × tiles_y × 3 × 3`. Default: 7,200. Velocity is always included (brief requirement).

#### Training (ApproximateSarsaConfig)

| Parameter | Default | Description |
|---|---|---|
| `episodes` | 3,000 | Number of training episodes |
| `alpha` | 0.10 | Learning rate (normalized by num_tilings) |
| `gamma` | 0.99 | Discount factor |
| `max_steps` | 750 | Max steps per episode |
| `seed` | 42 | RNG seed |
| `epsilon` | EXP(1.0→0.02, decay=0.997) | Exploration schedule |
| `start_mode` | RANDOM_LOWER_LEFT | Training start distribution |

### Update Equation

```
δ = r + γ·Q(s′,a′) − Q(s,a)       # non-terminal, non-truncated
δ = r − Q(s,a)                      # terminal or truncated
w[a, idx] += (α / num_tilings) × δ   # per active feature idx
```

### Start Modes

| Mode | Position | Velocity |
|---|---|---|
| `FIXED` | (0.5, 0.5) | (0, 0) |
| `RANDOM_LOWER_LEFT` | X,Y ∈ [0.25, 3.0] | (0, 0) |
| `RANDOM_ROOM` | Any position outside exit radius | (0, 0) |

All starts validated outside exit radius.

### Persistence Format

File pair: `weights.npz` + `metadata.json`. Validation: schema version, algorithm name, action schema, weight shape, tile coding config, finite values. Uses `os.replace` (atomic rename).

### Experiment Design

#### Stage A — One Factor at a Time

| Factor | Values |
|---|---|
| `num_tilings` | 4, 8, 16 |
| `tiles_xy` | 8, 10, 16 |
| `alpha` | 0.05, 0.10, 0.20 |
| `progress_scale` | 0.0, 0.5, 1.0 |
| `epsilon_decay` | 0.990, 0.997, 0.999 |

#### Stage B — Best 2 × Best 2 Combinations

#### Confirmation

Top 5 configs × 3 seeds × 3000 episodes × 100 eval starts.

### Test Coverage (59 tests)

| Group | Tests | Covers |
|---|---|---|
| Environment | 19 | State, actions, movement, boundary, exit, timeout, start modes, validation |
| Tile coder | 10 | Feature count, active count, determinism, overlap, velocity |
| Linear approximator | 5 | Init, action values, isolated update, finite, read-only |
| Semi-gradient SARSA | 8 | Exact update, terminal, truncated, next-action, reproducibility |
| Evaluation | 5 | Fixed, no-mutation, reproducibility, retention, categories |
| Persistence | 5 | Round-trip, metadata, mismatch, shape, finite |
| Visualization | 4 | Action field, value surface, dataframe, no-mutation |
| Learning sanity | 2 | Trained > random, seeded non-flaky |

---

## Room 5 — Dynamic Obstacles (Optional, Not in Phase 1)

| Property | Value |
|----------|-------|
| Algorithm | NumPy DQN with replay buffer and target network |
| Environment | Continuous 10×10 metre with moving obstacles |
| Observation | 22-feature vector with nearest K=4 visible obstacle records |
| Obstacles | Axis-aligned seeded squares, exact width 0.5m |
| Evaluation | Fixed, random, and unseen random layouts |

Room 5 is implemented as optional bonus work beyond the required four-room
assignment path.

### Environment Configuration

| Parameter | Default | Description |
|---|---:|---|
| `room_width_m` | 10.0 | Room width |
| `room_height_m` | 10.0 | Room height |
| `time_step_s` | 0.05 | Room 5 default step size |
| `obstacle_width_m` | 0.5 | Required exact obstacle width |
| `min_obstacles` | 3 | Minimum generated obstacle count |
| `max_obstacles` | 5 | Maximum generated obstacle count |
| `observation_distance_m` | 2.5 | Visibility radius X for obstacle records |
| `nearest_obstacles` | 4 | Padded nearest visible obstacle slots |
| `max_steps` | 260 | Episode truncation limit |

Obstacle visibility and reporting use center-to-center distance from the agent
to each obstacle center. Layout generation avoids the start, the exit, and
overlapping obstacle squares.

### Observation Schema

The DQN observation is:

```
x_norm, y_norm, vx, vy, exit_dx_norm, exit_dy_norm,
obstacle_0_visible, obstacle_0_dx_over_x, obstacle_0_dy_over_x, obstacle_0_distance_over_x,
obstacle_1_visible, obstacle_1_dx_over_x, obstacle_1_dy_over_x, obstacle_1_distance_over_x,
obstacle_2_visible, obstacle_2_dx_over_x, obstacle_2_dy_over_x, obstacle_2_distance_over_x,
obstacle_3_visible, obstacle_3_dx_over_x, obstacle_3_dy_over_x, obstacle_3_distance_over_x
```

Missing obstacle slots are padded as `(0, 0, 0, 1)`.

### Rewards

| Event | Default |
|---|---:|
| Step | -0.01 |
| Exit | +120.0 |
| Boundary collision | -1.0 |
| Obstacle collision | -60.0 |
| Timeout | -25.0 |
| Distance-progress shaping | 2.0 x distance gain |

Obstacle collision terminates as failure. Reaching the exit terminates as
success. Timeout truncates and does not bootstrap.

### DQN Training

| Parameter | Default |
|---|---:|
| Episodes | 600 |
| Learning rate | 0.001 |
| Gamma | 0.99 |
| Replay capacity | 20,000 |
| Batch size | 64 |
| Warmup steps | 128 |
| Target update interval | 100 |
| Hidden units | 64 |
| Epsilon schedule | EXP(1.0 -> 0.05, decay=0.995) |

The implementation is intentionally NumPy-only: one hidden ReLU layer, replay
buffer, epsilon-greedy behavior policy, mini-batch TD target, target network
copy, deterministic seeded RNG streams, snapshots, evaluation, save/load, and
finite-value validation.

### Persistence

Room 5 models use a JSON metadata file plus `.npz` weights. Metadata records
algorithm tag, environment config, observation/action schema, training config,
training seed, dimensions, and a SHA-256 checksum over the weights file.
