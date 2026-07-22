# Reinforcement Learning Escape Room

A Python-based reinforcement learning escape-room application with four rooms, each using a different RL method.

## Rooms

| Room | Name | Algorithm | Environment |
|------|------|-----------|-------------|
| 1 | Ice Maze | Value Iteration (DP) | Known 10×10 stochastic grid |
| 2 | Laser Corridor | SARSA | Unknown 10×10 stochastic grid |
| 3 | Key Vault | Q-Learning | Unknown 10×10 grid with key mechanic |
| 4 | Momentum Chamber | Semi-gradient SARSA + tile coding | Continuous 10×10 metre room |

## Value Iteration (Room 1)

Room 1 uses **Value Iteration** — a dynamic-programming algorithm for known MDPs.

**Bellman update (synchronous):**
```
V_new(s) = max_a Σ P(s'|s,a) × [r + (0 if terminated else γV(s'))]
```

- Terminal states (exit): `V = 0`, `policy = None` (no bootstrap).
- Iteration stops when `max_s |V_new(s) - V(s)| < tolerance`.
- If `max_iterations` reached without convergence, `converged = False`.
- Tie-breaking: UP → RIGHT → DOWN → LEFT (enum order).
- Pure: `solve()` uses only `get_transition_distribution()` — never calls `step()`, never mutates the environment, never samples RNG.

**To run from Streamlit:**
1. Select `Room 1 — DP` from the mode dropdown.
2. Adjust gamma, tolerance, max_iterations, and slip probabilities in the sidebar.
3. Click **Solve** to run Value Iteration.
4. View convergence, value grid, policy arrows, rollouts, and evaluation results in the tabs.

**Running experiments:**
```bash
python -c "from training.dp_experiments import run_room1_experiments; results = run_room1_experiments()"
```

Results are saved to `storage/experiments/room1_dp/`.

**Running tests:**
```bash
pytest tests/test_dynamic_programming.py -v
```

## SARSA (Room 2)

Room 2 uses **SARSA** — an on-policy TD control algorithm that learns Q-values from sampled experience.

**SARSA update:**
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]
```

- Non-terminal: full bootstrap with `γQ(s',a')`.
- Terminal: `target = reward` (no bootstrap).
- Truncated (timeout): `target = reward` — truncation does **not** bootstrap.
- Epsilon-greedy exploration with configurable decay (constant, exponential, linear).
- Greedy action selection uses **seeded uniform random tie-breaking** among tied maxima.
- Separate seeded RNG streams for environment, policy, snapshots, and bookkeeping.
- Snapshots capture Q-table + evaluation rollout at configurable episode milestones.
- Model persistence: versioned JSON metadata + aligned `.npz` array with map-signature validation.

**To run from Streamlit:**
1. Select `Room 2 — SARSA` from the mode dropdown.
2. Adjust alpha, gamma, max_steps, epsilon schedule, and episodes in the sidebar.
3. Click **Train** to run SARSA.
4. View metrics, Q-values, greedy policy, evaluation, and trajectory in the tabs.

**Running SARSA experiments:**
```bash
python -c "from training.sarsa_experiments import run_screening_experiments; run_screening_experiments()"
```

Results are saved to `storage/experiments/room2_sarsa/`.

## Q-Learning (Room 3)

Room 3 uses **Q-Learning** — an off-policy TD control algorithm for the Key Vault map.

**Augmented state:** `(row, column, has_key)` — Cartesian product of non-wall cells × `{False, True}`.

**Q-Learning update (off-policy):**
```
Q(s,a) ← Q(s,a) + α[r + γ maxₐ' Q(s',a') − Q(s,a)]
```

- The target uses the greedy maximum over next-state actions, **not** the behaviour action (off-policy).
- Terminal: `target = reward` (no bootstrap).
- Truncated (timeout): `target = reward` — no bootstrap.
- The locked exit without the key is **not** terminal; the agent must collect the key first.
- Key reward is issued exactly once per episode.
- Epsilon-greedy behaviour policy (same schedule as SARSA).
- Snapshots capture Q-table + evaluation rollout at episode milestones.
- Two policy views: without key and with key.
- Model persistence with schema version, algorithm tag, map signature, and state schema validation.

**To run from Streamlit:**
1. Select `Room 3 — Q-Learning` from the mode dropdown.
2. Adjust alpha, gamma, max_steps, epsilon schedule, and episodes in the sidebar.
3. Click **Train** to run Q-Learning.
4. View training progress, policies (with/without key), Q-values, replay, and evaluation in the tabs.

**Running Q-Learning experiments:**
```bash
python -c "from training.q_learning_experiments import run_screening_experiments; run_screening_experiments()"
```

## Function Approximation — Room 4: Momentum Chamber

Room 4 uses **semi-gradient SARSA** with **tile coding** and a **linear action-value function** for a continuous 10×10 metre room.

**State:** `(X, Y, Vx, Vy)` — continuous position, integer velocity in {-1, 0, 1}.

**Actions:** 9 velocity vectors (STOP, 4 cardinal, 4 diagonal). Diagonal movement has magnitude √2 per step. Velocity is set instantaneously (no acceleration).

**Tile coding:** Default 8 tilings × 10×10 tiles × 3×3 velocity categories = 7200 deterministic features. Exactly one tile active per tiling. Nearby states share features; distant states do not.

**Update equation:**
```
δ = r + γQ(s′,a′) − Q(s,a)        # for non-terminal S'
δ = r − Q(s,a)                     # for terminal or truncated S'
w[a] += α/n · δ · x(s)             # per active feature, per action weight vector
```

**Reward components:**
- `step = -0.01` per time step
- `exit = +100.0` when within 0.35m of exit centre (9.5, 9.5)
- `boundary_collision = -1.0` when movement would exit [0, 10]²; blocked velocity component zeroed
- `timeout = -25.0` at max_steps (default 750)
- `distance_progress_scale × (dist_before − dist_after)` configurable (0.0, 0.5, 1.0)

**Start modes:** `FIXED` (0.5, 0.5, 0, 0), `RANDOM_LOWER_LEFT` (X,Y ∈ [0.25, 3.0]²), `RANDOM_ROOM` (any valid position outside exit). Training default is RANDOM_LOWER_LEFT.

**Evaluation** includes fixed-start, fixed unseen starts, and random-start metrics.

**To run from Streamlit:** Select `Room 4 — Function Approximation` mode.

**Running experiments:**
```bash
python -c "from training.approximate_sarsa_experiments import run_screening_stage_a; results = run_screening_stage_a()"
```

## Algorithm Comparison

Compare SARSA, Q-Learning, and Approximate SARSA on benchmarks:

- **Comparison A (tabular)**: SARSA vs Q-Learning on Room 2 benchmark with identical hyperparameters and paired training seeds.
- **Comparison B (tabular)**: Each algorithm's best-tuned configuration on Room 2.
- **Comparison C (continuous)**: Approximate SARSA with tile coding on Room 4.
- Reports success rate, return, steps, collisions, side-by-side.

**To run from Streamlit:** Select `Algorithm Comparison` mode.

**Running all tests:**
```bash
pytest -v
```

## Cell Symbols

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
├── app.py                 # Streamlit app (manual mode + 5 RL modes)
├── config/                # Configuration dataclasses and room specs
├── core/                  # Shared types (RewardConfig, StepResult, enums, all configs)
├── environments/          # Room environments (grid-based Room 1–3, continuous Room 4)
├── agents/                # RL algorithms (DP, SARSA, Q-Learning, Approximate SARSA)
├── features/              # Tile coding for function approximation (NEW)
├── training/              # Training pipelines and experiment runners
├── visualization/         # Policy symbols, Q-value tables, trajectory overlays, action fields
├── storage/               # Saved models, metrics, episodes, experiment results
├── tests/                 # Comprehensive tests (263 tests: 44 env + 41 DP + 49 SARSA + 54 Q-Learning + 75 Approx SARSA)
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
| Member A | Shared grid, Dynamic Programming, SARSA, Testing |
| Member B | Q-Learning, Experiments |
| Member C | Room 4, Streamlit, Deployment |
| Everyone | Documentation, Defense |
