# Reinforcement Learning Escape Room

A Python-based reinforcement learning escape-room application with four required
rooms plus one optional bonus room, each solved by a different RL algorithm.

**Public Streamlit app:** https://rlescaperoom-etswi8z5v9b48mejvamdqw.streamlit.app/

## Project Objective

Apply four reinforcement learning algorithms — Value Iteration, SARSA,
Q-Learning, and semi-gradient SARSA with tile coding — to navigate a series of
grid-world and continuous control environments. Each room introduces a new
challenge: stochastic transitions, trap cells, key-collection mechanics, and
continuous state spaces.

Room 5 is implemented as optional bonus work: a continuous dynamic-obstacle
room solved with a lightweight NumPy DQN using replay and a target network.

## Assignment Requirements

| # | Requirement                                                                  | Status |
|---|------------------------------------------------------------------------------|--------|
| 1 | Four or more rooms with different tasks                                      | Done   |
| 2 | Each room has a final state (exit / timeout)                                 | Done   |
| 3 | Faster completion yields higher reward                                       | Done   |
| 4 | Rooms 1–3 use a 10×10 grid                                                   | Done   |
| 5 | Room 1 uses DP with a known model                                            | Done   |
| 6 | Room 2 uses SARSA with slippery cells and traps                              | Done   |
| 7 | Room 3 uses Q-Learning with key-collection mechanic                          | Done   |
| 8 | Room 4 uses a non-grid, continuous state space (X, Y, Vx, Vy)                | Done   |
| 9 | Room 4 is a 10×10 metre room                                                 | Done   |
| 10 | All agents run on the same platform (Streamlit)                              | Done   |
| 11 | Reproducible experiments with seeded RNG streams                             | Done   |
| 12 | Published as a public Streamlit app                                          | Done   |
| 13 | Defence presentation with oral questions                                     | Prepared |

## Room Overview

| Room | Name               | Algorithm                  | State Space          | Model Known | On/Off Policy |
|------|--------------------|----------------------------|----------------------|-------------|---------------|
| 1    | Ice Maze           | Value Iteration (DP)       | 10×10 grid           | Yes         | —             |
| 2    | Laser Corridor     | SARSA                      | 10×10 grid           | No          | On-policy     |
| 3    | Key Vault          | Q-Learning                 | Grid × {has_key}     | No          | Off-policy    |
| 4    | Momentum Chamber   | Semi-gradient SARSA + TC   | Continuous (X,Y,Vx,Vy) | No        | On-policy     |
| 5    | Dynamic Obstacles  | NumPy DQN                  | 22-feature continuous observation | No | Off-policy |

Room 3 has 46 non-wall physical positions, so its tabular state space is
46 positions x 2 key flags = 92 states.

### Room 1 — Ice Maze (Value Iteration)

- **State**: 10×10 grid cell (row, col).
- **Actions**: UP, RIGHT, DOWN, LEFT.
- **Rewards**: step=−1, wall=−4, exit=+99, trap=−21, timeout=−31.
- **Terminal**: exit cell.
- **Transition model**: known; stochastic on slippery cells (intended=0.80,
  left=0.10, right=0.10).
- **Algorithm**: synchronous Value Iteration.
- **Update**: `V_new(s) = max_a Σ P(s'|s,a) × [r + (terminated ? 0 : γV(s'))]`.
- **Stop condition**: `max_s |V_new(s) − V(s)| < tolerance` or max iterations.

### Room 2 — Laser Corridor (SARSA)

- **State**: 10×10 grid cell (row, col).
- **Actions**: UP, RIGHT, DOWN, LEFT.
- **Rewards**: same additive structure as Room 1 + trap penalty.
- **Terminal**: exit cell; truncated at max_steps.
- **Transition model**: unknown; agent learns from sampled experience.
- **Algorithm**: SARSA (on-policy TD control).
- **Update**: `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]`.
- **Exploration**: epsilon-greedy with exponential decay.

### Room 3 — Key Vault (Q-Learning)

- **State**: (row, col, has_key) — Cartesian product of non-wall cells × {False, True}.
- **Actions**: UP, RIGHT, DOWN, LEFT.
- **State count**: 46 non-wall positions x 2 key flags = 92 states.
- **Rewards**: same as Room 2 + key_reward=+10 when first collecting the key.
- **Terminal**: exit cell **with key**; locked exit without key is not terminal.
- **Transition model**: unknown.
- **Algorithm**: Q-Learning (off-policy TD control).
- **Update**: `Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]`.
- **Exploration**: epsilon-greedy.

### Room 4 — Momentum Chamber (Approximate SARSA)

- **State**: (X, Y, Vx, Vy) — continuous position [0,10]², integer velocity {−1,0,1}.
- **Actions**: 9 velocity vectors (STOP + 4 cardinal + 4 diagonal).
- **Rewards**: step=−0.01, exit=+100, boundary=−1, timeout=−25,
  progress_scale×distance_gain.
- **Terminal**: within 0.35 m of exit centre (9.5, 9.5); truncated at 750 steps.
- **Transition model**: unknown.
- **Algorithm**: semi-gradient SARSA with tile coding & linear approximation.
- **Update**: `δ = r + γQ(s',a') − Q(s,a)`; `w[a] += (α/n) · δ · x(s)`.
- **Feature representation**: tile coding — multi-scale overlapping tilings
  with velocity categories.

### Room 5 - Dynamic Obstacles (Optional NumPy DQN)

- **State**: 22-feature continuous observation vector: normalized position,
  velocity, exit delta, and nearest K=4 visible obstacle records
  `(visible, dx/X, dy/X, distance/X)`.
- **Actions**: same 9 velocity vectors as Room 4.
- **Environment**: continuous 10x10 metre room, seeded random obstacle layouts,
  exact obstacle width 0.5m, configurable obstacle count and observation
  distance X.
- **Rewards**: small step penalty, distance-progress shaping, exit reward,
  obstacle-collision penalty, boundary penalty, and timeout penalty.
- **Terminal**: exit success; obstacle collision terminates as failure; timeout
  truncates the episode.
- **Algorithm**: NumPy DQN with replay buffer, epsilon-greedy exploration,
  target network updates, mini-batch TD learning, snapshots, evaluation, and
  JSON/NPZ model persistence with checksums.

## Increasing Difficulty

1. **Room 1** – Known MDP, solved via dynamic programming (exact, no sampling).
2. **Room 2** – Unknown MDP with stochastic transitions; on-policy TD learns
   risk-aware behaviour.
3. **Room 3** – Extended state space (key flag); off-policy Q-Learning learns
   optimal value despite exploratory actions.
4. **Room 4** – Continuous state; function approximation generalises across
   infinitely many states; semi-gradient update trades bias for variance.

5. **Room 5** - Optional dynamic-obstacle DQN extension with local observations,
   seeded layouts, and held-out layout evaluation.

## Project Architecture

```
rl_escape_room/
├── app.py                 # Streamlit web interface (5 primary presentation modes)
├── requirements.txt       # Python dependencies
├── core/                  # Shared types, configs, enums
├── environments/          # Room environments (grid Room 1–3, continuous Room 4)
├── agents/                # RL algorithms (DP, SARSA, Q-Learning, Approx SARSA)
├── features/              # Tile coding for function approximation
├── training/              # Experiment pipelines, comparisons, utilities
├── visualization/         # Policy arrows, Q-tables, training curves, action fields
├── storage/               # Saved models, experiment results, metrics
├── tests/                 # 291 tests across all components
├── docs/                  # Design docs, defence prep, screenshots
└── tools/                 # Screenshot capture, result extraction
```

## Installation

```bash
pip install -r requirements.txt
```

## Local Run

```bash
streamlit run app.py
```

## Running Tests

```bash
pytest -v
```

**Exact test count: 308 tests (current local result).**

## Generating Local Showcase Models

Rooms 2-5 in the Escape Room Showcase load saved local models from
`storage/models/`. Generate them with:

```bash
python tools/generate_local_models.py
```

For a quick wiring check, use:

```bash
python tools/generate_local_models.py --smoke
```

For deterministic committed showcase artifact names, use:

```bash
python tools/generate_local_models.py --showcase
```

## Best Measured Parameters

| Room | Algorithm          | Best Config                                                                  |
|------|--------------------|------------------------------------------------------------------------------|
| 1    | Value Iteration    | gamma=0.8, tolerance=0.01, slip=deterministic                                |
| 2    | SARSA              | alpha=0.05, gamma=0.95, epsilon_decay=0.99, episodes=5000                    |
| 3    | Q-Learning         | alpha=0.50, gamma=0.99, epsilon_decay=0.999, episodes=5000                   |
| 4    | Approximate SARSA  | tilings=16, tiles_xy=16, alpha=0.05, progress_scale=1.0, epsilon_decay=0.995 |
| 5    | NumPy DQN          | hidden=48, lr=0.001, gamma=0.98, epsilon_decay=0.96, obs=3.5                 |

## Final Local Measured Results

| Room | Algorithm          | Success Rate | Mean Return | Mean Steps | Notes                      |
|------|--------------------|:------------:|:-----------:|:----------:|----------------------------|
| 1    | Value Iteration    | 100.00%      | 86.0        | 14.0       | Deterministic slip config  |
| 2    | SARSA              | 100.00%      | 84.8        | 15.1       | 5 seeds, 100 eval episodes |
| 3    | Q-Learning         | 100.00%      | 91.0        | 19.0       | 5 seeds, 100 eval episodes |
| 4    | Approximate SARSA  | 60.00%       | —           | —          | Fixed training start       |
| 5    | NumPy DQN          | 96.67%       | N/A         | N/A        | Optional; seeded_random_layouts, 5 seeds x 12 evals |

Room 4 generalisation: fixed unseen=8.00%, random lower-left=32.80%,
random room=14.40%. Fixed-start success is 60.00% with visible seed variance
across the five confirmation seeds.

Room 5 optional evaluation is reported with named categories:
fixed_validation_layout=40.00%, seeded_random_layouts=96.67%, and
unseen_random_layouts=86.67%. This result is reported separately and is not
part of the SARSA-vs-Q-Learning comparison.

## Matched SARSA vs Q-Learning Comparison

Room 2 benchmark, identical hyperparameters (α=0.10, γ=0.95, ε_decay=0.995,
2000 episodes), paired seeds 0–4, 100 eval episodes each.

| Metric            | SARSA     | Q-Learning |
|-------------------|:---------:|:----------:|
| Success rate      | 100.00%   | 100.00%    |
| Mean return       | 83.8      | 85.0       |
| Mean steps        | 15.3      | 15.0       |
| Total collisions  | 150       | 0          |
| Total traps       | 0         | 0          |
| Paired SR diff    | —         | 0.0000     |

Both algorithms reach 100% success on Room 2. Q-Learning achieves slightly
higher return and fewer collisions. SARSA learns risk-aware behaviour (it
collides more because it explores; the on-policy objective includes
exploratory actions in the backup).

## Tuned Comparison

Each algorithm's best-tuned configuration on Room 2 (5000 episodes, 5 seeds,
100 eval episodes):

| Algorithm   | Config                            | SR       | Return | Steps |
|-------------|-----------------------------------|:--------:|:------:|:-----:|
| SARSA       | α=0.05, γ=0.95, ed=0.99          | 100.00%  | 84.8   | 15.1  |
| Q-Learning  | α=0.50, γ=0.99, ed=0.999         | 100.00%  | 85.0   | 15.0  |

Both algorithms achieve perfect success with very similar return and steps.
No single algorithm dominates across all metrics.

## Experiment Methodology

- **Room 1**: Sweep over gamma (0.8, 0.9, 0.99), tolerance (0.001, 0.01, 0.1),
  and slip configuration (deterministic, stochastic). Rank by success rate.
- **Room 2–3**: Two-stage screening. Stage 1: 36 configs (4 α × 3 γ × 3 decay)
  at 2000/1000 episodes. Top 5 advance to confirmation: 5 seeds × 5000 episodes,
  evaluation on 100 seeds.
- **Room 4**: Staged search. Stage A: 15 one-factor-at-a-time configs at 250
  episodes. Stage B: combine best 2 per factor (32 combos) at 500 episodes.
  After the normal confirmation pass, the pipeline runs a targeted
  3000-episode follow-up for the best fixed-start config and a mixed-start
  variant when refreshing final artifacts.
  Top 5 advance to confirmation: 5 seeds × 1500 episodes, 4 start categories.
- **Comparison**: Matched (identical parameters) and tuned (best per-algorithm)
  on Room 2.

## Reproducibility and Seeds

- Independent RNG streams per component (env, policy, snapshots, bookkeeping)
  via `numpy.random.SeedSequence`.
- Every experiment records Git commit, map signature, parameter set, and
  training seeds.
- Rolling back to the same commit with the same seed reproduces results
  exactly.

## Persistence and Model Loading

- SARSA/Q-Learning: JSON metadata + `.npz` Q-table array.
- Approximate SARSA: JSON metadata + `.npz` weight matrix.
- Room 5 DQN: JSON metadata + `.npz` network weights with checksum validation.
- Map signature validation prevents loading models on incompatible maps.
- Algorithm tag prevents loading a SARSA model as Q-Learning.
- All experiment results stored in `storage/experiments/final/`.

## Screenshots

Screenshots from the Streamlit application:

| Mode                          | Screenshot                        |
|-------------------------------|-----------------------------------|
| Home page                     | `docs/screenshots/home.png`       |
| Room 1 value/policy           | `docs/screenshots/room1_value_policy.png` |
| Room 2 training graphs        | `docs/screenshots/room2_training.png` |
| Room 3 policy (no key)        | `docs/screenshots/room3_policy_no_key.png` |
| Room 4 trajectory             | `docs/screenshots/room4_trajectory.png` |
| Algorithm comparison          | `docs/screenshots/comparison.png` |

## Known Limitations

- Room 4 success rate (60%) is lower than tabular rooms and varies across
  seeds. The continuous control task is inherently harder; generalisation to
  unseen starts is poor
  (8–33%).
- Tile coding parameters were tuned only via a staged-search on a limited
  budget. More exhaustive tuning or alternative representations (RBFs, deep
  nets) may improve performance.
- Algorithm comparison uses Room 2 as the sole benchmark. Results may not
  generalise to other grid configurations.
- The comparison does not include Room 4 (function approximation) because
  SARSA and Q-Learning in their tabular forms cannot handle the continuous
  state space.

## Future Work

- Room 5 future work: broader hyperparameter search and longer DQN training
  budgets, with fixed_validation_layout tracked separately from random-layout
  evaluation.
- Keep the public Streamlit deployment synced with the latest saved artifacts.
- Refresh screenshots from the deployed app after major UI changes.
- Extended hyperparameter search for Room 4 (Bayesian optimisation, more
  tilings configurations).
- Cross-validation across multiple room maps.

## Team Responsibilities

| Member | Primary Responsibility |
|--------|------------------------|
| Member A | Shared grid, Dynamic Programming, SARSA, Testing |
| Member B | Q-Learning, Experiments |
| Member C | Room 4, Streamlit, Deployment |
| Everyone | Documentation, Defence |

## Defence Preparation

See `docs/DEFENSE_PREP.md` for detailed answers covering Bellman equations,
Value Iteration, SARSA, Q-Learning, tile coding, semi-gradient updates,
hyperparameters, reproducibility, and comparison methodology.

## Deployment

```text
Public app: https://rlescaperoom-etswi8z5v9b48mejvamdqw.streamlit.app/
Local development: run with `streamlit run app.py`.
```
