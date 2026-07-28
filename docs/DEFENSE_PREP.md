# Defence Preparation

## Room 1 — Ice Maze / Value Iteration

| Aspect              | Detail |
|---------------------|--------|
| State               | (row, col) on a 10×10 grid, excluding walls. Terminal: exit cell only. |
| Actions             | UP (0), RIGHT (1), DOWN (2), LEFT (3). |
| Rewards             | Additive: step=−1, wall=−4, exit=+99, trap=−21, timeout=−31. |
| Transition model    | Known deterministic transitions on normal cells; stochastic on slippery cells: P(intended)=0.80, P(left)=0.10, P(right)=0.10. Provided by `get_transition_distribution()`. |
| Terminal/Truncated  | `terminated=True` when agent reaches exit. `truncated=True` at max_steps (timeout). Both prevent further `step()` calls. |
| Algorithm           | Synchronous Value Iteration. |
| Update equation     | `V_new(s) = max_a Σ_{s'} P(s'|s,a) × [r + (terminated(s') ? 0 : γV(s'))]` |
| On/off policy       | N/A (DP — no sampling). |
| Exploration         | None (uses the known model directly). |
| Key hyperparameters | gamma=0.8–0.99, tolerance=0.001–0.1, max_iterations=100–10000, slip configuration. |
| Implementation      | `agents/dynamic_programming.py::ValueIterationAgent.solve` |
| Representative test | `tests/test_dynamic_programming.py::TestValueIteration::test_converges_on_simple_map` |
| Measured result     | gamma=0.8, tolerance=0.01, slip=deterministic → 100% success, 86.0 mean return, 14.0 mean steps. |
| Likely oral question | "How does Value Iteration differ from Policy Iteration?" |
| Answer              | VI iterates on the value function directly using the Bellman optimality operator; PI alternates between policy evaluation (solving linear system) and policy improvement (greedy update). VI is simpler per-iteration but may need more iterations; PI typically converges in fewer iterations but each iteration is more expensive. |
| Numerical example   | State A has two actions: UP → (r=−1, s'=exit, terminated) with p=0.8; LEFT → (r=−1, s'=B) with p=0.2. V_new(A) = max[0.8×(99+0) + 0.2×(−1+γV(B)), ...]. |

## Room 2 — Laser Corridor / SARSA

| Aspect              | Detail |
|---------------------|--------|
| State               | (row, col) on a 10×10 grid. |
| Actions             | UP, RIGHT, DOWN, LEFT. |
| Rewards             | Same structure as Room 1. |
| Transition model    | Unknown — agent samples `env.step(action)` to collect experience. Slippery cells still produce stochastic outcomes (sampled, not enumerated). |
| Terminal/Truncated  | Same as Room 1. Truncation does **not** bootstrap in the update. |
| Algorithm           | SARSA (State-Action-Reward-State-Action). |
| Update equation     | `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]` |
| On/off policy       | **On-policy** — the target uses the action `a'` selected by the behaviour policy (epsilon-greedy), not the greedy maximum. |
| Exploration         | Epsilon-greedy with configurable schedule (constant, exponential, linear). Default: exponential, ε_start=1.0, ε_min=0.05, decay=0.995. |
| Key hyperparameters | alpha=0.05–0.50, gamma=0.90–0.99, epsilon decay=0.990–0.999, episodes=2000–5000. |
| Implementation      | `agents/sarsa.py::SarsaAgent.train`; update at `agents/sarsa.py::SarsaAgent._update` (inline). |
| Representative test | `tests/test_sarsa.py::TestUpdate::test_non_terminal_update_exact` |
| Measured result     | Best: alpha=0.05, gamma=0.95, decay=0.99 → 100% success, 84.8 mean return, 15.1 mean steps. |
| Likely oral question | "Why is SARSA called on-policy and what does that imply about the learned policy?" |
| Answer              | SARSA evaluates and improves the **same** policy that generates behaviour. The update `Q(s,a) ← ... + α[r + γQ(s',a') − Q(s,a)]` uses the actual next action `a'` under the epsilon-greedy policy. This means SARSA learns a policy that accounts for exploration — it learns to avoid states where exploratory suboptimal actions could lead to danger (e.g., traps). |
| Numerical example   | s=(3,4), a=DOWN, r=−1, s'=(4,4). epsilon=0.1. Under ε-greedy, a' might be LEFT (not greedy). Q(s,a) += 0.1 × [−1 + 0.95×Q(s',LEFT) − Q(s,a)]. |

## Room 3 — Key Vault / Q-Learning

| Aspect              | Detail |
|---------------------|--------|
| State               | (row, col, has_key) - 92 states (46 non-wall positions x 2 key flags). |
| Actions             | UP, RIGHT, DOWN, LEFT. |
| Rewards             | Same as Room 2 + key_reward=+10 on first key collection. Locked exit (no key): penalty=−6, not terminal. |
| Transition model    | Unknown. |
| Terminal/Truncated  | Terminal only when agent is on exit cell **and** has_key=True. Locked exit without key is a regular (non-terminal) cell. Truncation does not bootstrap. |
| Algorithm           | Q-Learning. |
| Update equation     | `Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]` |
| On/off policy       | **Off-policy** — the target uses the greedy maximum over next-state actions, not the behaviour policy's action. This allows learning the optimal Q-function while following an exploratory behaviour policy. |
| Exploration         | Epsilon-greedy (same schedule as SARSA). |
| Key hyperparameters | alpha=0.05–0.50, gamma=0.90–0.99, epsilon decay=0.990–0.999, episodes=1000–5000. |
| Implementation      | `agents/q_learning.py::QLearningAgent.train`; update at `agents/q_learning.py::QLearningAgent._update` (inline). |
| Representative test | `tests/test_q_learning.py::TestQLearningUpdate::test_off_policy_target_uses_max` |
| Measured result     | Best: alpha=0.50, gamma=0.99, decay=0.999 → 100% success, 91.0 mean return, 19.0 mean steps, 100% key collection. |
| Likely oral question | "How does off-policy learning differ from on-policy, and why does Q-Learning sometimes diverge with function approximation?" |
| Answer              | Off-policy learning separates behaviour from target: the behaviour policy explores, while the target policy is greedy. The deadly triad (function approximation, bootstrapping, off-policy learning) can cause divergence because the update is not a gradient of any objective — the target depends on the same parameters being learned. |
| Numerical example   | s=(3,7,has_key=True), a=RIGHT, r=−1, s'=(3,8,has_key=True). max_a' Q(s',a') = 50. Q(s,a) += 0.5 × [−1 + 0.99×50 − Q(s,a)]. |

## Room 4 — Momentum Chamber / Approximate SARSA

| Aspect              | Detail |
|---------------------|--------|
| State               | (X, Y, Vx, Vy) continuous; X,Y ∈ [0,10], Vx,Vy ∈ {−1, 0, 1}. |
| Actions             | 9 velocity vectors: STOP, UP, RIGHT, DOWN, LEFT, and 4 diagonals. |
| Rewards             | step=−0.01, exit=+100 (within 0.35 m of (9.5,9.5)), boundary=−1, timeout=−25, progress_reward = scale × (d_before − d_after). |
| Transition model    | Unknown; deterministic physics: position += velocity × dt, velocity set instantaneously. |
| Terminal/Truncated  | Terminal when distance to exit centre < 0.35 m. Truncated at 750 steps. |
| Algorithm           | Semi-gradient SARSA with tile coding and linear function approximation. |
| Update equation     | `δ = r + γQ(s',a') − Q(s,a)` (non-terminal) or `δ = r − Q(s,a)` (terminal/truncated). `Q(s,a) = w[a] · x(s)` where x(s) is the tile-coding feature vector. `w[a] += (α/n_tilings) × δ × x(s)`. |
| On/off policy       | **On-policy** — same as SARSA. |
| Exploration         | Epsilon-greedy on action values (computed from linear weights). |
| Key hyperparameters | num_tilings=4–16, tiles_xy=8–16, alpha=0.05–0.20, progress_scale=0.0–1.0, epsilon_decay=0.995–0.999. |
| Implementation      | `agents/approximate_sarsa.py::ApproximateSarsaAgent.train` |
| Representative test | `tests/test_approximate_sarsa.py::TestSemiGradientSARSA::test_exact_non_terminal_update` |
| Reporting note      | Fixed-start variance is computed from the five per-seed confirmation success rates, not hard-coded in the summary CSV. |
| Measured result     | Best: tilings=16, tiles=16, α=0.05, ps=1.0, ed=0.995 → fixed SR=60%, unseen SR=8%, random LL SR=32.8%, random room SR=14.4%. |
| Likely oral question | "Why does the semi-gradient update use the same weights for both the current estimate and the target?" |
| Answer              | The semi-gradient treats the target `r + γQ(s',a')` as fixed (no gradient through the target). This is biased but stable in the on-policy linear case. Full gradient methods (residual gradients) are unbiased but can have higher variance and slower convergence. With tile coding and on-policy SARSA, the semi-gradient is guaranteed to converge (to a bounded region around the optimum). |
| Numerical example   | State s=(5.0, 5.0, 0, 0), a=STOP (action 0). Tile coder activates features {142, 319, 507, ...}. Q(s,a) = Σ w[0][feat] for active feats. After step: r=−0.01, s'=(5.0, 5.0, 0, 0), a'=RIGHT (action 1). δ = −0.01 + 0.99×Q(s',RIGHT) − Q(s,STOP). w[STOP] += (0.05/16) × δ × x(s). |

## Room 5 - Dynamic Obstacles / NumPy DQN

| Aspect              | Detail |
|---------------------|--------|
| State               | 22-feature vector: normalized position, velocity, exit delta, and nearest K=4 visible obstacle records. |
| Actions             | Same 9 velocity vectors as Room 4. |
| Environment         | Continuous 10x10m room; seeded square obstacles; exact obstacle width 0.5m. |
| Observation         | Obstacles are included only if center-to-center distance <= observation distance X. Missing slots are padded. |
| Rewards             | step=-0.01, exit=+120, obstacle=-60, boundary=-1, timeout=-25, plus distance-progress shaping. |
| Terminal/Truncated  | Exit terminates as success. Obstacle collision terminates as failure. Timeout truncates. |
| Algorithm           | NumPy DQN with replay buffer, target network, epsilon-greedy behavior, mini-batch TD updates, snapshots, and JSON/NPZ persistence. |
| Evaluation labels   | Report fixed_validation_layout, seeded_random_layouts, and unseen_random_layouts separately. |
| On/off policy       | Off-policy control: behavior is epsilon-greedy, target uses max next-state Q-value from the target network. |
| Likely oral question | "Why use a target network and replay buffer?" |
| Answer              | Replay reduces correlation between consecutive samples; the target network slows changes in the bootstrap target, making DQN updates less unstable than using the online network for both sides. |
| Numerical example   | For transition `(s,a,r,s')`, target is `r + gamma * max_a' Q_target(s',a')` unless terminal/truncated, where target is just `r`. The online network is updated only for the selected action's prediction. |

## Cross-Cutting Topics

### Bellman Optimality

The Bellman optimality equation expresses the optimal value recursively:
`V*(s) = max_a Σ P(s'|s,a)[r + γV*(s')]`. It is a fixed-point equation;
Value Iteration solves it by repeated application. Q-Learning's update
`Q(s,a) ← ... + α[r + γ max_a' Q(s',a') − Q(s,a)]` is a stochastic
approximation to the Bellman optimality operator.

### Alpha (α), Gamma (γ), Epsilon (ε)

| Parameter | Role | Typical Values | Effect Too High | Effect Too Low |
|-----------|------|:--------------:|:---------------:|:--------------:|
| α (learning rate) | Step size for Q-updates | 0.05–0.50 | Oscillation, divergence | Slow learning |
| γ (discount) | Present value of future reward | 0.90–0.99 | Agent too myopic | Agent ignores long-term consequences |
| ε (exploration) | Probability of random action | Starts 1.0, decays to 0.02–0.05 | Agent never exploits | Agent never explores |

### Exploration vs Exploitation

- **Exploration**: taking suboptimal actions to discover better ones. Implemented via epsilon-greedy.
- **Exploitation**: taking the best-known action to maximise return.
- The epsilon schedule balances this trade-off; exponential decay from 1.0 to 0.05 is the default.

### Terminal vs Truncated

| Outcome | Bootstraps in TD update? | `terminated` | `truncated` |
|---------|:------------------------:|:------------:|:-----------:|
| Exit reached | No (target = r) | True | False |
| Timeout (max_steps) | No (target = r) | False | True |

Both stop the episode and prevent further `step()` calls. The distinction matters for bootstrapping: neither terminal nor truncated states bootstrap because there is no valid next state to estimate.

### RNG Streams

`numpy.random.SeedSequence` splits the master seed into independent sub-streams
for environment, policy (action selection), snapshots, and bookkeeping. This
ensures that changing one component (e.g., snapshot frequency) does not alter
the training trajectory.

### Reproducibility

Same seed + same code → exact same results. All experiments record the Git
commit, map signature, and parameter set. Rolling back to the committed
version reproduces every number.

### Persistence

Models are saved as JSON metadata + `.npz` arrays. Checksums and map
signatures prevent loading incompatible models. Algorithm tags prevent
loading a SARSA model as Q-Learning.

### Fair Comparison Methodology

- **Matched**: identical hyperparameters, same training seeds, same room map.
- **Tuned**: each algorithm's own best configuration (from independent
  screening experiments) on the same benchmark.
- Both compared on the same eval seeds (0–99), same max_steps, same reward
  structure, same slip configuration.
- Results report paired differences between algorithms trained with the same
  seed.
