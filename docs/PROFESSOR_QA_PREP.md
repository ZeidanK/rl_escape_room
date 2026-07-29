# Professor Call Preparation: Questions and Strong Answers

This document is written for a 30-minute oral discussion about the
Reinforcement Learning Escape Room project. Use it as a speaking guide: answer
with the short version first, then add the details if the professor asks for
more.

## 30-Minute Call Strategy

Start with this 60-second overview:

> Our project is a Streamlit escape-room learning environment for reinforcement
> learning. It contains four required rooms and one optional bonus room. Each
> room increases the difficulty and introduces a different RL method. Room 1 is
> a known 10x10 grid MDP solved with Value Iteration. Room 2 is an unknown
> stochastic grid solved with SARSA. Room 3 adds a key/locked-exit mechanic and
> uses Q-Learning with an augmented state. Room 4 moves from tabular grids to a
> continuous 10x10 metre state space and uses semi-gradient SARSA with tile
> coding. Room 5 is optional bonus work using a small NumPy DQN with replay and
> a target network. We focused on clear environment-agent separation,
> reproducible experiments, saved models, visualizations, tests, and a deployed
> Streamlit app.

Recommended time split:

1. Project overview and demo: 3-5 minutes.
2. Room 1-4 algorithms: 12-15 minutes.
3. Results and comparison: 5 minutes.
4. Testing, reproducibility, architecture: 3-5 minutes.
5. Limitations and future work: 2-3 minutes.

If you get a difficult question, use this structure:

1. Define the concept in one sentence.
2. Connect it to the exact room or code decision.
3. Mention the measured result or limitation honestly.

## Project Facts to Remember

| Topic | Answer |
|---|---|
| Project name | Reinforcement Learning Escape Room |
| Platform | Python + Streamlit |
| Public app | https://rlescaperoom-etswi8z5v9b48mejvamdqw.streamlit.app/ |
| Required algorithms | Value Iteration, SARSA, Q-Learning, semi-gradient SARSA with tile coding |
| Bonus algorithm | NumPy DQN for dynamic obstacles |
| Rooms 1-3 | 10x10 grid worlds |
| Room 4 | Continuous 10x10 metre room with state `(X, Y, Vx, Vy)` |
| Main evaluation idea | Train with seeded experiments, then evaluate greedy policies over fixed seed sets |
| Test coverage | README reports 338 local tests |
| Saved artifacts | JSON metadata plus NPZ numeric arrays/weights |

Final measured results:

| Room | Algorithm | Best configuration | Success rate | Main note |
|---|---|---|---:|---|
| Room 1 | Value Iteration | `gamma=0.8`, `tol=0.01`, deterministic slip | 100% | Exact model-based solution |
| Room 2 | SARSA | `alpha=0.05`, `gamma=0.95`, `epsilon_decay=0.99` | 100% | On-policy TD control |
| Room 3 | Q-Learning | `alpha=0.5`, `gamma=0.99`, `epsilon_decay=0.999` | 100% | Key collection rate also 100% |
| Room 4 | Approximate SARSA | 16 tilings, 16x16 tiles, `alpha=0.05`, progress shaping | 60% fixed-start | Harder due to continuous state and generalization |
| Room 5 | NumPy DQN | hidden 48, lr 0.001, decay 0.96, obs 3.5m | 96.67% seeded-random | Optional bonus, reported separately |

## Core Concept Questions

### 1. What problem does the project solve?

Short answer:

The project demonstrates how different reinforcement learning algorithms solve
increasingly difficult escape-room environments, starting from exact planning
and moving toward model-free, continuous, and deep-value methods.

Detailed answer:

We designed a sequence of rooms where an agent must reach an exit while avoiding
walls, traps, slippery transitions, locked exits, or obstacles. The project is
not only about getting an agent to the goal; it is about matching the correct RL
method to the environment. When the transition model is known, we use Dynamic
Programming. When the model is unknown, we use sampled TD control. When the
state becomes continuous, a tabular Q-table is no longer practical, so we use
tile coding and a linear approximation. The optional Room 5 extends this to a
small DQN for dynamic obstacles.

### 2. What is reinforcement learning in this project?

Short answer:

RL is learning a policy through interaction: the agent observes a state, chooses
an action, receives reward, and updates its behavior to maximize long-term
return.

Detailed answer:

In our rooms, the state is the agent's position or continuous motion state, the
actions are movements, the reward encourages fast escape and penalizes unsafe
behavior, and the policy decides which action to take. The objective is to
maximize expected discounted return:

```text
G_t = r_{t+1} + gamma*r_{t+2} + gamma^2*r_{t+3} + ...
```

The algorithms differ mainly in how they estimate future return and whether
they need a known transition model.

### 3. Why did you use multiple algorithms instead of one algorithm for all rooms?

Short answer:

Because each room represents a different RL setting, and the assignment asks us
to show progression across those settings.

Detailed answer:

Value Iteration is ideal when the model is known. SARSA and Q-Learning are
model-free algorithms, so they learn from sampled experience. SARSA is
on-policy, which is useful for risk-aware behavior in stochastic environments.
Q-Learning is off-policy, so it can learn a greedy optimal policy while still
exploring. Room 4 has continuous state, so a finite table is not enough; tile
coding lets us generalize across nearby states. Room 5 is optional and shows how
a neural network can approximate Q-values from continuous observations.

### 4. What is the Markov property, and does your project satisfy it?

Short answer:

The Markov property means the current state contains all information needed to
predict the next state and reward distribution.

Detailed answer:

Rooms 1 and 2 use `(row, col)`, which is enough because the dynamics depend only
on current position and action. Room 3 needs an extra `has_key` flag because the
same physical cell can have different meaning depending on whether the key has
already been collected. Without `has_key`, the state would not be Markov. Room 4
uses `(X, Y, Vx, Vy)` because future position depends on current position and
velocity. Room 5 uses a 22-feature observation that includes position, velocity,
exit direction, and visible obstacle information.

### 5. What is a policy?

Short answer:

A policy maps states to actions.

Detailed answer:

In the grid rooms, the final policy can be visualized as arrows over the grid.
During training, SARSA and Q-Learning use epsilon-greedy policies: usually take
the action with the highest Q-value, but sometimes explore randomly. During
evaluation, we set epsilon to zero so the reported metrics reflect the learned
greedy policy.

### 6. What is the difference between value functions and Q-functions?

Short answer:

`V(s)` estimates how good a state is; `Q(s,a)` estimates how good it is to take a
specific action in a state.

Detailed answer:

Value Iteration in Room 1 primarily computes `V(s)` and then extracts a policy
by looking one step ahead. SARSA and Q-Learning directly learn `Q(s,a)`, which
is convenient when the model is unknown because the agent can update the value
of the exact action it sampled.

## Room 1: Value Iteration Questions

### 7. Why is Room 1 solved with Dynamic Programming?

Short answer:

Because Room 1 exposes a known transition model, so we can solve the MDP by
planning instead of learning from sampled episodes.

Detailed answer:

The environment has a function that returns the transition distribution for a
state-action pair. That lets Value Iteration compute the expected return of each
action exactly:

```text
V_new(s) = max_a sum_s' P(s'|s,a) * [r + gamma*V(s')]
```

For terminal outcomes, the bootstrap term is removed, so the contribution is
just the immediate reward. The algorithm repeats synchronous Bellman backups
until the maximum change in values is below the tolerance.

### 8. What does "known model" mean?

Short answer:

It means we know the transition probabilities and rewards for every action in
every state.

Detailed answer:

For example, on a slippery cell the intended direction may happen with
probability 0.80, while slipping left or right may each happen with probability
0.10. Since those probabilities are known, Room 1 can use expected values
instead of sampling many episodes.

### 9. What is the Bellman optimality equation?

Short answer:

It expresses the optimal value of a state as the best expected immediate reward
plus discounted future value.

Detailed answer:

For a state-value function:

```text
V*(s) = max_a sum_s' P(s'|s,a) * [r + gamma*V*(s')]
```

This is the core equation behind Value Iteration. Q-Learning can be viewed as a
sample-based approximation of the Bellman optimality update.

### 10. Why does Value Iteration converge?

Short answer:

Because the Bellman optimality operator is a contraction when `gamma < 1`.

Detailed answer:

Each sweep brings the value function closer to the fixed point. We also enforce
`gamma` in `[0, 1)` for Value Iteration. In practice, we stop when
`max_s |V_new(s) - V(s)|` is below the configured tolerance or when the maximum
iteration limit is reached.

### 11. Why did the best Room 1 result use `gamma=0.8`?

Short answer:

Because the room rewards strongly favor fast escape, and a lower discount still
captures the exit reward while emphasizing shorter paths.

Detailed answer:

Higher `gamma` values value distant rewards more, which can matter in long
horizon tasks. In Room 1, the exit is reachable in a small number of steps and
each step has a penalty. The measured best deterministic-slip configuration was
`gamma=0.8`, `tol=0.01`, with 100% success, mean return 86.0, and 14 mean steps.

### 12. What is the difference between Value Iteration and Policy Iteration?

Short answer:

Value Iteration directly updates values using the Bellman optimality operator;
Policy Iteration alternates between evaluating a policy and improving it.

Detailed answer:

Policy Iteration often needs fewer outer iterations, but each policy evaluation
can be more expensive. Value Iteration is simpler: repeatedly compute the best
one-step lookahead value for every state, then extract the greedy policy.

## Room 2: SARSA Questions

### 13. Why did you use SARSA in Room 2?

Short answer:

Room 2 is an unknown stochastic environment, and SARSA is an on-policy TD
control method that learns directly from sampled experience.

Detailed answer:

The agent does not enumerate transition probabilities during training. Instead,
it interacts with the environment, observes `(state, action, reward, next_state,
next_action)`, and updates its Q-table. The room includes slippery cells and a
trap, so on-policy learning is useful because the update accounts for the
actual exploratory behavior of the agent.

### 14. What does SARSA stand for?

Short answer:

State, Action, Reward, next State, next Action.

Detailed answer:

Those five values are exactly what the update uses:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma*Q(s',a') - Q(s,a)]
```

The key part is `a'`: SARSA uses the next action actually selected by the
current behavior policy.

### 15. Why is SARSA called on-policy?

Short answer:

Because it learns the value of the same policy it uses to behave.

Detailed answer:

During training, the behavior policy is epsilon-greedy. SARSA's target includes
`Q(s',a')`, where `a'` is selected by that epsilon-greedy policy. That means the
learned values include the risk of future exploratory actions. In dangerous
environments, SARSA may learn safer behavior than Q-Learning because it accounts
for the possibility of exploration near traps.

### 16. How did SARSA perform?

Short answer:

The best confirmed SARSA setup reached 100% success on Room 2.

Detailed answer:

The best configuration was `alpha=0.05`, `gamma=0.95`, and
`epsilon_decay=0.99`, trained for 5000 episodes across 5 seeds and evaluated
with 100 evaluation episodes. It achieved 100% success, mean return 84.76, and
about 15.06 mean successful steps.

### 17. What happens if epsilon is too high or too low?

Short answer:

Too high means the agent keeps exploring and may not exploit what it learned;
too low means it may stop exploring too early and get stuck with a weak policy.

Detailed answer:

We start with high epsilon so the agent samples the environment broadly. Then
epsilon decays so the agent gradually exploits the Q-values it has learned. We
use a minimum epsilon during training to preserve some exploration, while
evaluation uses epsilon zero to measure the greedy learned policy.

### 18. Why does the final TD update not bootstrap on timeout?

Short answer:

Because the episode has ended, so there is no valid next decision point in that
episode.

Detailed answer:

For terminal and truncated transitions, the target is just `reward`. This
prevents the algorithm from adding an estimated future value after the episode
has already stopped. We apply the same convention across SARSA, Q-Learning,
Approximate SARSA, and DQN.

## Room 3: Q-Learning Questions

### 19. Why did you use Q-Learning in Room 3?

Short answer:

Room 3 has an unknown model and a key-collection mechanic, and Q-Learning is a
good off-policy control method for learning the greedy optimal action values.

Detailed answer:

The environment requires the key before the locked exit becomes a success. We
represent the state as `(row, col, has_key)`, so the agent can distinguish
between being at the same grid cell before and after collecting the key.
Q-Learning learns from sampled transitions but uses the greedy max over the next
state in its update.

### 20. What is the Q-Learning update?

Short answer:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma*max_a' Q(s',a') - Q(s,a)]
```

Detailed answer:

The target uses the best possible next action according to the current Q-table,
not necessarily the action the agent actually takes next. That is why
Q-Learning is off-policy.

### 21. What is the difference between SARSA and Q-Learning?

Short answer:

SARSA uses the next action actually chosen; Q-Learning uses the best next
action.

Detailed answer:

SARSA target:

```text
r + gamma*Q(s',a')
```

Q-Learning target:

```text
r + gamma*max_a' Q(s',a')
```

So SARSA learns the value of the exploratory behavior policy, while Q-Learning
learns toward the greedy target policy. This is the key on-policy versus
off-policy distinction.

### 22. Why did Room 3 need 92 states?

Short answer:

There are 46 non-wall physical positions, and each position has two possible
key states: `has_key=False` or `has_key=True`.

Detailed answer:

The state count is:

```text
46 physical positions * 2 key flags = 92 states
```

This state augmentation is necessary for the Markov property. Without the key
flag, the same location would have different future consequences depending on
past history.

### 23. Is it a problem that some augmented states may be unreachable?

Short answer:

No, for this tabular project it is an acceptable simplicity tradeoff.

Detailed answer:

The table includes the Cartesian product of physical positions and key flags.
Some combinations might not occur naturally, but including them makes indexing
and validation predictable. The extra states are small in number, so the memory
cost is negligible.

### 24. How did Q-Learning perform?

Short answer:

The best Q-Learning setup reached 100% success and 100% key collection.

Detailed answer:

The best configuration was `alpha=0.5`, `gamma=0.99`,
`epsilon_decay=0.999`, trained for 5000 episodes across 5 seeds and evaluated
over 100 episodes. It achieved 100% success, 100% key collection, mean return
91.0, and 19.0 mean steps.

### 25. Why can Q-Learning sometimes be unstable with function approximation?

Short answer:

Because off-policy learning, bootstrapping, and function approximation together
can create instability.

Detailed answer:

This is often called the "deadly triad." The target depends on estimates from
the same learned function, while the data may come from a different behavior
policy. In our project, tabular Q-Learning is used in Room 3, so this risk is
much smaller. For the continuous Room 4, we intentionally use on-policy
semi-gradient SARSA with linear tile coding.

## Room 4: Approximate SARSA and Tile Coding Questions

### 26. Why could you not use a normal Q-table for Room 4?

Short answer:

Because Room 4 has continuous position values, so there are infinitely many
possible states.

Detailed answer:

The state is `(X, Y, Vx, Vy)`, where `X` and `Y` are continuous coordinates in a
10x10 metre room. A table would require one entry for every possible coordinate,
which is impossible. Function approximation solves this by learning weights
over features instead of storing a value for every exact state.

### 27. What is tile coding?

Short answer:

Tile coding converts a continuous state into a small set of active discrete
features using multiple overlapping grids.

Detailed answer:

Each tiling partitions the continuous room slightly differently. For any state,
one tile is active in each tiling. The Q-value is computed by summing the
weights for the active tiles for the selected action:

```text
Q(s,a) = w[a] dot x(s)
```

The overlapping tilings allow nearby states to share some features, so learning
from one state generalizes to nearby states.

### 28. Why use multiple tilings?

Short answer:

Multiple offset tilings give smoother generalization than one coarse grid.

Detailed answer:

With one tiling, two nearby states may fall on opposite sides of a bin boundary
and share no representation. With overlapping tilings, they likely share some
active features. That makes the approximation less brittle.

### 29. What is the semi-gradient SARSA update?

Short answer:

It is SARSA with a learned function approximator instead of a Q-table.

Detailed answer:

For non-terminal transitions:

```text
delta = r + gamma*Q(s',a') - Q(s,a)
w[a] <- w[a] + (alpha / num_tilings) * delta * x(s)
```

For terminal or truncated transitions:

```text
delta = r - Q(s,a)
```

Only the weights for the selected action and active features are updated.

### 30. Why divide alpha by the number of tilings?

Short answer:

Because each state activates one feature per tiling, so the raw update magnitude
would grow with the number of tilings.

Detailed answer:

If 16 tilings are active and we add `alpha*delta` to each active weight, the
effective update becomes much larger than with 4 tilings. Dividing by
`num_tilings` keeps the learning rate comparable across tile-coding settings.

### 31. Why was Room 4 success lower than the grid rooms?

Short answer:

Room 4 is much harder because it uses continuous states and requires
generalization instead of exact table lookup.

Detailed answer:

Rooms 1-3 have small tabular state spaces. Room 4 must approximate values over
continuous position and velocity. The final fixed-start success rate was 60%,
while generalization to unseen starts was weaker. We report that honestly as a
known limitation. It shows the real tradeoff of function approximation: it can
generalize, but performance depends heavily on representation, start
distribution, reward shaping, and training budget.

### 32. Was distance-progress reward shaping unfair?

Short answer:

No, because it is explicit, configurable, and reported as part of the reward
design.

Detailed answer:

The progress reward gives the agent denser feedback when it moves closer to the
exit. Without shaping, the agent may receive mostly small step penalties until
eventually reaching the exit, which makes learning much slower in a continuous
space. We expose the reward components and can set the progress scale to zero,
so it is transparent rather than hidden.

### 33. What are the Room 4 actions?

Short answer:

There are 9 discrete velocity actions: stop, four cardinal directions, and four
diagonals.

Detailed answer:

Each action directly sets `(Vx, Vy)` where each component is in `{-1, 0, 1}`.
The state then advances by simple Euler integration:

```text
position_next = position + velocity * dt
```

The project uses `dt=0.02` seconds for Room 4.

## Room 5: Optional DQN Questions

### 34. What is Room 5, and is it required?

Short answer:

Room 5 is optional bonus work: a continuous obstacle room solved with a small
NumPy DQN.

Detailed answer:

It is not part of the mandatory four-room path. It adds dynamic obstacle
avoidance, local observations, replay buffer training, a target network, and
model persistence. We report its results separately so it does not confuse the
required SARSA-vs-Q-Learning comparison.

### 35. Why use DQN in Room 5?

Short answer:

Because the observation is continuous and includes obstacle information, so a
neural approximator is more flexible than a tabular method.

Detailed answer:

The DQN input is a 22-feature vector containing normalized position, velocity,
exit direction, and up to four nearest visible obstacle records. The network
outputs one Q-value per action. The selected action is the one with the highest
Q-value during greedy evaluation.

### 36. Why use a replay buffer?

Short answer:

Replay makes training more stable by sampling older transitions instead of
learning only from consecutive correlated steps.

Detailed answer:

Without replay, every gradient step would use highly correlated data from the
current trajectory. The replay buffer mixes experiences from different times,
which improves sample efficiency and reduces oscillation.

### 37. Why use a target network?

Short answer:

A target network stabilizes bootstrapping by keeping the target values fixed
for several training steps.

Detailed answer:

If the online network is used for both prediction and target calculation, the
target moves every update. In DQN, the online network learns while the target
network is copied only periodically. The target is:

```text
target = r + gamma * max_a' Q_target(s',a')
```

For terminal or truncated transitions, the target is just `r`.

### 38. Why implement DQN in NumPy instead of PyTorch?

Short answer:

To keep the project dependency-light and easier to deploy in Streamlit.

Detailed answer:

The DQN is intentionally small: one hidden ReLU layer, manual backpropagation,
replay, target-network copies, epsilon-greedy exploration, snapshots, and
JSON/NPZ persistence. PyTorch would be more powerful for large experiments, but
NumPy was enough for this optional bonus room.

## Results and Comparison Questions

### 39. Which algorithm performed best?

Short answer:

There is no universal winner; performance depends on the environment and the
comparison setup.

Detailed answer:

Rooms 1-3 reached 100% success because they have small or exact state spaces.
Room 4 was harder due to continuous state approximation. In the matched Room 2
comparison, SARSA and Q-Learning both reached 100% success. Q-Learning had a
slightly higher mean return and fewer collisions in that benchmark, but the
result should not be generalized to all tasks.

### 40. How did you compare SARSA and Q-Learning fairly?

Short answer:

We compared them on the same Room 2 benchmark with matched hyperparameters,
paired training seeds, and identical evaluation seeds.

Detailed answer:

The matched comparison used `alpha=0.10`, `gamma=0.95`,
`epsilon_decay=0.995`, 2000 episodes, seeds 0-4, and 100 evaluation episodes
for each trained model. Both reached 100% success. Q-Learning had mean return
85.0 versus SARSA 83.8 and 0 collisions versus SARSA's 150 total collisions in
the matched evaluation.

### 41. Why does Q-Learning have fewer collisions in the matched comparison?

Short answer:

In this specific Room 2 setup, its greedy target learned a slightly cleaner
route under greedy evaluation.

Detailed answer:

SARSA learns the value of the exploratory policy, while Q-Learning learns toward
the greedy policy. Because final evaluation uses epsilon zero, Q-Learning's
greedy target can look better in this specific metric. We avoid claiming
Q-Learning is always better because SARSA can be safer during exploration in
other risky environments.

### 42. Why do you evaluate with epsilon zero?

Short answer:

Because we want to measure the final learned policy, not the noisy training
behavior.

Detailed answer:

During training, epsilon-greedy exploration is necessary. During evaluation, we
turn exploration off so the metrics represent what the learned Q-values
recommend. This also makes comparisons more stable and easier to interpret.

### 43. What metrics did you report?

Short answer:

Success rate, mean return, mean steps, and room-specific metrics like key
collection, collisions, traps, and generalization categories.

Detailed answer:

For Rooms 2 and 3, success rate and mean return show whether the learned policy
escapes reliably and efficiently. Room 3 also reports key collection rate.
Room 4 reports fixed-start and unseen/random-start success categories because
generalization matters. Room 5 reports fixed validation, seeded random, and
unseen random layouts separately.

### 44. How do step penalties make faster completion better?

Short answer:

Every step has a negative reward, so shorter successful paths lose fewer points.

Detailed answer:

For example, reaching the exit gives a large positive reward, but each move
costs something. A 14-step solution receives less total penalty than a 30-step
solution, so the return naturally favors faster completion.

### 45. Why not compare all algorithms on all rooms?

Short answer:

Because the algorithms are designed for different assumptions and state spaces.

Detailed answer:

Value Iteration needs a known transition model. Tabular SARSA and Q-Learning
need manageable discrete state spaces. Room 4 is continuous, so tabular methods
are not directly appropriate without discretizing. We therefore compare SARSA
and Q-Learning on the shared Room 2 grid benchmark, where both algorithms are
applicable under the same conditions.

## Implementation and Architecture Questions

### 46. How is the code organized?

Short answer:

The project separates environments, agents, visualization, game UI, training
pipelines, storage, and tests.

Detailed answer:

Environment classes define states, transitions, rewards, and terminal
conditions. Agent classes implement learning algorithms and policies. Training
scripts run experiments and save results. Visualization modules create policy
arrows, Q-tables, training curves, and action fields. The Streamlit app ties
those pieces together for demonstration.

### 47. Why separate agents from environments?

Short answer:

So algorithms stay reusable and do not depend on room-specific map logic.

Detailed answer:

The environment answers "what happens if this action is taken?" The agent
answers "which action should be taken and how should values be updated?" This
separation makes it easier to test each part, compare algorithms fairly, and
extend the project with new rooms.

### 48. How do you make experiments reproducible?

Short answer:

We use seeded random number generators and store experiment metadata.

Detailed answer:

The code uses `numpy.random.SeedSequence` to split a master seed into
independent streams for environment randomness, policy exploration, snapshot
evaluation, replay sampling, and other bookkeeping. Results also store
configuration, map signature, seeds, and Git commit where available.

### 49. Why split random number streams?

Short answer:

So changing one source of randomness does not accidentally change the whole
training trajectory.

Detailed answer:

For example, if snapshot evaluation used the same RNG stream as exploration,
adding a new snapshot could change future training actions. Separate streams
prevent that kind of hidden coupling and make experiments easier to reproduce.

### 50. How are models saved?

Short answer:

Metadata is saved in JSON and numeric parameters are saved in NPZ files.

Detailed answer:

For SARSA and Q-Learning, NPZ files store Q-table states and action values. For
Approximate SARSA and DQN, NPZ files store learned weights. JSON metadata stores
algorithm name, schema, configuration, map signature or environment config, and
validation information. Some models also use checksums to detect corrupted or
incompatible files.

### 51. What prevents loading the wrong model?

Short answer:

The loaders validate algorithm tags, state/action schemas, map signatures,
state counts, action counts, finite values, and checksums where applicable.

Detailed answer:

For example, a SARSA model should not be loaded as a Q-Learning model, and a
model trained on one map should not silently run on another map. Validation
turns those cases into clear errors instead of incorrect demonstrations.

### 52. What tests did you write?

Short answer:

Tests cover environment behavior, algorithm updates, persistence,
visualization, Streamlit rendering helpers, final summaries, and regression
cases.

Detailed answer:

Important tests verify exact update equations, terminal/truncated behavior,
known-model transition parity, key collection, Room 4 tile coding, approximate
SARSA updates, DQN artifacts, and final summary aggregation. The README reports
338 local tests.

## Hyperparameter Questions

### 53. What does alpha control?

Short answer:

`alpha` is the learning rate.

Detailed answer:

It controls how strongly each new TD error changes the current estimate. If it
is too high, values can oscillate or become unstable. If it is too low, learning
can be very slow. In Room 4, the effective step size is normalized by the number
of tilings.

### 54. What does gamma control?

Short answer:

`gamma` is the discount factor for future rewards.

Detailed answer:

A high `gamma` makes the agent care more about long-term rewards, which is
useful when the goal is far away. A lower `gamma` emphasizes immediate outcomes.
In our rooms, `gamma` was tuned because the best value depends on path length,
step penalties, traps, and delayed exit rewards.

### 55. What does epsilon control?

Short answer:

`epsilon` controls exploration in epsilon-greedy action selection.

Detailed answer:

With probability epsilon, the agent explores by choosing a random action. With
probability `1 - epsilon`, it exploits by choosing the current best action. We
start with high epsilon and decay it so the agent first discovers the map and
then commits more strongly to good actions.

### 56. How did you choose hyperparameters?

Short answer:

We used structured sweeps and confirmation runs with multiple seeds.

Detailed answer:

Rooms 2 and 3 used screening over learning rate, discount, and epsilon decay,
then confirmed the best candidates across multiple seeds. Room 4 used staged
search over tile-coding and learning parameters because each run is more
expensive. The final summary reports the best measured configuration for each
room.

### 57. Why are multiple seeds important?

Short answer:

Because one training run can be lucky or unlucky.

Detailed answer:

Stochastic transitions, exploration, random starts, obstacle layouts, and
network initialization can all affect results. Multiple seeds give a more
honest estimate of stability and variance.

## Reward and Environment Design Questions

### 58. How did you design the rewards?

Short answer:

Rewards encourage reaching the exit quickly and penalize unsafe or invalid
behavior.

Detailed answer:

Grid rooms use a negative step penalty, wall penalty, trap penalty, exit reward,
and timeout penalty. Room 3 adds key reward and locked-exit penalty. Room 4 uses
smaller step penalties because episodes are much longer, plus progress shaping,
boundary penalty, exit reward, and timeout penalty. Room 5 adds obstacle
collision penalties.

### 59. Are traps terminal?

Short answer:

No, traps are penalized but non-terminal by default.

Detailed answer:

This lets the agent recover from mistakes while still learning that traps are
bad. It also creates more interesting learning behavior than ending immediately
on every trap.

### 60. What is the difference between terminated and truncated?

Short answer:

`terminated` means the task ended naturally, usually success. `truncated` means
the episode stopped due to a limit, usually timeout.

Detailed answer:

In both cases we stop the episode. In our TD updates, neither terminal nor
truncated transitions bootstrap from the next state. The target is just the
observed reward.

### 61. Why use slippery cells?

Short answer:

They introduce stochasticity, making the problem more realistic than a fully
deterministic grid.

Detailed answer:

A slippery cell can cause the effective action to differ from the requested
action. That forces the policy to account for transition risk rather than only
planning the shortest path.

### 62. Why use a key mechanic in Room 3?

Short answer:

It forces the agent to solve a task where physical location alone is not enough
to define the state.

Detailed answer:

The key mechanic demonstrates state augmentation. The agent must first collect
the key, then reach the locked exit. This is why the state includes `has_key`.

## Demo Questions

### 63. What should we show first in the app?

Recommended answer:

Start with the home page to explain the room progression, then show Room 1's
value/policy view, Room 2 or Room 3 training curves, Room 4 trajectory/action
field, and the algorithm comparison page.

### 64. If the professor asks to run a policy, what should we point out?

Recommended answer:

Point out the state representation, the chosen action, reward events,
terminated/truncated status, and how the path reflects the learned policy. For
SARSA/Q-Learning, mention that the displayed/evaluated policy is greedy
(`epsilon=0`) while training used exploration.

### 65. If something in the demo is slow, what should we say?

Recommended answer:

The app loads saved showcase models and final artifacts for presentation. Full
hyperparameter searches are available in the training scripts but are more
expensive, so they are not meant to be rerun live during a 30-minute call.

## Critical Thinking and Limitation Questions

### 66. What is the biggest weakness of the project?

Strong answer:

The biggest weakness is Room 4 generalization. The tabular rooms perform very
well, but the continuous room is harder and the learned tile-coded policy does
not generalize equally well to all unseen starts. We report this directly
instead of hiding it. Better performance would likely require broader
hyperparameter search, richer features, longer training, or a neural method.

### 67. Did reward shaping bias the results?

Strong answer:

Reward shaping changes the learning signal, so yes, it affects what the agent
learns. We handle that by making the shaping explicit and configurable. The
purpose is to make sparse continuous learning feasible, not to hide the true
objective. The final evaluation still measures whether the agent actually
reaches the exit.

### 68. Could the agents be overfitting to one map?

Strong answer:

Yes, especially in the grid rooms where the policies are trained and evaluated
on fixed layouts. We reduce randomness issues through multiple seeds, but map
generalization is limited. That is why future work includes cross-validation
across multiple maps and broader random-layout evaluation.

### 69. Why is Room 5 reported separately?

Strong answer:

Room 5 is optional bonus work and uses a different problem setup with obstacle
layouts and DQN. Reporting it separately keeps the required algorithm
comparison clean and prevents overclaiming.

### 70. What would you improve with more time?

Strong answer:

I would improve Room 4 generalization first: broader training starts, longer
training, richer tile-coding search, or a neural approximator. I would also
evaluate all algorithms across more maps, add confidence intervals to the app,
and keep the deployed Streamlit artifacts synced with the latest final results.

### 71. Why not use deep learning for every room?

Strong answer:

Deep learning would be unnecessary for the tabular rooms. Value Iteration,
SARSA, and Q-Learning are more interpretable and better matched to small
discrete state spaces. A neural network adds complexity and instability where a
table or exact dynamic programming solution is simpler and more reliable.

### 72. Is 100% success enough to prove optimality?

Strong answer:

No. Success rate shows reliability, but not necessarily optimality. We also
look at mean return, mean steps, collisions, traps, and comparison under matched
conditions. For Value Iteration in the known model, optimality is supported by
Bellman convergence. For model-free methods, results are empirical.

## "Professor Tries to Catch You" Questions

### 73. If Q-Learning is off-policy, why does it still use epsilon-greedy actions?

Answer:

Off-policy does not mean no exploration. It means the behavior policy and target
policy are different. The behavior policy is epsilon-greedy so the agent
explores, but the update target uses the greedy max action.

### 74. If SARSA is safer, why did Q-Learning get fewer collisions in your matched result?

Answer:

"SARSA is safer" is a common tendency, not a universal guarantee. In our
specific Room 2 map and greedy evaluation setup, Q-Learning learned a slightly
cleaner greedy route. SARSA's on-policy update accounts for exploratory risk
during training, but final evaluation removes exploration.

### 75. Why is the terminal state's value zero in Value Iteration if the exit reward is positive?

Answer:

The positive reward is received when transitioning into the terminal state. Once
the agent is already terminal, there are no future rewards, so `V(terminal)=0`.

### 76. Why not bootstrap after truncation?

Answer:

Because the episode ended due to the time limit, so there is no next action in
that episode. Bootstrapping could overestimate the final transition by adding a
future value that the agent will not actually receive.

### 77. Does Room 3's key flag break the 10x10 grid requirement?

Answer:

No. The physical environment is still a 10x10 grid. The learning state is
augmented with `has_key` because the task requires memory of whether the key has
been collected.

### 78. Are unreachable augmented states a bug?

Answer:

No. They are a deliberate tabular representation choice. The extra states are
few, simplify the Q-table schema, and do not affect correctness because updates
only occur for states that are actually visited.

### 79. Does tile coding discretize the state?

Answer:

It creates discrete features from continuous states, but it is not the same as a
single hard discretization. Multiple overlapping tilings allow generalization
across nearby states.

### 80. Why is Approximate SARSA "semi-gradient"?

Answer:

Because we take the gradient only with respect to the current estimate
`Q(s,a)`. We treat the bootstrap target `r + gamma*Q(s',a')` as fixed during
the update.

### 81. What is the deadly triad?

Answer:

The deadly triad is the combination of function approximation, bootstrapping,
and off-policy learning. Together they can cause instability or divergence. In
our project, Room 4 uses on-policy linear semi-gradient SARSA, and Room 5 uses
DQN stabilization tools like replay and a target network.

### 82. Why do you need a map signature?

Answer:

A Q-table trained on one map may be invalid on another map. The map signature
lets the loader detect a mismatch before using a model in the app.

### 83. Why are screenshots and Streamlit important for an RL project?

Answer:

They make the learning process inspectable. Instead of only reporting numbers,
the app shows policies, trajectories, training curves, and comparisons. That
helps verify that the agent learned sensible behavior.

## Code-Specific Questions

### 84. Where is Value Iteration implemented?

Answer:

`agents/dynamic_programming.py`, mainly in `ValueIterationAgent.solve`. It
iterates through all states, skips terminal states, computes each action's
expected value with `calculate_action_value`, and extracts a greedy policy after
convergence.

### 85. Where is SARSA implemented?

Answer:

`agents/sarsa.py`, mainly in `SarsaAgent.train` and `SarsaAgent.update`. The
training loop selects the next epsilon-greedy action before the update, which is
the on-policy part.

### 86. Where is Q-Learning implemented?

Answer:

`agents/q_learning.py`, mainly in `QLearningAgent.train` and
`QLearningAgent.update`. The update uses `max` over the next state's action
values.

### 87. Where is Approximate SARSA implemented?

Answer:

`agents/approximate_sarsa.py`. The `LinearTileQFunction` stores action-feature
weights, and `ApproximateSarsaAgent.train` performs the semi-gradient SARSA
update.

### 88. Where is tile coding implemented?

Answer:

`features/tile_coding.py`. It maps each continuous state to active feature
indices based on overlapping tilings and velocity categories.

### 89. Where is DQN implemented?

Answer:

`agents/dqn.py`. It includes a one-hidden-layer NumPy network, replay buffer,
target network updates, training loop, evaluation, and persistence.

### 90. How does the app avoid retraining everything live?

Answer:

The app can load saved showcase models and final experiment artifacts from
`storage/models` and `storage/experiments/final`. That makes the demo fast and
consistent while still keeping training scripts available.

## Short Answers for Rapid Fire

| Question | Best short answer |
|---|---|
| What is the agent? | The learner choosing actions in each room. |
| What is the environment? | The room dynamics, rewards, states, and terminal rules. |
| What is a state? | The information the agent observes before acting. |
| What is an action? | A movement decision. |
| What is reward? | The scalar feedback used to learn behavior. |
| What is return? | Discounted cumulative reward. |
| What is TD error? | The difference between the target and current value estimate. |
| What is exploration? | Trying actions to discover better behavior. |
| What is exploitation? | Choosing the best-known action. |
| What is epsilon-greedy? | Random action with probability epsilon, otherwise greedy. |
| What is model-free? | Learning without enumerating transition probabilities. |
| What is model-based? | Using a known transition/reward model for planning. |
| What is bootstrapping? | Updating an estimate using another learned estimate. |
| What is function approximation? | Representing values with learned weights instead of a table. |
| What is generalization? | Applying learning from seen states to unseen but similar states. |

## Closing Statement

Use this if the professor asks for a final summary:

> The main achievement is that the project does not only implement isolated
> algorithms; it shows why different RL methods are appropriate under different
> assumptions. We start with exact planning in a known MDP, move to model-free
> on-policy and off-policy TD control, then handle continuous state with tile
> coding and semi-gradient learning. The results are strong for the tabular
> rooms, honest about Room 4's generalization limits, and extended with an
> optional DQN room. The system is reproducible, tested, documented, persisted,
> and deployed in Streamlit for visual explanation.
