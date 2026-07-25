"""NumPy DQN agent for optional Room 5."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from agents.tabular_utils import default_snapshot_episodes, epsilon_for_episode
from core.types import (
    DQNConfig,
    DQNEpisodeMetrics,
    DQNEvaluationSummary,
    DQNProgressCallback,
    DQNSnapshot,
    DQNTrainingResult,
    Room5Factory,
    Room5Observation,
    Room5RolloutResult,
    Room5TrajectoryStep,
    VelocityAction,
)


class DQNNetwork:
    """Small one-hidden-layer Q network implemented with NumPy."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: int,
        action_count: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.action_count = action_count
        rng = rng or np.random.default_rng(0)
        scale1 = np.sqrt(2.0 / max(1, input_dim))
        scale2 = np.sqrt(2.0 / max(1, hidden_units))
        self.W1 = rng.normal(0.0, scale1, size=(input_dim, hidden_units)).astype(np.float64)
        self.b1 = np.zeros(hidden_units, dtype=np.float64)
        self.W2 = rng.normal(0.0, scale2, size=(hidden_units, action_count)).astype(np.float64)
        self.b2 = np.zeros(action_count, dtype=np.float64)

    def copy(self) -> "DQNNetwork":
        other = DQNNetwork(self.input_dim, self.hidden_units, self.action_count)
        other.set_weights(self.weights)
        return other

    @property
    def weights(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.W1 = np.array(weights["W1"], dtype=np.float64, copy=True)
        self.b1 = np.array(weights["b1"], dtype=np.float64, copy=True)
        self.W2 = np.array(weights["W2"], dtype=np.float64, copy=True)
        self.b2 = np.array(weights["b2"], dtype=np.float64, copy=True)

    @classmethod
    def from_weights(cls, weights: dict[str, np.ndarray]) -> "DQNNetwork":
        input_dim = int(weights["W1"].shape[0])
        hidden_units = int(weights["W1"].shape[1])
        action_count = int(weights["W2"].shape[1])
        net = cls(input_dim, hidden_units, action_count)
        net.set_weights(weights)
        return net

    def predict(self, obs: Room5Observation | np.ndarray) -> np.ndarray:
        # Forward pass.  Supports either one observation vector or a batch,
        # which lets training reuse the same method for replay batches.
        x = np.asarray(obs, dtype=np.float64)
        if x.ndim == 1:
            h = np.maximum(0.0, x @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> tuple[float, float, float]:
        # Manual backpropagation for mean-squared TD error.  This keeps Room 5
        # dependency-light while still behaving like a small DQN.
        z1 = states @ self.W1 + self.b1
        h = np.maximum(0.0, z1)
        q = h @ self.W2 + self.b2
        pred = q[np.arange(states.shape[0]), actions]
        td = targets - pred
        loss = float(np.mean(td * td))

        dq = np.zeros_like(q)
        dq[np.arange(states.shape[0]), actions] = -2.0 * td / max(1, states.shape[0])
        dW2 = h.T @ dq
        db2 = dq.sum(axis=0)
        dh = dq @ self.W2.T
        dz1 = dh * (z1 > 0.0)
        dW1 = states.T @ dz1
        db1 = dz1.sum(axis=0)

        for grad in (dW1, db1, dW2, db2):
            np.clip(grad, -10.0, 10.0, out=grad)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        return loss, float(np.mean(np.abs(td))), float(np.max(np.abs(td)))


@dataclass
class _Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    # Fixed-size circular buffer for off-policy DQN updates.
    def __init__(self, capacity: int, input_dim: int) -> None:
        self.capacity = capacity
        self.input_dim = input_dim
        self.states = np.zeros((capacity, input_dim), dtype=np.float64)
        self.next_states = np.zeros((capacity, input_dim), dtype=np.float64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.dones = np.zeros(capacity, dtype=bool)
        self._idx = 0
        self._size = 0

    def add(self, transition: _Transition) -> None:
        idx = self._idx
        self.states[idx] = transition.state
        self.next_states[idx] = transition.next_state
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.dones[idx] = transition.done
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        idxs = rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )


def _select_dqn_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    max_val = np.max(q_values)
    tied = np.where(np.abs(q_values - max_val) < 1e-12)[0]
    return int(rng.choice(tied))


def _greedy_action(q_values: np.ndarray) -> int:
    max_val = np.max(q_values)
    tied = np.where(np.abs(q_values - max_val) < 1e-12)[0]
    return int(tied[0])


class DQNAgent:
    # Optional Room 5 agent.  It uses an online network for action selection
    # and a target network for more stable TD targets.
    def __init__(
        self,
        environment_factory: Room5Factory,
        config: DQNConfig,
        network: DQNNetwork | None = None,
    ) -> None:
        self.env_factory = environment_factory
        self.config = config
        sample_env = self.env_factory()
        sample_obs = sample_env.reset(seed=config.seed, layout_seed=config.seed)
        self.input_dim = len(sample_obs)
        self.action_count = len(sample_env.actions)
        self.network = network

    def _snapshot_episodes(self) -> tuple[int, ...]:
        if self.config.snapshot_episodes:
            return tuple(sorted(e for e in self.config.snapshot_episodes if 1 <= e <= self.config.episodes))
        return default_snapshot_episodes(self.config.episodes)

    def train(
        self,
        progress_callback: DQNProgressCallback | None = None,
        progress_every: int = 1,
    ) -> DQNTrainingResult:
        cfg = self.config
        # Independent RNG streams make the training run reproducible while
        # keeping layout generation, exploration, replay sampling, and network
        # initialization separate.
        seed_seq = np.random.SeedSequence(cfg.seed)
        env_ss, policy_ss, replay_ss, network_ss, snapshot_ss = seed_seq.spawn(5)
        env_rng = np.random.Generator(np.random.PCG64(env_ss))
        policy_rng = np.random.Generator(np.random.PCG64(policy_ss))
        replay_rng = np.random.Generator(np.random.PCG64(replay_ss))
        network_rng = np.random.Generator(np.random.PCG64(network_ss))
        snapshot_rng = np.random.Generator(np.random.PCG64(snapshot_ss))

        online = self.network or DQNNetwork(self.input_dim, cfg.hidden_units, self.action_count, network_rng)
        target = online.copy()
        replay = ReplayBuffer(cfg.replay_capacity, self.input_dim)
        metrics: list[DQNEpisodeMetrics] = []
        snapshots: dict[int, DQNSnapshot] = {}
        snapshot_eps = set(self._snapshot_episodes())
        global_step = 0

        for ep in range(cfg.episodes):
            epsilon = epsilon_for_episode(ep, cfg.epsilon)
            env_seed = int(env_rng.integers(0, 2**31))
            layout_seed = int(env_rng.integers(0, 2**31))
            env = self.env_factory()
            obs = env.reset(seed=env_seed, layout_seed=layout_seed)
            state = np.asarray(obs, dtype=np.float64)
            total_reward = 0.0
            obstacle_collisions = 0
            boundary_collisions = 0
            visible_steps = 0
            losses: list[float] = []
            mean_tds: list[float] = []
            max_tds: list[float] = []
            result = None

            for step in range(cfg.max_steps):
                q_values = online.predict(state)
                action = _select_dqn_action(q_values, epsilon, policy_rng)
                result = env.step(action)
                next_state = np.asarray(result.next_state, dtype=np.float64)
                done = result.terminated or result.truncated
                replay.add(
                    _Transition(
                        state=state,
                        action=action,
                        reward=float(result.reward),
                        next_state=next_state,
                        done=done,
                    )
                )
                total_reward += float(result.reward)
                if result.info.get("collision") == "obstacle":
                    obstacle_collisions += 1
                if result.info.get("collision") == "boundary":
                    boundary_collisions += 1
                if int(result.info.get("visible_obstacle_count", 0)) > 0:
                    visible_steps += 1

                if len(replay) >= max(cfg.batch_size, cfg.warmup_steps):
                    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size, replay_rng)
                    next_q = target.predict(next_states)
                    # DQN target uses the target network for the bootstrap
                    # value, and removes the future term on terminal steps.
                    targets = rewards + cfg.gamma * (1.0 - dones.astype(float)) * np.max(next_q, axis=1)
                    loss, mean_td, max_td = online.train_batch(
                        states, actions, targets, cfg.learning_rate
                    )
                    losses.append(loss)
                    mean_tds.append(mean_td)
                    max_tds.append(max_td)

                global_step += 1
                if global_step % cfg.target_update_interval == 0:
                    # Periodically copy online weights into the target network
                    # instead of chasing a moving target every gradient step.
                    target.set_weights(online.weights)
                state = next_state
                if done:
                    break

            m = DQNEpisodeMetrics(
                episode=ep,
                total_reward=float(total_reward),
                steps=step + 1,
                success=bool(result and result.info.get("success", False)),
                terminated=bool(result and result.terminated),
                truncated=bool(result and result.truncated),
                epsilon=epsilon,
                obstacle_collisions=obstacle_collisions,
                boundary_collisions=boundary_collisions,
                visible_obstacle_steps=visible_steps,
                mean_loss=float(np.mean(losses)) if losses else 0.0,
                mean_abs_td_error=float(np.mean(mean_tds)) if mean_tds else 0.0,
                max_abs_td_error=float(np.max(max_tds)) if max_tds else 0.0,
            )
            metrics.append(m)

            if (ep + 1) in snapshot_eps:
                frozen = _freeze_weights(online.weights)
                rollout = rollout_dqn_policy(
                    self.env_factory,
                    DQNNetwork.from_weights(frozen),
                    seed=int(snapshot_rng.integers(0, 2**31)),
                    layout_seed=int(snapshot_rng.integers(0, 2**31)),
                    max_steps=cfg.max_steps,
                )
                snapshots[ep + 1] = DQNSnapshot(
                    episode=ep + 1,
                    epsilon=epsilon,
                    weights=frozen,
                    rollout=rollout,
                )

            if progress_callback is not None and (ep % progress_every == 0 or ep == cfg.episodes - 1):
                progress_callback(ep, cfg.episodes, m)

        final_weights = _freeze_weights(online.weights)
        return DQNTrainingResult(
            config=cfg,
            weights=MappingProxyType(final_weights),
            metrics=tuple(metrics),
            snapshots=MappingProxyType(dict(snapshots)),
            final_epsilon=epsilon_for_episode(cfg.episodes - 1, cfg.epsilon),
            training_seed=cfg.seed,
            input_dim=self.input_dim,
            action_count=self.action_count,
        )


def _network_from_result(result: DQNTrainingResult) -> DQNNetwork:
    return DQNNetwork.from_weights(dict(result.weights))


def rollout_dqn_policy(
    environment_factory: Room5Factory,
    network: DQNNetwork,
    *,
    seed: int,
    layout_seed: int | None = None,
    epsilon: float = 0.0,
    max_steps: int | None = None,
) -> Room5RolloutResult:
    # Runs a trained DQN greedily by default and records both observations and
    # raw physical states for replay visualizations.
    env = environment_factory()
    layout = seed if layout_seed is None else layout_seed
    obs = env.reset(seed=seed, layout_seed=layout)
    start = env.raw_state
    rng = np.random.default_rng(seed + 991)
    limit = max_steps if max_steps is not None else env._max_steps
    steps: list[Room5TrajectoryStep] = []
    total_reward = 0.0
    obstacle_collisions = 0
    boundary_collisions = 0
    visible_steps = 0
    cumulative = 0.0

    for i in range(limit):
        if env.is_done:
            break
        q_values = network.predict(obs)
        action = _select_dqn_action(q_values, epsilon, rng) if epsilon > 0 else _greedy_action(q_values)
        raw_state = env.raw_state
        result = env.step(action)
        cumulative += float(result.reward)
        collision = result.info.get("collision")
        if collision == "obstacle":
            obstacle_collisions += 1
        if collision == "boundary":
            boundary_collisions += 1
        visible_count = int(result.info.get("visible_obstacle_count", 0))
        if visible_count > 0:
            visible_steps += 1
        steps.append(
            Room5TrajectoryStep(
                index=i,
                observation=obs,
                raw_state=raw_state,
                requested_action=VelocityAction(action),
                reward=float(result.reward),
                next_observation=result.next_state,
                next_raw_state=env.raw_state,
                collision=collision,
                event=result.info.get("event"),
                terminated=result.terminated,
                truncated=result.truncated,
                cumulative_reward=cumulative,
                visible_obstacle_count=visible_count,
                distance_to_exit_m=float(result.info.get("distance_after", 0.0)),
            )
        )
        total_reward += float(result.reward)
        obs = result.next_state
        if result.terminated or result.truncated:
            break

    return Room5RolloutResult(
        seed=seed,
        layout_seed=layout,
        start_state=start,
        final_state=env.raw_state,
        total_reward=float(total_reward),
        steps=len(steps),
        simulated_time_s=len(steps) * env.motion.time_step_s,
        success=bool(env._success),
        terminated=bool(env._terminated),
        truncated=bool(env._truncated),
        boundary_collisions=boundary_collisions,
        obstacle_collisions=obstacle_collisions,
        visible_obstacle_steps=visible_steps,
        trajectory=tuple(steps),
    )


def evaluate_dqn_policy(
    environment_factory: Room5Factory,
    weights_or_network: DQNTrainingResult | DQNNetwork | dict[str, np.ndarray],
    *,
    n_episodes: int = 25,
    seeds: range | list[int] | None = None,
    layout_seeds: range | list[int] | None = None,
    max_steps: int | None = None,
    category: str = "",
) -> DQNEvaluationSummary:
    if isinstance(weights_or_network, DQNTrainingResult):
        network = _network_from_result(weights_or_network)
    elif isinstance(weights_or_network, DQNNetwork):
        network = weights_or_network
    else:
        network = DQNNetwork.from_weights(weights_or_network)
    if seeds is None:
        seeds = range(n_episodes)
    seed_list = list(seeds)
    if layout_seeds is None:
        layout_list = [s + 10_000 for s in seed_list]
    else:
        layout_list = list(layout_seeds)
    rollouts = [
        rollout_dqn_policy(
            environment_factory,
            network,
            seed=int(seed),
            layout_seed=int(layout_list[i % len(layout_list)]),
            max_steps=max_steps,
        )
        for i, seed in enumerate(seed_list)
    ]
    return _summarize_room5_rollouts(tuple(rollouts), category=category)


def _summarize_room5_rollouts(
    rollouts: tuple[Room5RolloutResult, ...],
    *,
    category: str = "",
) -> DQNEvaluationSummary:
    n = len(rollouts)
    if n == 0:
        return DQNEvaluationSummary(
            n_episodes=0, successes=0, success_rate=0.0,
            mean_return=0.0, std_return=0.0, mean_steps=0.0,
            mean_successful_steps=None, truncated_count=0,
            obstacle_collision_count=0, boundary_collision_count=0,
            rollouts=(), category=category,
        )
    returns = [r.total_reward for r in rollouts]
    steps = [r.steps for r in rollouts]
    successes = sum(1 for r in rollouts if r.success)
    successful_steps = [r.steps for r in rollouts if r.success]
    return DQNEvaluationSummary(
        n_episodes=n,
        successes=successes,
        success_rate=successes / n,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_steps=float(np.mean(steps)),
        mean_successful_steps=float(np.mean(successful_steps)) if successful_steps else None,
        truncated_count=sum(1 for r in rollouts if r.truncated),
        obstacle_collision_count=sum(r.obstacle_collisions for r in rollouts),
        boundary_collision_count=sum(r.boundary_collisions for r in rollouts),
        rollouts=rollouts,
        category=category,
    )


def _freeze_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for key, value in weights.items():
        arr = np.array(value, dtype=np.float64, copy=True)
        arr.flags.writeable = False
        frozen[key] = arr
    return frozen


def _weights_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


_MODEL_VERSION = 1


def _metadata_for_result(result: DQNTrainingResult, env_sample) -> dict:
    return {
        "schema_version": _MODEL_VERSION,
        "algorithm": "NumPy DQN",
        "room": "Room5Obstacles",
        "observation_schema": [
            "x_norm", "y_norm", "vx", "vy", "exit_dx_norm", "exit_dy_norm",
            "obstacle_0_visible", "obstacle_0_dx_over_x", "obstacle_0_dy_over_x", "obstacle_0_distance_over_x",
            "obstacle_1_visible", "obstacle_1_dx_over_x", "obstacle_1_dy_over_x", "obstacle_1_distance_over_x",
            "obstacle_2_visible", "obstacle_2_dx_over_x", "obstacle_2_dy_over_x", "obstacle_2_distance_over_x",
            "obstacle_3_visible", "obstacle_3_dx_over_x", "obstacle_3_dy_over_x", "obstacle_3_distance_over_x",
        ],
        "action_schema": [a.name for a in VelocityAction],
        "input_dim": result.input_dim,
        "hidden_units": result.config.hidden_units,
        "action_count": result.action_count,
        "training_seed": result.training_seed,
        "training_config": {
            "episodes": result.config.episodes,
            "learning_rate": result.config.learning_rate,
            "gamma": result.config.gamma,
            "max_steps": result.config.max_steps,
            "epsilon": {
                "kind": result.config.epsilon.kind.value,
                "start": result.config.epsilon.start,
                "minimum": result.config.epsilon.minimum,
                "decay": result.config.epsilon.decay,
                "linear_decay_episodes": result.config.epsilon.linear_decay_episodes,
            },
            "replay_capacity": result.config.replay_capacity,
            "batch_size": result.config.batch_size,
            "warmup_steps": result.config.warmup_steps,
            "target_update_interval": result.config.target_update_interval,
        },
        "environment_config": {
            "room_width_m": env_sample.motion.room_width_m,
            "room_height_m": env_sample.motion.room_height_m,
            "time_step_s": env_sample.motion.time_step_s,
            "exit_center": list(env_sample.motion.exit_center),
            "exit_radius_m": env_sample.motion.exit_radius_m,
            "obstacle_width_m": env_sample.obstacle_config.obstacle_width_m,
            "min_obstacles": env_sample.obstacle_config.min_obstacles,
            "max_obstacles": env_sample.obstacle_config.max_obstacles,
            "observation_distance_m": env_sample.obstacle_config.observation_distance_m,
            "nearest_obstacles": env_sample.obstacle_config.nearest_obstacles,
        },
    }


def save_dqn_model(result: DQNTrainingResult, filepath_stem: str, *, environment_factory: Room5Factory | None = None) -> str:
    # Store numeric weights in NPZ and schema/config metadata in JSON so the
    # app can validate the model before using it.
    dirpath = os.path.dirname(filepath_stem)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    if environment_factory is None:
        from environments.room5_obstacles import Room5Obstacles
        environment_factory = lambda: Room5Obstacles(max_steps=result.config.max_steps)
    env_sample = environment_factory()
    env_sample.reset(seed=result.config.seed, layout_seed=result.config.seed)
    metadata = _metadata_for_result(result, env_sample)

    npz_path = filepath_stem + ".npz"
    json_path = filepath_stem + ".json"
    tmp_npz = npz_path + ".tmp"
    tmp_json = json_path + ".tmp"
    try:
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(f, **dict(result.weights))
        loaded = np.load(tmp_npz, allow_pickle=False)
        loaded.close()
        metadata["weights_sha256"] = _weights_sha256(tmp_npz)
        with open(tmp_json, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
    finally:
        for tmp in (tmp_npz, tmp_json):
            if os.path.exists(tmp):
                os.remove(tmp)
    return json_path


def load_dqn_model(filepath_stem: str) -> tuple[DQNNetwork, dict]:
    json_path = filepath_stem + ".json"
    npz_path = filepath_stem + ".npz"
    with open(json_path) as f:
        metadata = json.load(f)
    if metadata.get("schema_version") != _MODEL_VERSION:
        raise ValueError("Unsupported DQN schema version")
    if metadata.get("algorithm") != "NumPy DQN":
        raise ValueError("Unknown DQN algorithm")
    expected_checksum = metadata.get("weights_sha256")
    if expected_checksum:
        actual = _weights_sha256(npz_path)
        if not hmac.compare_digest(expected_checksum, actual):
            raise ValueError("DQN weights checksum mismatch")
    loaded = np.load(npz_path, allow_pickle=False)
    weights = {key: loaded[key] for key in ("W1", "b1", "W2", "b2")}
    for value in weights.values():
        if not np.all(np.isfinite(value)):
            raise ValueError("Non-finite DQN weights")
    net = DQNNetwork.from_weights(weights)
    if net.input_dim != metadata.get("input_dim"):
        raise ValueError("DQN input dimension mismatch")
    if net.hidden_units != metadata.get("hidden_units"):
        raise ValueError("DQN hidden dimension mismatch")
    if net.action_count != metadata.get("action_count"):
        raise ValueError("DQN action count mismatch")
    return net, metadata


def extract_dqn_action_values(network: DQNNetwork, observation: Room5Observation) -> dict[str, float]:
    q_values = network.predict(observation)
    return {VelocityAction(i).name: float(q_values[i]) for i in range(len(q_values))}
