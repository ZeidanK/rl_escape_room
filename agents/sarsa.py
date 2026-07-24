from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from agents.tabular_utils import (
    default_snapshot_episodes,
    epsilon_for_episode,
    extract_deterministic_greedy_policy,
    freeze_q_table,
    map_signature,
    select_epsilon_greedy_action,
)
from core.types import (
    Action,
    EpsilonDecayKind,
    EpsilonScheduleConfig,
    Position,
    ProgressCallback,
    RolloutResult,
    Room2Factory,
    SarsaConfig,
    SarsaEvaluationSummary,
    SarsaSnapshot,
    SarsaTrainingResult,
    TrajectoryStep,
    TrainingEpisodeMetrics,
)
from environments.grid_environment import GridEnvironment

# Backward-compatible aliases
select_action = select_epsilon_greedy_action
extract_greedy_policy = extract_deterministic_greedy_policy


def rollout_sarsa_policy(
    environment_factory: Room2Factory,
    q_values: Mapping[Position, tuple[float, ...]],
    *,
    seed: int,
    epsilon: float = 0.0,
    max_steps: int | None = None,
) -> RolloutResult:
    env = environment_factory()
    env.reset(seed=seed)
    limit = max_steps if max_steps is not None else env.max_steps
    steps: list[TrajectoryStep] = []
    total_reward = 0.0
    collisions = 0
    slipped_actions = 0
    trap_count = 0
    q_dict = {s: np.array(v, dtype=float) for s, v in q_values.items()}
    rng = np.random.default_rng(seed + 9999)

    for i in range(limit):
        if env.is_done:
            break
        state = env.agent_position
        action = select_action(state, epsilon=epsilon, rng=rng, q_table=q_dict)
        result = env.step(action)
        info = result.info
        slipped = info.get("slipped", False)
        collision = info.get("collision")
        event = info.get("event")
        if event == "trap":
            trap_count += 1
        step = TrajectoryStep(
            index=i,
            state=state,
            requested_action=Action(info.get("requested_action", int(action))),
            effective_action=Action(info.get("effective_action", int(action))),
            reward=result.reward,
            next_state=result.next_state,
            slipped=slipped,
            collision=collision,
            event=event,
            terminated=result.terminated,
            truncated=result.truncated,
        )
        steps.append(step)
        total_reward += result.reward
        if collision:
            collisions += 1
        if slipped:
            slipped_actions += 1

    terminated = env._terminated if hasattr(env, '_terminated') else False
    truncated = env._truncated if hasattr(env, '_truncated') else False
    success = terminated
    return RolloutResult(
        steps=tuple(steps),
        terminated=terminated,
        truncated=truncated,
        success=success,
        total_steps=len(steps),
        total_reward=total_reward,
        collisions=collisions,
        slipped_actions=slipped_actions,
        trap_count=trap_count,
    )


def evaluate_sarsa_policy(
    environment_factory: Room2Factory,
    q_values: Mapping[Position, tuple[float, ...]],
    *,
    n_episodes: int = 100,
    seeds: range | None = None,
    max_steps: int | None = None,
) -> SarsaEvaluationSummary:
    if seeds is None:
        seeds = range(n_episodes)
    rollouts: list[RolloutResult] = []
    for seed in seeds:
        roll = rollout_sarsa_policy(
            environment_factory, q_values,
            seed=seed, epsilon=0.0, max_steps=max_steps,
        )
        rollouts.append(roll)

    n = len(rollouts)
    successes = sum(1 for r in rollouts if r.success)
    truncated = sum(1 for r in rollouts if r.truncated)
    total_rewards = [r.total_reward for r in rollouts]
    total_steps_list = [r.total_steps for r in rollouts]
    mean_return = sum(total_rewards) / n
    std_return = (sum((x - mean_return) ** 2 for x in total_rewards) / n) ** 0.5
    mean_steps = sum(total_steps_list) / n
    successful_steps = [r.total_steps for r in rollouts if r.success]
    mean_successful = (sum(successful_steps) / len(successful_steps)) if successful_steps else None
    total_coll = sum(r.collisions for r in rollouts)
    total_slip = sum(r.slipped_actions for r in rollouts)
    total_traps = sum(r.trap_count for r in rollouts)

    return SarsaEvaluationSummary(
        episodes=n,
        successes=successes,
        success_rate=successes / n,
        mean_return=mean_return,
        std_return=std_return,
        mean_steps=mean_steps,
        mean_successful_steps=mean_successful,
        truncated_episodes=truncated,
        total_collisions=total_coll,
        total_slipped_actions=total_slip,
        total_traps=total_traps,
        rollouts=tuple(rollouts),
    )


class SarsaAgent:
    def __init__(
        self,
        environment_factory: Room2Factory,
        config: SarsaConfig,
    ) -> None:
        self.env_factory = environment_factory
        self.config = config

    def _build_q_table(self) -> dict[Position, np.ndarray]:
        env = self.env_factory()
        n_actions = len(env.actions)
        return {s: np.zeros(n_actions, dtype=float) for s in env.states}

    def _infer_snapshot_episodes(self) -> tuple[int, ...]:
        config = self.config
        if config.snapshot_episodes:
            return tuple(sorted(
                e for e in config.snapshot_episodes if 1 <= e <= config.episodes
            ))
        return default_snapshot_episodes(config.episodes)

    def _take_snapshot(
        self,
        episode: int,
        epsilon: float,
        q_table: dict[Position, np.ndarray],
        snapshot_rng: np.random.Generator,
    ) -> SarsaSnapshot:
        q_copy = freeze_q_table(q_table)
        seed = int(snapshot_rng.integers(0, 2**31))
        roll = rollout_sarsa_policy(
            self.env_factory, q_copy,
            seed=seed, epsilon=0.0,
        )
        return SarsaSnapshot(
            episode=episode,
            epsilon=epsilon,
            q_values=q_copy,
            rollout=roll,
        )

    def update(
        self,
        state: Position,
        action: Action,
        reward: float,
        next_state: Position,
        next_action: Action | None,
        *,
        terminated: bool,
        truncated: bool,
        q_table: dict[Position, np.ndarray],
    ) -> float:
        config = self.config
        q_sa = q_table[state][int(action)]
        if terminated or truncated:
            target = reward
        else:
            q_next = q_table[next_state][int(next_action)]
            target = reward + config.gamma * q_next
        td_error = target - q_sa
        q_table[state][int(action)] += config.alpha * td_error
        return td_error

    def train(
        self,
        progress_callback: ProgressCallback | None = None,
        progress_every: int = 1,
    ) -> SarsaTrainingResult:
        config = self.config

        # RNG streams
        seed_seq = np.random.SeedSequence(config.seed)
        env_ss, policy_ss, snapshot_ss, bookkeeping_ss = seed_seq.spawn(4)
        policy_rng = np.random.Generator(np.random.PCG64(policy_ss))
        snapshot_rng = np.random.Generator(np.random.PCG64(snapshot_ss))
        env_rng = np.random.Generator(np.random.PCG64(env_ss))

        q_table = self._build_q_table()
        snapshot_episodes = set(self._infer_snapshot_episodes())
        metrics: list[TrainingEpisodeMetrics] = []
        snapshots: dict[int, SarsaSnapshot] = {}

        for ep in range(config.episodes):
            epsilon = epsilon_for_episode(ep, config.epsilon)
            env_seed = int(env_rng.integers(0, 2**31))
            env = self.env_factory()
            state = env.reset(seed=env_seed)

            action = select_action(
                state, epsilon=epsilon, rng=policy_rng, q_table=q_table,
            )

            ep_reward = 0.0
            ep_collisions = 0
            ep_slips = 0
            ep_traps = 0
            ep_td_errors: list[float] = []
            step_count = 0

            while True:
                result = env.step(action)
                step_count += 1
                ep_reward += result.reward
                info = result.info
                if info.get("collision"):
                    ep_collisions += 1
                if info.get("slipped"):
                    ep_slips += 1
                if info.get("event") == "trap":
                    ep_traps += 1

                if result.terminated or result.truncated:
                    td = self.update(
                        state, action, result.reward, result.next_state, None,
                        terminated=result.terminated,
                        truncated=result.truncated,
                        q_table=q_table,
                    )
                    ep_td_errors.append(td)
                    break

                next_action = select_action(
                    result.next_state, epsilon=epsilon, rng=policy_rng, q_table=q_table,
                )

                td = self.update(
                    state, action, result.reward, result.next_state, next_action,
                    terminated=False, truncated=False,
                    q_table=q_table,
                )
                ep_td_errors.append(td)

                state = result.next_state
                action = next_action

            mean_td = float(np.mean(ep_td_errors)) if ep_td_errors else 0.0
            max_td = float(np.max(np.abs(ep_td_errors))) if ep_td_errors else 0.0

            m = TrainingEpisodeMetrics(
                episode=ep,
                total_reward=ep_reward,
                steps=step_count,
                success=result.terminated,
                terminated=result.terminated,
                truncated=result.truncated,
                epsilon=epsilon,
                collision_count=ep_collisions,
                slipped_action_count=ep_slips,
                trap_count=ep_traps,
                mean_abs_td_error=mean_td,
                max_abs_td_error=max_td,
            )
            metrics.append(m)

            # One-based snapshot check: after completed episode (ep+1)
            if (ep + 1) in snapshot_episodes:
                snapshots[ep + 1] = self._take_snapshot(
                    ep + 1, epsilon, q_table, snapshot_rng,
                )

            if progress_callback is not None and (ep % progress_every == 0 or ep == config.episodes - 1):
                progress_callback(ep, config.episodes, m)

        public_q = freeze_q_table(q_table)

        return SarsaTrainingResult(
            config=config,
            q_values=public_q,
            metrics=tuple(metrics),
            snapshots=MappingProxyType(dict(snapshots)),
            final_epsilon=epsilon_for_episode(config.episodes - 1, config.epsilon),
            training_seed=config.seed,
        )


# ============================================================
# Model persistence
# ============================================================

_MODEL_VERSION = 1


def _build_metadata(
    result: SarsaTrainingResult,
    reward_config,
    slip_config,
    map_grid,
) -> dict:
    recent_metrics = result.metrics[-100:]
    return {
        "version": _MODEL_VERSION,
        "map_signature": map_signature(map_grid),
        "grid_shape": list(map_grid.shape),
        "n_states": len(result.q_values),
        "n_actions": 4,
        "config": {
            "episodes": result.config.episodes,
            "alpha": result.config.alpha,
            "gamma": result.config.gamma,
            "max_steps": result.config.max_steps,
            "seed": result.config.seed,
            "epsilon": {
                "kind": result.config.epsilon.kind.value,
                "start": result.config.epsilon.start,
                "minimum": result.config.epsilon.minimum,
                "decay": result.config.epsilon.decay,
                "linear_decay_episodes": result.config.epsilon.linear_decay_episodes,
            },
        },
        "rewards": {
            "step_penalty": reward_config.step_penalty,
            "exit_reward": reward_config.exit_reward,
            "wall_penalty": reward_config.wall_penalty,
            "trap_penalty": reward_config.trap_penalty,
            "step_limit_penalty": reward_config.step_limit_penalty,
        },
        "slip_config": {
            "intended_probability": slip_config.intended_probability,
            "left_probability": slip_config.left_probability,
            "right_probability": slip_config.right_probability,
        },
        "training": {
            "final_epsilon": result.final_epsilon,
            "total_episodes": len(result.metrics),
            "final_mean_reward": float(np.mean([m.total_reward for m in recent_metrics])) if recent_metrics else 0.0,
            "final_success_rate": float(np.mean([1.0 if m.success else 0.0 for m in recent_metrics])) if recent_metrics else 0.0,
            "snapshot_episodes": sorted(result.snapshots.keys()),
        },
    }


def _aligned_arrays(
    result: SarsaTrainingResult,
) -> tuple[np.ndarray, np.ndarray]:
    states_list: list[list[int]] = []
    values_list: list[list[float]] = []
    for state, vals in result.q_values.items():
        states_list.append([state[0], state[1]])
        values_list.append(list(vals))
    return np.array(states_list, dtype=np.int32), np.array(values_list, dtype=np.float64)


def save_model(
    result: SarsaTrainingResult,
    filepath_stem: str,
    *,
    reward_config=None,
    slip_config=None,
    map_grid=None,
) -> str:
    import json
    import os

    if reward_config is None:
        from core.types import RewardConfig
        reward_config = RewardConfig()
    if slip_config is None:
        from core.types import SlipConfig
        slip_config = SlipConfig()
    if map_grid is None:
        from environments.room2_sarsa import ROOM2_GRID
        map_grid = ROOM2_GRID

    dirpath = os.path.dirname(filepath_stem)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    metadata = _build_metadata(result, reward_config, slip_config, map_grid)
    states_arr, values_arr = _aligned_arrays(result)

    json_path = filepath_stem + ".json"
    npz_path = filepath_stem + ".npz"

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    np.savez_compressed(npz_path, states=states_arr, values=values_arr)
    return json_path


def _validate_metadata(metadata: dict, map_grid) -> None:
    if metadata.get("version") != _MODEL_VERSION:
        raise ValueError(f"Unsupported model version {metadata.get('version')}; expected {_MODEL_VERSION}")
    loaded_sig = metadata.get("map_signature", "")
    current_sig = map_signature(map_grid)
    if loaded_sig != current_sig:
        raise ValueError(
            f"Map signature mismatch: loaded={loaded_sig}, current={current_sig}. "
            "The model was trained on a different map."
        )
    expected_states = metadata.get("n_states")
    expected_actions = metadata.get("n_actions", 4)
    if expected_actions != 4:
        raise ValueError(f"Expected 4 actions; got {expected_actions}")


def load_model(
    filepath_stem: str,
    *,
    map_grid=None,
) -> tuple[Mapping[Position, tuple[float, ...]], dict]:
    import json

    if map_grid is None:
        from environments.room2_sarsa import ROOM2_GRID
        map_grid = ROOM2_GRID

    json_path = filepath_stem + ".json"
    npz_path = filepath_stem + ".npz"

    with open(json_path) as f:
        metadata = json.load(f)
    _validate_metadata(metadata, map_grid)

    loaded = np.load(npz_path)
    states_arr: np.ndarray = loaded["states"]
    values_arr: np.ndarray = loaded["values"]

    if values_arr.shape[1] != 4:
        raise ValueError(f"Expected 4 action values; got shape {values_arr.shape}")

    q_values: dict[Position, tuple[float, ...]] = {}
    for i in range(states_arr.shape[0]):
        state = (int(states_arr[i, 0]), int(states_arr[i, 1]))
        vals = tuple(float(v) for v in values_arr[i])
        if not all(np.isfinite(vals)):
            raise ValueError(f"Non-finite Q-values at state {state}")
        q_values[state] = vals

    if len(q_values) != metadata["n_states"]:
        raise ValueError(
            f"Expected {metadata['n_states']} states; loaded {len(q_values)}"
        )

    return MappingProxyType(q_values), metadata
