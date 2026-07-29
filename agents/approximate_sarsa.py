"""Approximate SARSA implementation for the continuous Room 4 task."""

import hashlib
import hmac
import json
import math
import os
from types import MappingProxyType

import numpy as np

from collections.abc import Mapping

from core.types import (
    ApproximateEpisodeMetrics,
    ApproximateEvaluationSummary,
    ApproximateSarsaConfig,
    ApproximateSarsaSnapshot,
    ApproximateSarsaTrainingResult,
    ApproxProgressCallback,
    ContinuousRolloutResult,
    ContinuousState,
    ContinuousTrajectoryStep,
    FIXED_UNSEEN_STARTS,
    Room4Factory,
    StartMode,
    VelocityAction,
)
from features.tile_coding import TileCoder, TileCodingConfig
from agents.tabular_utils import epsilon_for_episode, default_snapshot_episodes


# Room 4 cannot store a table for every possible continuous state, so this
# module uses tile coding plus a linear action-value function.
def _select_continuous_action(
    action_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    actions: list[VelocityAction],
) -> VelocityAction:
    if rng.random() < epsilon:
        return VelocityAction(int(rng.integers(0, len(actions))))
    max_val = np.max(action_values)
    tied = np.where(np.abs(action_values - max_val) < 1e-12)[0]
    return VelocityAction(int(rng.choice(tied)))


class LinearTileQFunction:
    # Linear Q(s,a): sum the weights for the active tile-coded features of
    # state s in the row for action a.
    def __init__(
        self,
        tile_coder: TileCoder,
        n_actions: int = 9,
    ) -> None:
        self.tile_coder = tile_coder
        self.n_actions = n_actions
        nf = tile_coder.feature_count
        self._weights = np.zeros((n_actions, nf), dtype=np.float64)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    def features(self, state: ContinuousState) -> tuple[int, ...]:
        return self.tile_coder.active_features(state)

    def value(self, state: ContinuousState, action: VelocityAction) -> float:
        return float(self._weights[int(action), self.features(state)].sum())

    def value_at(self, feats: tuple[int, ...], action: int) -> float:
        return float(self._weights[action, feats].sum())

    def action_values(self, state: ContinuousState) -> np.ndarray:
        feats = self.tile_coder.active_features(state)
        return self._weights[:, feats].sum(axis=1)

    def action_values_at(self, feats: tuple[int, ...]) -> np.ndarray:
        return self._weights[:, feats].sum(axis=1)

    def update(self, state: ContinuousState, action: VelocityAction, scaled_td_error: float) -> None:
        feats = self.tile_coder.active_features(state)
        self._weights[int(action), feats] += scaled_td_error

    def update_at(self, feats: tuple[int, ...], action: int, scaled_td_error: float) -> None:
        # Semi-gradient update: only active feature weights for the selected
        # action are adjusted.
        self._weights[action, feats] += scaled_td_error


def rollout_approximate_policy(
    environment_factory: Room4Factory,
    q_function: LinearTileQFunction,
    *,
    seed: int,
    epsilon: float = 0.0,
    max_steps: int | None = None,
    start_state: ContinuousState | None = None,
) -> ContinuousRolloutResult:
    # Runs a trained linear Q-function in the continuous environment and keeps
    # the full trajectory for visualization.
    env = environment_factory()
    start_state = env.reset(seed=seed, start_state=start_state)
    limit = max_steps if max_steps is not None else env._max_steps
    rng = np.random.default_rng(seed + 9999)
    actions_list = list(VelocityAction)

    steps: list[ContinuousTrajectoryStep] = []
    total_reward = 0.0
    coll_count = 0
    state = start_state

    for i in range(limit):
        if env.is_done:
            break
        av = q_function.action_values(state)
        action = _select_continuous_action(av, epsilon, rng, actions_list) if epsilon > 0 else _greedy_continuous_action(av)
        result = env.step(int(action))
        info = result.info
        collision = info.get("collision")
        event = info.get("event")
        if collision:
            coll_count += 1
        dist = math.sqrt(
            (state[0] - env.motion.exit_center[0]) ** 2
            + (state[1] - env.motion.exit_center[1]) ** 2
        )
        step = ContinuousTrajectoryStep(
            index=i,
            state=state,
            requested_action=action,
            reward=result.reward,
            next_state=result.next_state,
            collision=collision,
            event=event,
            terminated=result.terminated,
            truncated=result.truncated,
            distance_to_exit_m=dist,
        )
        steps.append(step)
        total_reward += result.reward
        state = result.next_state

    terminated = env.is_done and env._terminated
    truncated = env.is_done and env._truncated
    success = terminated

    return ContinuousRolloutResult(
        seed=seed,
        start_state=start_state,
        final_state=state,
        total_reward=total_reward,
        steps=len(steps),
        simulated_time_s=len(steps) * env.motion.time_step_s,
        success=success,
        terminated=terminated,
        truncated=truncated,
        collision_count=coll_count,
        distance_travelled_m=env._distance_travelled if hasattr(env, '_distance_travelled') else 0.0,
        trajectory=tuple(steps),
    )


def _greedy_continuous_action(action_values: np.ndarray) -> VelocityAction:
    max_val = np.max(action_values)
    tied = np.where(np.abs(action_values - max_val) < 1e-12)[0]
    return VelocityAction(int(tied[0]))


class ApproximateSarsaAgent:
    # Training loop for semi-gradient SARSA.  It follows the same on-policy
    # idea as tabular SARSA, but updates linear weights instead of a Q-table.
    def __init__(
        self,
        environment_factory: Room4Factory,
        config: ApproximateSarsaConfig,
        tile_coder: TileCoder | None = None,
    ) -> None:
        self.env_factory = environment_factory
        self.config = config
        self.actions_list = list(VelocityAction)

    def _make_tile_coder(self) -> TileCoder:
        # Build the tile coder from the environment size so feature indexes are
        # aligned with the actual 10x10m room.
        env = self.env_factory()
        tc = TileCoder(
            self.config.tile_coding,
            room_width=env.motion.room_width_m,
            room_height=env.motion.room_height_m,
        )
        return tc

    def _infer_snapshot_episodes(self) -> tuple[int, ...]:
        config = self.config
        if config.snapshot_episodes:
            return tuple(sorted(
                e for e in config.snapshot_episodes if 1 <= e <= config.episodes
            ))
        return default_snapshot_episodes(config.episodes)

    def train(
        self,
        progress_callback: ApproxProgressCallback | None = None,
        progress_every: int = 1,
    ) -> ApproximateSarsaTrainingResult:
        config = self.config
        # Separate RNG streams make training reproducible while preventing
        # exploration, environment starts, and snapshots from influencing each
        # other through shared random draws.
        seed_seq = np.random.SeedSequence(config.seed)
        env_ss, policy_ss, snapshot_ss, bookkeeping_ss = seed_seq.spawn(4)
        policy_rng = np.random.Generator(np.random.PCG64(policy_ss))
        snapshot_rng = np.random.Generator(np.random.PCG64(snapshot_ss))
        env_rng = np.random.Generator(np.random.PCG64(env_ss))
        bookkeeping_rng = np.random.Generator(np.random.PCG64(bookkeeping_ss))

        tile_coder = self._make_tile_coder()
        q_func = LinearTileQFunction(tile_coder, n_actions=9)

        snapshot_episodes = set(self._infer_snapshot_episodes())
        metrics: list[ApproximateEpisodeMetrics] = []
        snapshots: dict[int, ApproximateSarsaSnapshot] = {}

        for ep in range(config.episodes):
            epsilon = epsilon_for_episode(ep, config.epsilon)
            env_seed = int(env_rng.integers(0, 2**31))
            env = self.env_factory()
            state = env.reset(seed=env_seed)
            state_feats = q_func.features(state)
            av = q_func.action_values_at(state_feats)
            action = _select_continuous_action(av, epsilon, policy_rng, self.actions_list)
            action_int = int(action)

            ep_reward = 0.0
            ep_collisions = 0
            ep_td_errors: list[float] = []
            step_count = 0
            start_pos = state
            norm = config.alpha / config.tile_coding.num_tilings

            while True:
                result = env.step(action_int)
                step_count += 1
                ep_reward += result.reward
                if result.info.get("collision"):
                    ep_collisions += 1

                nxt = result.next_state
                nxt_feats = q_func.features(nxt)

                if result.terminated or result.truncated:
                    # No bootstrap term after terminal/truncated transitions.
                    td_error = result.reward - q_func.value_at(state_feats, action_int)
                    q_func.update_at(state_feats, action_int, norm * td_error)
                    ep_td_errors.append(td_error)
                    break

                av_next = q_func.action_values_at(nxt_feats)
                next_action = _select_continuous_action(av_next, epsilon, policy_rng, self.actions_list)
                next_action_int = int(next_action)
                # On-policy semi-gradient SARSA target:
                # r + gamma * Q(next_state, next_action).
                target = result.reward + config.gamma * q_func.value_at(nxt_feats, next_action_int)
                td_error = target - q_func.value_at(state_feats, action_int)
                q_func.update_at(state_feats, action_int, norm * td_error)
                ep_td_errors.append(td_error)

                state = nxt
                state_feats = nxt_feats
                action = next_action
                action_int = next_action_int

            final_dist = math.sqrt(
                (state[0] - env.motion.exit_center[0]) ** 2
                + (state[1] - env.motion.exit_center[1]) ** 2
            )
            mean_td = float(np.mean(ep_td_errors)) if ep_td_errors else 0.0
            max_td = float(np.max(np.abs(ep_td_errors))) if ep_td_errors else 0.0

            m = ApproximateEpisodeMetrics(
                episode=ep,
                total_reward=ep_reward,
                steps=step_count,
                simulated_time_s=step_count * env.motion.time_step_s,
                success=result.terminated,
                terminated=result.terminated,
                truncated=result.truncated,
                epsilon=epsilon,
                collision_count=ep_collisions,
                distance_travelled_m=env._distance_travelled if hasattr(env, '_distance_travelled') else 0.0,
                final_distance_to_exit_m=final_dist,
                mean_abs_td_error=mean_td,
                max_abs_td_error=max_td,
            )
            metrics.append(m)

            if (ep + 1) in snapshot_episodes:
                snap_epsilon = epsilon
                snap_weights = q_func.weights
                snap_weights.flags.writeable = False
                snap_rng_seed = int(snapshot_rng.integers(0, 2**31))
                env_snap = self.env_factory()
                snap_start = env_snap.reset(seed=snap_rng_seed)
                snap_q = LinearTileQFunction(tile_coder, n_actions=9)
                snap_q._weights = snap_weights.copy()
                snap_steps: list[ContinuousTrajectoryStep] = []
                snap_total_reward = 0.0
                snap_collisions = 0
                snap_state = snap_start
                for i in range(config.max_steps):
                    if env_snap.is_done:
                        break
                    sav = snap_q.action_values(snap_state)
                    saction = _greedy_continuous_action(sav)
                    sresult = env_snap.step(int(saction))
                    snap_total_reward += sresult.reward
                    if sresult.info.get("collision"):
                        snap_collisions += 1
                    sdist = math.sqrt(
                        (snap_state[0] - env_snap.motion.exit_center[0]) ** 2
                        + (snap_state[1] - env_snap.motion.exit_center[1]) ** 2
                    )
                    snap_steps.append(ContinuousTrajectoryStep(
                        index=i, state=snap_state, requested_action=saction,
                        reward=sresult.reward, next_state=sresult.next_state,
                        collision=sresult.info.get("collision"), event=sresult.info.get("event"),
                        terminated=sresult.terminated, truncated=sresult.truncated,
                        distance_to_exit_m=sdist,
                    ))
                    snap_state = sresult.next_state
                snapshots[ep + 1] = ApproximateSarsaSnapshot(
                    episode=ep + 1,
                    epsilon=snap_epsilon,
                    weights=snap_weights,
                    rollout=ContinuousRolloutResult(
                        seed=snap_rng_seed,
                        start_state=snap_start,
                        final_state=snap_state,
                        total_reward=snap_total_reward,
                        steps=len(snap_steps),
                        simulated_time_s=len(snap_steps) * env_snap.motion.time_step_s,
                        success=env_snap._terminated,
                        terminated=env_snap._terminated,
                        truncated=env_snap._truncated,
                        collision_count=snap_collisions,
                        distance_travelled_m=env_snap._distance_travelled if hasattr(env_snap, '_distance_travelled') else 0.0,
                        trajectory=tuple(snap_steps),
                    ),
                )

            if progress_callback is not None and (ep % progress_every == 0 or ep == config.episodes - 1):
                progress_callback(ep, config.episodes, m)

        final_weights = q_func.weights
        final_weights.flags.writeable = False

        return ApproximateSarsaTrainingResult(
            config=config,
            weights=final_weights,
            metrics=tuple(metrics),
            snapshots=MappingProxyType(dict(snapshots)),
            final_epsilon=epsilon_for_episode(config.episodes - 1, config.epsilon),
            training_seed=config.seed,
        )


def evaluate_approximate_policy(
    environment_factory: Room4Factory,
    weights: np.ndarray,
    tile_coding_config: TileCodingConfig,
    motion_config,
    *,
    n_episodes: int = 50,
    seeds: range | None = None,
    start_mode: StartMode = StartMode.FIXED,
    max_steps: int | None = None,
) -> ApproximateEvaluationSummary:
    if seeds is None:
        seeds = range(n_episodes)
    env_sample = environment_factory()
    tile_coder = TileCoder(tile_coding_config, room_width=env_sample.motion.room_width_m, room_height=env_sample.motion.room_height_m)
    q_func = LinearTileQFunction(tile_coder, n_actions=9)
    q_func._weights = weights.copy()

    rollout_factory = _make_category_factory(environment_factory, start_mode)
    rollouts: list[ContinuousRolloutResult] = []
    for seed_val in seeds:
        roll = rollout_approximate_policy(
            rollout_factory, q_func,
            seed=int(seed_val), epsilon=0.0, max_steps=max_steps,
        )
        rollouts.append(roll)

    n = len(rollouts)
    successes = sum(1 for r in rollouts if r.success)
    truncated = sum(1 for r in rollouts if r.truncated)
    returns = [r.total_reward for r in rollouts]
    steps_list = [r.steps for r in rollouts]
    mean_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    mean_steps = float(np.mean(steps_list)) if steps_list else 0.0
    successful_steps = [r.steps for r in rollouts if r.success]
    mean_successful = float(np.mean(successful_steps)) if successful_steps else None
    total_coll = sum(r.collision_count for r in rollouts)
    dists = [r.distance_travelled_m for r in rollouts]

    return ApproximateEvaluationSummary(
        n_episodes=n,
        successes=successes,
        success_rate=successes / n if n > 0 else 0.0,
        mean_return=mean_return,
        std_return=std_return,
        mean_steps=mean_steps,
        mean_successful_steps=mean_successful,
        truncated_count=truncated,
        total_collisions=total_coll,
        mean_distance_travelled_m=float(np.mean(dists)) if dists else 0.0,
        total_distance_travelled_m=float(np.sum(dists)) if dists else 0.0,
        rollouts=tuple(rollouts),
        start_category=start_mode.value,
    )


def evaluate_approximate_policy_all_categories(
    environment_factory: Room4Factory,
    weights: np.ndarray,
    tile_coding_config: TileCodingConfig,
    motion_config,
    *,
    n_episodes: int = 25,
    max_steps: int | None = None,
) -> Mapping[str, ApproximateEvaluationSummary]:
    # Evaluates the same learned weights across fixed and random start
    # categories to show whether tile coding generalized beyond training.
    env_sample = environment_factory()
    tile_coder = TileCoder(tile_coding_config, room_width=env_sample.motion.room_width_m, room_height=env_sample.motion.room_height_m)
    q_func = LinearTileQFunction(tile_coder, n_actions=9)
    q_func._weights = weights.copy()

    results: dict[str, ApproximateEvaluationSummary] = {}

    # Fixed training start
    fixed_factory = _make_category_factory(environment_factory, StartMode.FIXED)
    results["fixed_training_start"] = evaluate_approximate_policy(
        fixed_factory, weights, tile_coding_config, motion_config,
        n_episodes=n_episodes, start_mode=StartMode.FIXED, max_steps=max_steps,
    )

    # Fixed unseen starts — per-start and aggregate
    unseen_per_start = n_episodes // max(1, len(FIXED_UNSEEN_STARTS))
    all_unseen_rollouts: list[ContinuousRolloutResult] = []
    for start in FIXED_UNSEEN_STARTS:
        start_rollouts: list[ContinuousRolloutResult] = []
        for i in range(unseen_per_start):
            seed_val = i * 1000 + hash(start) % (2 ** 16)
            roll = rollout_approximate_policy(
                environment_factory, q_func,
                seed=seed_val, epsilon=0.0, max_steps=max_steps,
                start_state=start,
            )
            start_rollouts.append(roll)
        all_unseen_rollouts.extend(start_rollouts)
        label = f"unseen_start_{start[0]}_{start[1]}_{start[2]}_{start[3]}".replace(".", "_")
        results[label] = _summarize_rollouts(
            tuple(start_rollouts), start_category="fixed_unseen_starts",
        )

    results["fixed_unseen_starts"] = _summarize_rollouts(
        tuple(all_unseen_rollouts), start_category="fixed_unseen_starts",
    )

    # Random lower left
    ll_factory = _make_category_factory(environment_factory, StartMode.RANDOM_LOWER_LEFT)
    results["random_lower_left"] = evaluate_approximate_policy(
        ll_factory, weights, tile_coding_config, motion_config,
        n_episodes=n_episodes, start_mode=StartMode.RANDOM_LOWER_LEFT, max_steps=max_steps,
    )

    # Random room
    rr_factory = _make_category_factory(environment_factory, StartMode.RANDOM_ROOM)
    results["random_room"] = evaluate_approximate_policy(
        rr_factory, weights, tile_coding_config, motion_config,
        n_episodes=n_episodes, start_mode=StartMode.RANDOM_ROOM, max_steps=max_steps,
    )

    return results


def _make_category_factory(
    base_factory: Room4Factory,
    start_mode: StartMode,
) -> Room4Factory:
    def factory():
        env = base_factory()
        env._start_mode = start_mode
        return env
    return factory


def _summarize_rollouts(
    rollouts: tuple[ContinuousRolloutResult, ...],
    start_category: str = "",
) -> ApproximateEvaluationSummary:
    n = len(rollouts)
    if n == 0:
        return ApproximateEvaluationSummary(
            n_episodes=0, successes=0, success_rate=0.0,
            mean_return=0.0, std_return=0.0, mean_steps=0.0,
            mean_successful_steps=None, truncated_count=0,
            total_collisions=0, mean_distance_travelled_m=0.0,
            total_distance_travelled_m=0.0, rollouts=(),
            start_category=start_category,
        )
    successes = sum(1 for r in rollouts if r.success)
    truncated = sum(1 for r in rollouts if r.truncated)
    returns = [r.total_reward for r in rollouts]
    steps_list = [r.steps for r in rollouts]
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    mean_steps = float(np.mean(steps_list))
    successful_steps = [r.steps for r in rollouts if r.success]
    mean_successful = float(np.mean(successful_steps)) if successful_steps else None
    total_coll = sum(r.collision_count for r in rollouts)
    dists = [r.distance_travelled_m for r in rollouts]
    return ApproximateEvaluationSummary(
        n_episodes=n,
        successes=successes,
        success_rate=successes / n,
        mean_return=mean_return,
        std_return=std_return,
        mean_steps=mean_steps,
        mean_successful_steps=mean_successful,
        truncated_count=truncated,
        total_collisions=total_coll,
        mean_distance_travelled_m=float(np.mean(dists)),
        total_distance_travelled_m=float(np.sum(dists)),
        rollouts=rollouts,
        start_category=start_category,
    )


_MODEL_VERSION = 1


def _build_metadata(
    result: ApproximateSarsaTrainingResult,
    tile_coding_config: TileCodingConfig,
    motion_config,
    reward_config,
) -> dict:
    return {
        "schema_version": _MODEL_VERSION,
        "algorithm": "Semi-gradient SARSA with tile coding",
        "room": "Room4Continuous",
        "state_schema": ["X", "Y", "Vx", "Vy"],
        "action_schema": [a.name for a in VelocityAction],
        "motion_config": {
            "room_width_m": motion_config.room_width_m,
            "room_height_m": motion_config.room_height_m,
            "time_step_s": motion_config.time_step_s,
            "exit_center": list(motion_config.exit_center),
            "exit_radius_m": motion_config.exit_radius_m,
        },
        "reward_config": {
            "step": reward_config.step,
            "exit": reward_config.exit,
            "boundary_collision": reward_config.boundary_collision,
            "timeout": reward_config.timeout,
            "distance_progress_scale": reward_config.distance_progress_scale,
        },
        "tile_coding_config": {
            "num_tilings": tile_coding_config.num_tilings,
            "tiles_x": tile_coding_config.tiles_x,
            "tiles_y": tile_coding_config.tiles_y,
            "include_velocity": tile_coding_config.include_velocity,
        },
        "training_config": {
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
            "start_mode": result.config.start_mode.value,
        },
        "feature_count": result.weights.shape[1],
        "action_count": result.weights.shape[0],
        "training_seed": result.config.seed,
    }


def _sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_approximate_model(
    result: ApproximateSarsaTrainingResult,
    filepath_stem: str,
    *,
    tile_coding_config: TileCodingConfig | None = None,
    motion_config=None,
    reward_config=None,
) -> str:
    # Save weights atomically and include a checksum, because these artifacts
    # are loaded by the Streamlit showcase without retraining.
    if tile_coding_config is None:
        tile_coding_config = result.config.tile_coding
    if motion_config is None:
        from environments.room4_continuous import Room4Continuous
        motion_config = Room4Continuous().motion
    if reward_config is None:
        from environments.room4_continuous import Room4Continuous
        reward_config = Room4Continuous().rewards

    dirpath = os.path.dirname(filepath_stem)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    metadata = _build_metadata(result, tile_coding_config, motion_config, reward_config)

    json_path = filepath_stem + ".json"
    npz_path = filepath_stem + ".npz"
    tmp_npz = npz_path + ".tmp"
    tmp_json = json_path + ".tmp"

    try:
        # Write NPZ first
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(f, weights=result.weights)

        # Validate the NPZ can be loaded with allow_pickle=False
        test = np.load(tmp_npz, allow_pickle=False)
        test.close()

        # Compute SHA-256 of the NPZ
        metadata["weights_sha256"] = _sha256_file(tmp_npz)

        # Write metadata JSON
        with open(tmp_json, "w") as f:
            json.dump(metadata, f, indent=2)

        # Atomic replace: weights first, metadata last
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
    finally:
        for tmp in (tmp_npz, tmp_json):
            if os.path.exists(tmp):
                os.remove(tmp)

    return json_path


def load_approximate_model(
    filepath_stem: str,
    *,
    expected_tile_coding: TileCodingConfig | None = None,
    expected_action_count: int = 9,
) -> tuple[np.ndarray, dict]:
    json_path = filepath_stem + ".json"
    npz_path = filepath_stem + ".npz"

    with open(json_path) as f:
        metadata = json.load(f)

    if metadata.get("schema_version") != _MODEL_VERSION:
        raise ValueError(f"Unsupported schema version {metadata.get('schema_version')}; expected {_MODEL_VERSION}")
    if metadata.get("algorithm") != "Semi-gradient SARSA with tile coding":
        raise ValueError(f"Unknown algorithm: {metadata.get('algorithm')}")
    if metadata.get("action_count") != expected_action_count:
        raise ValueError(f"Expected {expected_action_count} actions; got {metadata.get('action_count')}")
    if metadata.get("action_schema") != [a.name for a in VelocityAction]:
        raise ValueError("Action schema mismatch")

    if expected_tile_coding is not None:
        tc = metadata.get("tile_coding_config", {})
        if tc.get("num_tilings") != expected_tile_coding.num_tilings:
            raise ValueError("Tile coding mismatch: num_tilings")
        if tc.get("tiles_x") != expected_tile_coding.tiles_x:
            raise ValueError("Tile coding mismatch: tiles_x")
        if tc.get("tiles_y") != expected_tile_coding.tiles_y:
            raise ValueError("Tile coding mismatch: tiles_y")

    # Checksum verification
    expected_checksum = metadata.get("weights_sha256")
    if expected_checksum:
        actual_checksum = _sha256_file(npz_path)
        if not hmac.compare_digest(expected_checksum, actual_checksum):
            raise ValueError(
                f"Weights checksum mismatch: expected {expected_checksum}, "
                f"got {actual_checksum}"
            )

    loaded = np.load(npz_path, allow_pickle=False)
    weights = loaded["weights"]

    expected_feature_count = metadata.get("feature_count")
    if expected_feature_count is not None and weights.shape[1] != expected_feature_count:
        raise ValueError(f"Expected feature count {expected_feature_count}; got {weights.shape[1]}")
    if weights.shape[0] != expected_action_count:
        raise ValueError(f"Expected {expected_action_count} action rows; got {weights.shape[0]}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite weights in loaded model")

    return weights, metadata
