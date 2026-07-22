from types import MappingProxyType

import numpy as np

from core.types import (
    ACTION_DELTAS,
    Action,
    PolicyEvaluationSummary,
    Position,
    RolloutResult,
    TrajectoryStep,
    ValueIterationConfig,
    ValueIterationResult,
)
from environments.grid_environment import KnownModelGridEnvironment


class ValueIterationAgent:
    def __init__(
        self,
        environment: KnownModelGridEnvironment,
        config: ValueIterationConfig | None = None,
    ):
        self.env = environment
        self.config = config or ValueIterationConfig()

    def solve(self) -> ValueIterationResult:
        config = self.config
        env = self.env
        states = env.states
        actions = env.actions
        goal_states = {s for s in states if env.is_terminal_state(s)}

        values: dict[Position, float] = {s: 0.0 for s in states}
        delta_history: list[float] = []

        for iteration in range(config.max_iterations):
            new_values: dict[Position, float] = {}
            max_delta = 0.0
            for s in states:
                if s in goal_states:
                    new_values[s] = 0.0
                    continue
                best_value: float | None = None
                for a in actions:
                    q = self.calculate_action_value(s, a, values)
                    if best_value is None or q > best_value:
                        best_value = q
                new_values[s] = best_value if best_value is not None else 0.0
                delta = abs(new_values[s] - values[s])
                if delta > max_delta:
                    max_delta = delta
            delta_history.append(max_delta)
            values = new_values
            if max_delta < config.tolerance:
                policy = self.extract_policy(values)
                return ValueIterationResult(
                    values=MappingProxyType(dict(values)),
                    policy=MappingProxyType(dict(policy)),
                    iterations=iteration + 1,
                    converged=True,
                    final_delta=max_delta,
                    delta_history=tuple(delta_history),
                    start_state_value=values[env.start_position],
                )

        policy = self.extract_policy(values)
        return ValueIterationResult(
            values=MappingProxyType(dict(values)),
            policy=MappingProxyType(dict(policy)),
            iterations=config.max_iterations,
            converged=False,
            final_delta=delta_history[-1],
            delta_history=tuple(delta_history),
            start_state_value=values[env.start_position],
        )

    def calculate_action_value(
        self,
        state: Position,
        action: Action,
        values: dict[Position, float],
    ) -> float:
        env = self.env
        config = self.config
        outcomes = env.get_transition_distribution(state, action)
        q = 0.0
        for outcome in outcomes:
            if outcome.terminated:
                contribution = outcome.reward
            else:
                contribution = outcome.reward + config.gamma * values[outcome.next_state]
            q += outcome.probability * contribution
        return q

    def extract_policy(
        self,
        values: dict[Position, float],
    ) -> dict[Position, Action | None]:
        env = self.env
        actions = env.actions
        policy: dict[Position, Action | None] = {}
        for s in env.states:
            if env.is_terminal_state(s):
                policy[s] = None
                continue
            best_action: Action | None = None
            best_value: float | None = None
            for a in actions:
                q = self.calculate_action_value(s, a, values)
                if best_value is None or q > best_value + self.config.tie_tolerance:
                    best_value = q
                    best_action = a
            policy[s] = best_action
        return policy


def rollout_policy(
    env: KnownModelGridEnvironment,
    policy: MappingProxyType | dict,
    *,
    seed: int,
    max_steps: int | None = None,
) -> RolloutResult:
    env.reset(seed=seed)
    limit = max_steps if max_steps is not None else env.max_steps
    steps: list[TrajectoryStep] = []
    total_reward = 0.0
    collisions = 0
    slipped_actions = 0
    for i in range(limit):
        if env.is_done:
            break
        state = env.agent_position
        action = policy.get(state)
        if action is None:
            break
        result = env.step(action)
        info = result.info
        slipped = info.get("slipped", False)
        collision = info.get("collision")
        event = info.get("event")
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
    # After loop, determine terminal status
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
    )


def _run_single_evaluation(
    env: KnownModelGridEnvironment,
    policy: MappingProxyType | dict,
    *,
    seed: int,
    max_steps: int | None = None,
) -> RolloutResult:
    return rollout_policy(env, policy, seed=seed, max_steps=max_steps)


def evaluate_policy(
    env: KnownModelGridEnvironment,
    policy: MappingProxyType | dict,
    *,
    n_episodes: int = 100,
    seeds: range | None = None,
) -> PolicyEvaluationSummary:
    if seeds is None:
        seeds = range(n_episodes)
    results: list[RolloutResult] = []
    trajectories: list[tuple[Position, ...]] = []
    for seed in seeds:
        # Use a fresh env instance for each episode
        import copy
        env_copy = copy.deepcopy(env)
        result = _run_single_evaluation(env_copy, policy, seed=seed)
        results.append(result)
        traj = tuple(s.state for s in result.steps)
        if result.success:
            traj = traj + (env_copy.goal_position,)
        trajectories.append(traj)

    successes = sum(1 for r in results if r.success)
    total_steps_list = [r.total_steps for r in results]
    total_reward_list = [r.total_reward for r in results]
    n = len(results)
    mean_return = sum(total_reward_list) / n
    std_return = (sum((x - mean_return) ** 2 for x in total_reward_list) / n) ** 0.5
    mean_steps = sum(total_steps_list) / n
    std_steps = (sum((x - mean_steps) ** 2 for x in total_steps_list) / n) ** 0.5
    successful_steps = [r.total_steps for r in results if r.success]
    mean_successful_steps = (sum(successful_steps) / len(successful_steps)) if successful_steps else None
    min_s = min(total_steps_list) if total_steps_list else None
    max_s = max(total_steps_list) if total_steps_list else None
    total_coll = sum(r.collisions for r in results)
    total_slip = sum(r.slipped_actions for r in results)

    return PolicyEvaluationSummary(
        episodes=n,
        successes=successes,
        success_rate=successes / n,
        mean_return=mean_return,
        std_return=std_return,
        mean_steps=mean_steps,
        std_steps=std_steps,
        min_steps=min_s,
        max_steps=max_s,
        mean_successful_steps=mean_successful_steps,
        total_collisions=total_coll,
        total_slipped=total_slip,
        trajectories=tuple(trajectories),
    )
