# Reinforcement Learning Escape Room

A Python-based reinforcement learning escape-room application with four rooms, each using a different RL method.

## Rooms

| Room | Name | Algorithm | Environment |
|------|------|-----------|-------------|
| 1 | Ice Maze | Value Iteration (DP) | Known 10×10 stochastic grid |
| 2 | Laser Corridor | SARSA | Unknown 10×10 stochastic grid |
| 3 | Key Vault | Q-Learning | Unknown 10×10 grid with key mechanic |
| 4 | Momentum Chamber | Semi-gradient SARSA + tile coding | Continuous 10×10 metre room |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
rl_escape_room/
├── app.py                 # Streamlit entry point
├── config/                # Configuration dataclasses and room specs
├── core/                  # Shared types (RewardConfig, RoomSpec, enums)
├── environments/          # Room environments
├── agents/                # RL algorithm stubs (implemented Phase 2+)
├── training/              # Training pipeline stubs (implemented Phase 2+)
├── visualization/         # Renderer stubs (implemented Phase 2+)
├── storage/               # Saved models, metrics, episodes
├── tests/                 # Smoke tests
└── docs/                  # Design documents
```

## Rewards

All reward values are configurable via `RewardConfig`. Defaults:

| Event | Default |
|-------|--------:|
| Normal step | -1.0 |
| Reach exit | +100.0 |
| Hit wall/boundary | -3.0 |
| Enter trap | -20.0 |
| Collect key | +10.0 |
| Attempt locked exit (no key) | -5.0 |
| Time bonus scale | 0.0 (disabled) |

Faster completion is incentivised through negative step costs. An optional time bonus is available but disabled by default.

## Design Documents

- `docs/ROOM_SPECS.md` — Detailed specifications for each room.
- `docs/DECISIONS.md` — Design decisions and rationale.
- `PHASE_1_AGENT_BRIEF.md` — Original Phase 1 requirements.

## Team

| Member | Primary Responsibility |
|--------|----------------------|
| Member A | Shared grid, Dynamic Programming, Testing |
| Member B | SARSA, Q-Learning, Experiments |
| Member C | Room 4, Streamlit, Deployment |
| Everyone | Documentation, Defense |
