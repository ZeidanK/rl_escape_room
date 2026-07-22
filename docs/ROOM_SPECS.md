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

### Default Grid
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

### Rewards
| Event | Formula | Default |
|-------|---------|--------:|
| Normal step | `step_penalty` | -1.0 |
| Reach exit | `step_penalty + exit_reward` | +99.0 |
| Hit wall/boundary | `step_penalty + wall_penalty` | -4.0 |
| Enter trap | `step_penalty + trap_penalty` | -21.0 |

---

## Room 3 — Key Vault (Q-Learning)

| Property | Value |
|----------|-------|
| Algorithm | Q-Learning (off-policy TD control) |
| Grid | 10×10, unknown environment |
| State | `(row, column, has_key)` — 200 discrete states |
| Actions | `UP(0), RIGHT(1), DOWN(2), LEFT(3)` |
| Special mechanic | Key must be collected before exit becomes available |
| Terminal condition | Reaching the exit cell while `has_key == True` |
| Non-terminal | Reaching the exit cell without the key (episode continues) |

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

### Rewards
| Event | Formula | Default |
|-------|---------|--------:|
| Normal step | `step_penalty` | -1.0 |
| Reach exit (with key) | `step_penalty + exit_reward` | +99.0 |
| Hit wall/boundary | `step_penalty + wall_penalty` | -4.0 |
| Enter trap | `step_penalty + trap_penalty` | -21.0 |
| Collect key | `step_penalty + key_reward` | +9.0 |
| Attempt locked exit (no key) | `step_penalty + locked_exit_penalty` | -6.0 |

---

## Room 4 — Momentum Chamber (Function Approximation)

| Property | Value |
|----------|-------|
| Algorithm | Semi-gradient SARSA with tile coding |
| Environment | Continuous 10×10 metre room |
| State | `(X, Y, Vx, Vy)` — 4 continuous dimensions |
| Actions | `UP(0)`→`(0,1)`, `RIGHT(1)`→`(1,0)`, `DOWN(2)`→`(0,-1)`, `LEFT(3)`→`(-1,0)`, `STAY(4)`→`(0,0)` |
| Time step | 0.02 seconds |
| Velocity values | `{-1, 0, 1}` for each of Vx, Vy |
| Position update | `X_next = X + Vx * 0.02`, `Y_next = Y + Vy * 0.02` |
| Exit region | Circle at centre `(9.5, 9.5)` with radius `0.3` m |
| Boundary rule | Clip to `[0, 10]`, set blocked velocity component to zero, apply collision penalty |

### Rewards
Same defaults as Room 1, with `step_penalty = -0.1` for the continuous domain.

---

## Room 5 — Dynamic Obstacles (Optional, Not in Phase 1)

| Property | Value |
|----------|-------|
| Algorithm | Extensible from Room 4 |
| Environment | Continuous 10×10 metre with moving obstacles |
| Observation | Configurable observation distance |
| Obstacles | 0.5 metres wide, dynamic placement |
| Evaluation | Unseen random layouts |
