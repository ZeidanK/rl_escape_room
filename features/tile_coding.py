import numpy as np

from core.types import ContinuousState, TileCodingConfig

_VELOCITY_INDEX: dict[int, int] = {-1: 0, 0: 1, 1: 2}


class TileCoder:
    # Converts continuous Room 4 states into a small set of active feature
    # indexes.  Multiple offset tilings let nearby positions share some, but
    # not all, features.
    def __init__(self, config: TileCodingConfig, room_width: float, room_height: float) -> None:
        self.config = config
        self.room_width = room_width
        self.room_height = room_height
        self._tile_width = room_width / config.tiles_x
        self._tile_height = room_height / config.tiles_y
        nv = 3 if config.include_velocity else 1
        self._n_vel = nv
        self._vx_stride = nv
        self._vy_stride = nv * nv
        self._y_stride = config.tiles_y * nv * nv
        self._x_stride = config.tiles_x * config.tiles_y * nv * nv
        self._feature_count = config.num_tilings * config.tiles_x * config.tiles_y * nv * nv

    @property
    def feature_count(self) -> int:
        return self._feature_count

    def active_features(self, state: ContinuousState) -> tuple[int, ...]:
        # One active tile per tiling.  Velocity is included as discrete bins so
        # the agent can learn different values for the same location at
        # different movement directions.
        x, y, vx, vy = state
        x = float(np.clip(x, 0.0, self.room_width))
        y = float(np.clip(y, 0.0, self.room_height))
        vxi = _VELOCITY_INDEX.get(vx, 1)
        vyi = _VELOCITY_INDEX.get(vy, 1)
        features = []
        for t in range(self.config.num_tilings):
            offset = t * 7  # arbitrary prime offset to shift each tiling
            ox = (offset % self.config.tiles_x) * self._tile_width / self.config.num_tilings
            oy = (offset // self.config.tiles_x) * self._tile_height / self.config.num_tilings
            bx = min(int((x + ox) / self._tile_width), self.config.tiles_x - 1)
            by = min(int((y + oy) / self._tile_height), self.config.tiles_y - 1)
            idx = (
                t * self._x_stride
                + bx * self._y_stride
                + by * self._vy_stride
                + vxi * self._vx_stride
                + vyi
            )
            features.append(idx)
        return tuple(features)
