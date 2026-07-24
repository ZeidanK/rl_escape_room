"""Achievement system — tracks real episode metrics and unlocks achievements."""

import json
import os
from pathlib import Path

from game.models import Achievement, AchievementId

ACHIEVEMENTS_FILE = Path("storage/achievements.json")

ALL_ACHIEVEMENTS: dict[AchievementId, Achievement] = {
    AchievementId.FIRST_ESCAPE: Achievement(
        id=AchievementId.FIRST_ESCAPE,
        name="First Escape",
        description="Successfully escape any room for the first time",
        emoji="\U0001f3c6",
    ),
    AchievementId.ICE_MASTER: Achievement(
        id=AchievementId.ICE_MASTER,
        name="Ice Master",
        description="Escape Room 1 without any unintended slip",
        emoji="\u2744\ufe0f",
    ),
    AchievementId.LASER_DODGER: Achievement(
        id=AchievementId.LASER_DODGER,
        name="Laser Dodger",
        description="Escape Room 2 without visiting any trap cell",
        emoji="\u26a1",
    ),
    AchievementId.VAULT_EXPERT: Achievement(
        id=AchievementId.VAULT_EXPERT,
        name="Vault Expert",
        description="Collect the key and exit Room 3 successfully",
        emoji="\U0001f511",
    ),
    AchievementId.MOMENTUM_MASTER: Achievement(
        id=AchievementId.MOMENTUM_MASTER,
        name="Momentum Master",
        description="Solve Room 4 from an unseen start position",
        emoji="\U0001f300",
    ),
    AchievementId.SPEED_RUNNER: Achievement(
        id=AchievementId.SPEED_RUNNER,
        name="Speed Runner",
        description="Achieve a new minimum step count in any room",
        emoji="\u23f1\ufe0f",
    ),
}


class AchievementTracker:
    def __init__(self):
        self._unlocked: set[AchievementId] = set()

    def is_unlocked(self, ach_id: AchievementId) -> bool:
        return ach_id in self._unlocked

    def unlock(self, ach_id: AchievementId) -> Achievement | None:
        if ach_id not in self._unlocked:
            self._unlocked.add(ach_id)
            self._save()
            return ALL_ACHIEVEMENTS[ach_id]
        return None

    def try_unlock_first_escape(self) -> Achievement | None:
        return self.unlock(AchievementId.FIRST_ESCAPE)

    def try_unlock_ice_master(self, slip_count: int) -> Achievement | None:
        if slip_count == 0:
            return self.unlock(AchievementId.ICE_MASTER)
        return None

    def try_unlock_laser_dodger(self, trap_count: int) -> Achievement | None:
        if trap_count == 0:
            return self.unlock(AchievementId.LASER_DODGER)
        return None

    def try_unlock_vault_expert(self, success: bool, key_collected: bool) -> Achievement | None:
        if success and key_collected:
            return self.unlock(AchievementId.VAULT_EXPERT)
        return None

    def try_unlock_momentum_master(self, unseen_start: bool, success: bool) -> Achievement | None:
        if unseen_start and success:
            return self.unlock(AchievementId.MOMENTUM_MASTER)
        return None

    def try_unlock_speed_runner(self, is_new_best: bool) -> Achievement | None:
        if is_new_best:
            return self.unlock(AchievementId.SPEED_RUNNER)
        return None

    def get_unlocked(self) -> list[Achievement]:
        return [ALL_ACHIEVEMENTS[a] for a in self._unlocked]

    def get_all(self) -> list[Achievement]:
        return list(ALL_ACHIEVEMENTS.values())

    def _save(self):
        try:
            ACHIEVEMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = [a.value for a in self._unlocked]
            ACHIEVEMENTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    @classmethod
    def _load_from_file(cls) -> set[AchievementId]:
        unlocked: set[AchievementId] = set()
        try:
            if ACHIEVEMENTS_FILE.exists():
                data = json.loads(ACHIEVEMENTS_FILE.read_text())
                for name in data:
                    try:
                        unlocked.add(AchievementId(name))
                    except ValueError:
                        continue
        except Exception:
            pass
        return unlocked

    @classmethod
    def from_session_state(cls):
        import streamlit as st
        if "achievement_tracker" not in st.session_state:
            tracker = cls()
            tracker._unlocked = cls._load_from_file()
            st.session_state.achievement_tracker = tracker
        return st.session_state.achievement_tracker
