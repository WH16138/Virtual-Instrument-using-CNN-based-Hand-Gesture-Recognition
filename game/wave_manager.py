import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnemyType:
    name: str
    base_hp: int
    base_damage: int
    color: tuple
    action_weights: dict
    model_path: str | None = None
    ground_model_path: str | None = None


class WaveManager:
    """Infinite wave progression and enemy selection."""

    def __init__(self):
        self.enemy_types = [
            EnemyType(
                name="Slime",
                base_hp=50,
                base_damage=6,
                color=(50, 200, 50),
                action_weights={"Attack": 0.50, "Defend": 0.30, "Skill": 0.20},
                model_path=str(Path("assets") / "models" / "Slime.glb"),
                ground_model_path=str(Path("assets") / "models" / "Grass.glb"),
            ),
            EnemyType(
                name="Skeleton",
                base_hp=60,
                base_damage=10,
                color=(200, 200, 200),
                action_weights={"Attack": 0.55, "Defend": 0.35, "Skill": 0.10},
                model_path=str(Path("assets") / "models" / "Skeleton.glb"),
                ground_model_path=str(Path("assets") / "models" / "StoneGround.glb"),
            ),
            EnemyType(
                name="Ghost",
                base_hp=40,
                base_damage=8,
                color=(150, 150, 255),
                action_weights={"Attack": 0.50, "Defend": 0.20, "Skill": 0.30},
                model_path=str(Path("assets") / "models" / "Ghost.glb"),
                ground_model_path=str(Path("assets") / "models" / "BlackStoneGround.glb"),
            )
        ]
        self.current_wave = 0
        self.best_wave = 0
        self.current_enemy_type = self.enemy_types[0]
        self.global_difficulty_multiplier = 1.0

    def reset_run(self):
        self.current_wave = 0
        self.global_difficulty_multiplier = 1.0
        self.current_enemy_type = self.enemy_types[0]

    def next_wave(self):
        self.current_wave += 1
        self.global_difficulty_multiplier = 1.0 + (self.current_wave - 1) * 0.18
        self.current_enemy_type = random.choice(self.enemy_types)
        return self.current_enemy_type, self.global_difficulty_multiplier

    def finish_run(self):
        self.best_wave = max(self.best_wave, self.current_wave)
