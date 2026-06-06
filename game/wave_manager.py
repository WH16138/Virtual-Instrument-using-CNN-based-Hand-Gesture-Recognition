import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnemyType:
    name: str
    base_hp: int
    base_damage: int
    color: tuple
    full_health_action_weights: dict
    zero_health_action_weights: dict
    action_weight_random_delta: float
    min_wave: int = 1
    model_path: str | None = None
    ground_model_path: str | None = None


class WaveManager:
    """Infinite wave progression and enemy selection."""

    DIFFICULTY_GROWTH_RATE = 1.15

    def __init__(self):
        self.enemy_types = [
            EnemyType(
                name="Slime",
                base_hp=70,
                base_damage=6,
                color=(50, 200, 50),
                full_health_action_weights={"Attack": 60, "Defend": 35, "Skill": 15},
                zero_health_action_weights={"Attack": 30, "Defend": 50, "Skill": 20},
                action_weight_random_delta=6,
                min_wave=1,
                model_path=str(Path("assets") / "models" / "Slime.glb"),
                ground_model_path=str(Path("assets") / "models" / "Grass.glb"),
            ),
            EnemyType(
                name="Skeleton",
                base_hp=60,
                base_damage=10,
                color=(200, 200, 200),
                full_health_action_weights={"Attack": 50, "Defend": 30, "Skill": 20},
                zero_health_action_weights={"Attack": 35, "Defend": 35, "Skill": 30},
                action_weight_random_delta=3,
                min_wave=1,
                model_path=str(Path("assets") / "models" / "Skeleton.glb"),
                ground_model_path=str(Path("assets") / "models" / "StoneGround.glb"),
            ),
            EnemyType(
                name="Ghost",
                base_hp=50,
                base_damage=8,
                color=(150, 150, 255),
                full_health_action_weights={"Attack": 25, "Defend": 45, "Skill": 30},
                zero_health_action_weights={"Attack": 40, "Defend": 15, "Skill": 50},
                action_weight_random_delta=15,
                min_wave=1,
                model_path=str(Path("assets") / "models" / "Ghost.glb"),
                ground_model_path=str(Path("assets") / "models" / "BlackStoneGround.glb"),
            ),
            EnemyType(
                name="Dragon",
                base_hp=120,
                base_damage=18,
                color=(255, 100, 50),
                full_health_action_weights={"Attack": 55, "Defend": 10, "Skill": 45},
                zero_health_action_weights={"Attack": 30, "Defend": 40, "Skill": 30},
                action_weight_random_delta=10,
                min_wave=4,
                model_path=str(Path("assets") / "models" / "Dragon.glb"),
                ground_model_path=str(Path("assets") / "models" / "PowerGround.glb"),
            ),
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
        self.global_difficulty_multiplier = self.DIFFICULTY_GROWTH_RATE ** (self.current_wave - 1)
        eligible_enemy_types = [
            enemy_type
            for enemy_type in self.enemy_types
            if enemy_type.min_wave <= self.current_wave
        ]
        if not eligible_enemy_types:
            eligible_enemy_types = self.enemy_types
        self.current_enemy_type = random.choice(eligible_enemy_types)
        return self.current_enemy_type, self.global_difficulty_multiplier

    def finish_run(self):
        self.best_wave = max(self.best_wave, self.current_wave)
