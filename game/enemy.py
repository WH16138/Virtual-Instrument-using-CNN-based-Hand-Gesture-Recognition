import random


class Enemy:
    """Runtime enemy instance configured from an EnemyType."""

    def __init__(self, max_hp=80, name="Dungeon Fiend"):
        self.name = name
        self.max_hp = max_hp
        self.hp = max_hp
        self.base_damage = 8
        self.is_defending = False
        self.defense_multiplier = 0.5
        self.color = (50, 80, 220)
        self.model_path = None
        self.ground_model_path = None
        self.action_weights = {"Attack": 0.65, "Defend": 0.2, "Skill": 0.15}

    def configure(self, enemy_type, difficulty_multiplier):
        self.name = enemy_type.name
        self.max_hp = max(1, int(enemy_type.base_hp * difficulty_multiplier))
        self.hp = self.max_hp
        self.base_damage = max(1, int(enemy_type.base_damage * difficulty_multiplier))
        self.color = enemy_type.color
        self.model_path = enemy_type.model_path
        self.ground_model_path = enemy_type.ground_model_path
        self.action_weights = dict(enemy_type.action_weights)
        self.is_defending = False

    def take_damage(self, damage):
        actual_damage = int(damage * self.defense_multiplier if self.is_defending else damage)
        self.hp = max(0, self.hp - actual_damage)
        self.is_defending = False
        return actual_damage

    def choose_action(self):
        actions = list(self.action_weights.keys())
        weights = list(self.action_weights.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def set_defense(self, is_defending):
        self.is_defending = is_defending

    def reset(self):
        self.hp = self.max_hp
        self.is_defending = False

    @property
    def is_alive(self):
        return self.hp > 0

    @property
    def hp_percentage(self):
        return self.hp / max(self.max_hp, 1)
