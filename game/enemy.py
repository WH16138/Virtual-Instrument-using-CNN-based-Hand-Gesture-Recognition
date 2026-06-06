import random


class Enemy:
    """Runtime enemy instance configured from an EnemyType."""

    ACTIONS = ("Attack", "Defend", "Skill")

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
        self.full_health_action_weights = {"Attack": 0.65, "Defend": 0.20, "Skill": 0.15}
        self.zero_health_action_weights = {"Attack": 0.45, "Defend": 0.35, "Skill": 0.20}
        self.action_weight_random_delta = 0.06
        self.action_weights = dict(self.full_health_action_weights)
        self.action_probabilities = self._normalize_weights(self.action_weights)

    def configure(self, enemy_type, difficulty_multiplier):
        self.name = enemy_type.name
        self.max_hp = max(1, int(enemy_type.base_hp * difficulty_multiplier))
        self.hp = self.max_hp
        self.base_damage = max(1, int(enemy_type.base_damage * difficulty_multiplier))
        self.color = enemy_type.color
        self.model_path = enemy_type.model_path
        self.ground_model_path = enemy_type.ground_model_path
        self.full_health_action_weights = self._ordered_weights(enemy_type.full_health_action_weights)
        self.zero_health_action_weights = self._ordered_weights(enemy_type.zero_health_action_weights)
        self.action_weight_random_delta = max(0.0, float(enemy_type.action_weight_random_delta))
        self.is_defending = False
        self.prepare_action_weights()

    def _ordered_weights(self, weights):
        weights = weights or {}
        return {
            action: max(0.0, float(weights.get(action, 0.0)))
            for action in self.ACTIONS
        }

    def _normalize_weights(self, weights):
        ordered = self._ordered_weights(weights)
        total = sum(ordered.values())
        if total <= 1e-6:
            return {action: 1.0 / len(self.ACTIONS) for action in self.ACTIONS}
        return {action: value / total for action, value in ordered.items()}

    def _health_interpolated_action_weights(self):
        hp_ratio = max(0.0, min(1.0, self.hp_percentage))
        return {
            action: self.zero_health_action_weights[action] * (1.0 - hp_ratio)
            + self.full_health_action_weights[action] * hp_ratio
            for action in self.ACTIONS
        }

    def prepare_action_weights(self):
        """Roll this turn's enemy action probabilities from HP-based base weights."""
        base_weights = self._health_interpolated_action_weights()
        varied_weights = {}
        for action, base_value in base_weights.items():
            random_offset = random.uniform(-1.0, 1.0) * self.action_weight_random_delta
            varied_weights[action] = max(0.001, base_value + random_offset)
        self.action_weights = varied_weights
        self.action_probabilities = self._normalize_weights(varied_weights)
        return dict(self.action_probabilities)

    def get_action_probabilities(self):
        if not self.action_probabilities:
            self.prepare_action_weights()
        return dict(self.action_probabilities)

    def take_damage(self, damage):
        actual_damage = int(damage * self.defense_multiplier if self.is_defending else damage)
        self.hp = max(0, self.hp - actual_damage)
        self.is_defending = False
        return actual_damage

    def choose_action(self):
        probabilities = self.get_action_probabilities()
        actions = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + int(amount))

    def set_defense(self, is_defending):
        self.is_defending = is_defending

    def reset(self):
        self.hp = self.max_hp
        self.is_defending = False
        self.prepare_action_weights()

    @property
    def is_alive(self):
        return self.hp > 0

    @property
    def hp_percentage(self):
        return self.hp / max(self.max_hp, 1)
