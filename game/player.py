class Player:
    """Runtime player state and run-limited growth values."""

    def __init__(self, max_hp=100, attack_power=15):
        self.base_max_hp = int(max_hp)
        self.base_attack_power = int(attack_power)
        self.max_hp = int(max_hp)
        self.hp = int(max_hp)
        self.is_defending = False
        self.defense_multiplier = 0.5
        self.attack_power = self.base_attack_power
        self.strike_bonus = 0
        self.guard_bonus = 0
        self.guard_heal_ratio_bonus = 0.0
        self.shot_bonus = 0
        self.damage_multiplier = 1.0
        self.heal_multiplier = 1.0
        self.augment_flags = set()

    def take_damage(self, damage):
        actual_damage = int(damage * self.defense_multiplier if self.is_defending else damage)
        self.hp = max(0, self.hp - actual_damage)
        self.is_defending = False
        return actual_damage

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + int(amount))

    def set_defense(self, is_defending):
        self.is_defending = is_defending

    def reset_growth(self):
        self.max_hp = self.base_max_hp
        self.hp = self.max_hp
        self.attack_power = self.base_attack_power
        self.strike_bonus = 0
        self.guard_bonus = 0
        self.guard_heal_ratio_bonus = 0.0
        self.shot_bonus = 0
        self.damage_multiplier = 1.0
        self.heal_multiplier = 1.0
        self.augment_flags = set()

    def reset(self):
        self.reset_growth()
        self.is_defending = False

    @property
    def is_alive(self):
        return self.hp > 0

    @property
    def hp_percentage(self):
        return self.hp / max(self.max_hp, 1)