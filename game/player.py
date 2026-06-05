class Player:
    """Runtime player state."""

    def __init__(self, max_hp=100):
        self.max_hp = max_hp
        self.hp = max_hp
        self.is_defending = False
        self.defense_multiplier = 0.5

    def take_damage(self, damage):
        actual_damage = int(damage * self.defense_multiplier if self.is_defending else damage)
        self.hp = max(0, self.hp - actual_damage)
        self.is_defending = False
        return actual_damage

    def heal(self, amount):
        self.hp = min(self.max_hp, self.hp + amount)

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
