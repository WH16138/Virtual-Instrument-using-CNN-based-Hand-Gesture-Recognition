class Player:
    """플레이어 캐릭터"""
    
    def __init__(self, max_hp=100):
        self.max_hp = max_hp
        self.hp = max_hp
        self.is_defending = False
        self.defense_multiplier = 0.5  # 방어 시 받는 데미지 50%
    
    def take_damage(self, damage):
        """데미지 받기"""
        actual_damage = int(damage * self.defense_multiplier if self.is_defending else damage)
        self.hp = max(0, self.hp - actual_damage)
        return actual_damage
    
    def heal(self, amount):
        """회복"""
        self.hp = min(self.max_hp, self.hp + amount)
    
    def set_defense(self, is_defending):
        """방어 상태 설정"""
        self.is_defending = is_defending
    
    def reset(self):
        """초기화"""
        self.hp = self.max_hp
        self.is_defending = False
    
    @property
    def is_alive(self):
        return self.hp > 0
    
    @property
    def hp_percentage(self):
        return self.hp / self.max_hp
