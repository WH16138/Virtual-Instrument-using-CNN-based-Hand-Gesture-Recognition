import random

class Enemy:
    """적 캐릭터 (기본 AI)"""
    
    def __init__(self, max_hp=80, name="몬스터"):
        self.max_hp = max_hp
        self.hp = max_hp
        self.is_defending = False
        self.defense_multiplier = 0.5
        self.name = name
    
    def take_damage(self, damage):
        """데미지 받기"""
        actual_damage = int(damage * self.defense_multiplier if self.is_defending else damage)
        self.hp = max(0, self.hp - actual_damage)
        return actual_damage
    
    def choose_action(self):
        """AI 행동 선택 (무작위)"""
        actions = ['Attack', 'Defend', 'Attack']  # Attack 확률이 더 높음
        return random.choice(actions)
    
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
