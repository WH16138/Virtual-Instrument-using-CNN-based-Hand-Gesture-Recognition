class Skill:
    """스킬"""
    
    def __init__(self, name, damage, description):
        self.name = name
        self.damage = damage
        self.description = description


class SkillManager:
    """스킬 관리"""
    
    def __init__(self):
        self.skills = {
            'Fist': Skill('기본 공격', damage=10, description='주먹으로 공격'),
            'Open_Palm': Skill('방어', damage=0, description='손을 펼쳐 방어'),
            'V_Sign': Skill('강력한 공격', damage=20, description='V자 손가락으로 강력한 공격')
        }
    
    def get_skill(self, gesture):
        """제스처에 해당하는 스킬 반환"""
        return self.skills.get(gesture, None)
    
    def get_action_from_gesture(self, gesture):
        """제스처 → 게임 액션 매핑"""
        action_map = {
            'Fist': 'Attack',
            'Open_Palm': 'Defend',
            'V_Sign': 'Skill'
        }
        return action_map.get(gesture, None)
