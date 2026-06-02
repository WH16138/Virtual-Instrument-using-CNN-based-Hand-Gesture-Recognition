from enum import Enum

class BattleState(Enum):
    """전투 상태"""
    WAITING = 0          # 게임 시작 대기
    PLAYER_TURN = 1      # 플레이어 턴
    ENEMY_TURN = 2       # 적 턴
    VICTORY = 3          # 플레이어 승리
    DEFEAT = 4           # 플레이어 패배


class BattleSystem:
    """턴 기반 전투 시스템"""
    
    def __init__(self, player, enemy, skill_manager):
        self.player = player
        self.enemy = enemy
        self.skill_manager = skill_manager
        self.state = BattleState.WAITING
        self.last_action = None
        self.last_damage = 0
        self.turn_count = 0
    
    def start_battle(self):
        """전투 시작"""
        self.player.reset()
        self.enemy.reset()
        self.state = BattleState.PLAYER_TURN
        self.turn_count = 0
    
    def handle_player_action(self, gesture):
        """
        플레이어 행동 처리
        
        Args:
            gesture: 'Fist', 'Open_Palm', 'V_Sign'
        """
        if self.state != BattleState.PLAYER_TURN:
            return False
        
        action = self.skill_manager.get_action_from_gesture(gesture)
        skill = self.skill_manager.get_skill(gesture)
        
        if action is None:
            return False
        
        self.last_action = action
        self.last_damage = 0
        
        # 플레이어 행동 처리
        if action == 'Attack':
            self.last_damage = skill.damage
            self.enemy.take_damage(self.last_damage)
            
        elif action == 'Defend':
            self.player.set_defense(True)
            
        elif action == 'Skill':
            self.last_damage = skill.damage
            self.enemy.take_damage(self.last_damage)
        
        # 적이 살아있으면 적 턴으로 전환
        if self.enemy.is_alive:
            self.state = BattleState.ENEMY_TURN
        else:
            self.state = BattleState.VICTORY
        
        return True
    
    def enemy_turn(self):
        """적 턴 처리"""
        if self.state != BattleState.ENEMY_TURN:
            return
        
        action = self.enemy.choose_action()
        self.last_action = action
        self.last_damage = 0
        
        if action == 'Attack':
            damage = 8
            self.last_damage = self.player.take_damage(damage)
            
        elif action == 'Defend':
            self.enemy.set_defense(True)
            
        elif action == 'Skill':
            damage = 15
            self.last_damage = self.player.take_damage(damage)
        
        # 플레이어가 죽었는지 확인
        if not self.player.is_alive:
            self.state = BattleState.DEFEAT
        else:
            self.state = BattleState.PLAYER_TURN
        
        self.turn_count += 1
    
    def reset_defenses(self):
        """방어 상태 초기화"""
        self.player.set_defense(False)
        self.enemy.set_defense(False)
    
    def reset_battle(self):
        """전투 초기화"""
        self.state = BattleState.WAITING
        self.player.reset()
        self.enemy.reset()
        self.turn_count = 0
    
    @property
    def is_battle_active(self):
        return self.state in [BattleState.PLAYER_TURN, BattleState.ENEMY_TURN]
