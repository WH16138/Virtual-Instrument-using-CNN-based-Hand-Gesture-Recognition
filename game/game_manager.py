from game.battle_system import BattleSystem, BattleState
from game.player import Player
from game.enemy import Enemy
from game.skills import SkillManager

class GameManager:
    """게임 로직 오케스트레이터"""
    
    def __init__(self):
        self.player = Player(max_hp=100)
        self.enemy = Enemy(max_hp=80)
        self.skill_manager = SkillManager()
        self.battle_system = BattleSystem(self.player, self.enemy, self.skill_manager)
        
        # 게임 상태
        self.player_pos = (200, 150)  # 평면 좌표
        self.enemy_pos = (200, 50)
        self.gesture_buffer = []  # 최근 인식된 제스처 버퍼
    
    def start_game(self):
        """게임 시작"""
        self.battle_system.start_battle()
    
    def process_gesture(self, gesture_info):
        """
        제스처 처리
        
        Args:
            gesture_info: {'gesture': str, 'confidence': float, 'smoothed_gesture': str}
        """
        if not self.battle_system.is_battle_active:
            return
        
        gesture = gesture_info.get('smoothed_gesture', 'Unknown')
        confidence = gesture_info.get('confidence', 0.0)
        
        # 신뢰도가 높을 때만 처리
        if confidence > 0.6 and gesture != 'Unknown':
            # 버퍼에 추가하여 중복 입력 방지
            if len(self.gesture_buffer) == 0 or self.gesture_buffer[-1] != gesture:
                self.gesture_buffer.append(gesture)
                
                # 플레이어 턴에 행동 처리
                if self.battle_system.state == BattleState.PLAYER_TURN:
                    self.battle_system.handle_player_action(gesture)
    
    def update(self):
        """게임 상태 업데이트"""
        # 적 턴 자동 진행
        if self.battle_system.state == BattleState.ENEMY_TURN:
            self.battle_system.enemy_turn()
            self.battle_system.reset_defenses()
    
    def reset_game(self):
        """게임 초기화"""
        self.battle_system.reset_battle()
        self.gesture_buffer.clear()
    
    def get_game_state(self):
        """현재 게임 상태 반환"""
        return {
            'player': {
                'hp': self.player.hp,
                'max_hp': self.player.max_hp,
                'is_defending': self.player.is_defending
            },
            'enemy': {
                'hp': self.enemy.hp,
                'max_hp': self.enemy.max_hp,
                'is_defending': self.enemy.is_defending
            },
            'battle_state': self.battle_system.state,
            'last_action': self.battle_system.last_action,
            'last_damage': self.battle_system.last_damage,
            'turn_count': self.battle_system.turn_count
        }
