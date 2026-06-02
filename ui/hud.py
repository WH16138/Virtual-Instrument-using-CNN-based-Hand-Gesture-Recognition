import cv2

class HUD:
    """게임 HUD 렌더링"""
    
    @staticmethod
    def draw_hp_bar(frame, player, enemy, x_offset=10, y_offset=10):
        """
        HP 바 그리기
        
        Args:
            frame: BGR 프레임
            player: Player 객체
            enemy: Enemy 객체
        """
        bar_width = 200
        bar_height = 20
        gap = 50
        
        # 플레이어 HP 바
        player_hp_ratio = player.hp_percentage
        cv2.rectangle(frame, (x_offset, y_offset), 
                     (x_offset + bar_width, y_offset + bar_height), (100, 100, 100), 2)
        cv2.rectangle(frame, (x_offset, y_offset),
                     (x_offset + int(bar_width * player_hp_ratio), y_offset + bar_height),
                     (0, 0, 255), -1)
        cv2.putText(frame, f'Player HP: {player.hp}/{player.max_hp}',
                   (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 적 HP 바
        enemy_hp_ratio = enemy.hp_percentage
        cv2.rectangle(frame, (x_offset, y_offset + gap + bar_height),
                     (x_offset + bar_width, y_offset + gap + 2*bar_height), (100, 100, 100), 2)
        cv2.rectangle(frame, (x_offset, y_offset + gap + bar_height),
                     (x_offset + int(bar_width * enemy_hp_ratio), y_offset + gap + 2*bar_height),
                     (0, 255, 0), -1)
        cv2.putText(frame, f'Enemy HP: {enemy.hp}/{enemy.max_hp}',
                   (x_offset, y_offset + gap + bar_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    @staticmethod
    def draw_game_state(frame, game_state, x_offset=10, y_offset=100):
        """
        게임 상태 정보 그리기
        """
        battle_state_names = {
            0: "대기 중",
            1: "플레이어 턴",
            2: "적 턴",
            3: "승리!",
            4: "패배!"
        }
        
        state = game_state['battle_state'].value
        state_name = battle_state_names.get(state, "Unknown")
        
        cv2.putText(frame, f'상태: {state_name}', (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if game_state['last_action']:
            cv2.putText(frame, f'마지막 행동: {game_state["last_action"]}',
                       (x_offset, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if game_state['last_damage'] > 0:
            cv2.putText(frame, f'데미지: {game_state["last_damage"]}',
                       (x_offset, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    @staticmethod
    def draw_gesture_recognition(frame, gesture_info, x_offset=10, y_offset=200):
        """
        제스처 인식 결과 표시
        """
        gesture = gesture_info.get('gesture', 'Unknown')
        confidence = gesture_info.get('confidence', 0.0)
        smoothed_gesture = gesture_info.get('smoothed_gesture', 'Unknown')
        
        color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
        
        cv2.putText(frame, f'제스처: {gesture} ({confidence:.2f})',
                   (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f'스무딩: {smoothed_gesture}',
                   (x_offset, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    @staticmethod
    def draw_instructions(frame, game_state):
        """게임 안내 메시지 표시"""
        height, width = frame.shape[:2]
        
        cv2.putText(frame, 'VisionQuest - AR 손 제스처 전투',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, '[SPACE] 게임 시작 | [R] 재시작 | [Q] 종료',
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        battle_state = game_state['battle_state'].value
        if battle_state == 3:  # VICTORY
            cv2.putText(frame, '*** 플레이어 승리! ***',
                       (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        elif battle_state == 4:  # DEFEAT
            cv2.putText(frame, '*** 플레이어 패배! ***',
                       (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return frame
