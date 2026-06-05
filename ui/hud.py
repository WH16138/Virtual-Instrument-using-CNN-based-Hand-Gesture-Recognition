import cv2

from game.battle_system import BattleState


class HUD:
    """OpenCV HUD drawing helpers."""

    @staticmethod
    def _panel(frame, x, y, w, h, color=(18, 18, 18), alpha=0.72):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    @staticmethod
    def _text(frame, text, x, y, scale=0.55, color=(255, 255, 255), thickness=1):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _hp_bar(frame, label, hp, max_hp, x, y, w, color):
        h = 22
        ratio = max(0.0, min(1.0, hp / max(max_hp, 1)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (65, 65, 65), -1)
        cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (230, 230, 230), 1)
        HUD._text(frame, f"{label} {hp}/{max_hp}", x, y - 8, 0.55, (255, 255, 255), 1)

    @staticmethod
    def draw_hp_bar(frame, player, enemy, x_offset=12, y_offset=62):
        HUD._panel(frame, x_offset - 8, y_offset - 40, 270, 128)
        HUD._hp_bar(frame, "PLAYER", player.hp, player.max_hp, x_offset, y_offset, 230, (35, 95, 235))
        HUD._hp_bar(frame, "ENEMY", enemy.hp, enemy.max_hp, x_offset, y_offset + 62, 230, (50, 190, 70))
        return frame

    @staticmethod
    def draw_game_state(frame, game_state, x_offset=12, y_offset=205):
        state = game_state["battle_state"]
        state_names = {
            BattleState.WAITING: "WAITING",
            BattleState.PLAYER_TURN: "PLAYER TURN",
            BattleState.ENEMY_TURN: "ENEMY TURN",
            BattleState.VICTORY: "VICTORY",
            BattleState.DEFEAT: "DEFEAT",
        }
        state_colors = {
            BattleState.PLAYER_TURN: (0, 220, 255),
            BattleState.ENEMY_TURN: (0, 150, 255),
            BattleState.VICTORY: (60, 230, 80),
            BattleState.DEFEAT: (60, 60, 255),
        }
        state_name = state_names.get(state, "UNKNOWN")
        state_color = state_colors.get(state, (235, 235, 235))
        delay = game_state.get("turn_delay_remaining", 0.0)
        can_act = game_state.get("can_act", False)

        HUD._panel(frame, x_offset - 8, y_offset - 36, 310, 120)
        HUD._text(frame, f"TURN {game_state.get('turn_count', 0)}", x_offset, y_offset - 12, 0.55, (210, 210, 210), 1)
        HUD._text(frame, state_name, x_offset, y_offset + 18, 0.82, state_color, 2)

        if state == BattleState.PLAYER_TURN:
            prompt = "Action ready" if can_act else f"Ready in {delay:.1f}s"
        elif state == BattleState.ENEMY_TURN:
            prompt = f"Enemy acts in {delay:.1f}s"
        else:
            prompt = game_state.get("last_action") or "Press SPACE after registering A4"
        HUD._text(frame, prompt, x_offset, y_offset + 48, 0.55, (255, 255, 255), 1)

        if game_state.get("last_action"):
            damage = game_state.get("last_damage", 0)
            suffix = f" | Damage {damage}" if damage > 0 else ""
            HUD._text(frame, f"{game_state['last_action']}{suffix}", x_offset, y_offset + 76, 0.5, (255, 230, 120), 1)

        return frame

    @staticmethod
    def draw_gesture_recognition(frame, gesture_info, x_offset=12, y_offset=350):
        gesture = gesture_info.get("gesture", "Unknown")
        confidence = gesture_info.get("confidence", 0.0)
        smoothed_gesture = gesture_info.get("smoothed_gesture", "Unknown")
        color = (0, 230, 90) if confidence > 0.6 else (0, 170, 255)

        HUD._panel(frame, x_offset - 8, y_offset - 34, 310, 76)
        HUD._text(frame, f"Gesture {gesture}  {confidence:.2f}", x_offset, y_offset - 8, 0.52, color, 1)
        HUD._text(frame, f"Stable {smoothed_gesture}", x_offset, y_offset + 22, 0.58, (255, 230, 120), 1)
        return frame

    @staticmethod
    def draw_instructions(frame, game_state):
        height, width = frame.shape[:2]
        state = game_state["battle_state"]

        HUD._panel(frame, 8, 8, 440, 34, alpha=0.6)
        HUD._text(frame, "VisionQuest - A4 Gesture Battle", 18, 32, 0.65, (255, 255, 255), 2)

        HUD._panel(frame, 8, height - 38, 520, 30, alpha=0.62)
        HUD._text(frame, "[SPACE] register/start fallback   [R] reset   [Q] quit", 18, height - 16, 0.5, (230, 230, 230), 1)

        if state in (BattleState.VICTORY, BattleState.DEFEAT):
            label = "VICTORY" if state == BattleState.VICTORY else "DEFEAT"
            color = (60, 230, 80) if state == BattleState.VICTORY else (60, 60, 255)
            HUD._panel(frame, width // 2 - 170, height // 2 - 50, 340, 100, alpha=0.78)
            HUD._text(frame, label, width // 2 - 95, height // 2 + 10, 1.15, color, 3)

        return frame
