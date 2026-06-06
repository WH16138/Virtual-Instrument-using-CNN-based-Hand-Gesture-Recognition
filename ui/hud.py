import cv2

from game.battle_system import BattleState


class HUD:
    """Minimal OpenCV HUD for setup and non-AR global game state."""

    @staticmethod
    def _panel(frame, x, y, w, h, color=(12, 12, 18), alpha=0.62, border=(92, 86, 68)):
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame.shape[1], int(x + w))
        y2 = min(frame.shape[0], int(y + h))
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        overlay[:, :] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)

    @staticmethod
    def _text(frame, text, x, y, scale=0.5, color=(240, 240, 240), thickness=1):
        cv2.putText(frame, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _bar(frame, x, y, w, h, ratio, color, label):
        ratio = max(0.0, min(1.0, ratio))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 36, 44), -1)
        cv2.rectangle(frame, (x, y), (x + int(w * ratio), y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (205, 190, 140), 1, cv2.LINE_AA)
        HUD._text(frame, label, x + 6, y + h - 4, 0.38, (250, 246, 226), 1)

    @staticmethod
    def draw_game_layer(frame, game_state, gesture_info, plane_registered, game_started, setup_hold_progress=0.0):
        if game_started:
            return HUD._draw_minimal_game_hud(frame, game_state)
        return HUD._draw_setup_panel(frame, plane_registered, setup_hold_progress)

    @staticmethod
    def _draw_minimal_game_hud(frame, game_state):
        height, width = frame.shape[:2]
        panel_w = min(360, width - 16)
        HUD._panel(frame, 8, 8, panel_w, 48, alpha=0.45, border=(80, 76, 60))

        wave = game_state.get("wave", 0)
        best = game_state.get("best_wave", 0)
        phase = game_state.get("phase_label", "")
        state = game_state.get("battle_state")
        if state == BattleState.PLAYER_TURN and game_state.get("can_act"):
            phase = "Choose an action"
        elif state == BattleState.ROUND_REVEAL:
            reveal = game_state.get("round_reveal", {})
            phase = f"Reveal: {reveal.get('player_action', '?')} vs {reveal.get('enemy_action', '?')}"

        HUD._text(frame, f"Wave {wave}  Best {best}", 20, 28, 0.56, (255, 245, 210), 2)
        HUD._text(frame, phase, 20, 48, 0.42, (120, 230, 255), 1)

        if state == BattleState.DEFEAT:
            text = f"DEFEAT - BEST WAVE {game_state.get('best_wave', 0)}"
            HUD._panel(frame, width // 2 - 190, height // 2 - 34, 380, 68, alpha=0.72, border=(80, 80, 255))
            HUD._text(frame, text, width // 2 - 160, height // 2 + 8, 0.72, (95, 95, 255), 2)
        elif state == BattleState.WAVE_CLEAR:
            HUD._panel(frame, width // 2 - 140, 96, 280, 48, alpha=0.60, border=(90, 230, 130))
            HUD._text(frame, "WAVE CLEAR", width // 2 - 112, 128, 0.78, (90, 230, 130), 2)
        elif state == BattleState.WAVE_INTRO:
            HUD._panel(frame, width // 2 - 190, 96, 380, 48, alpha=0.58, border=(120, 230, 255))
            HUD._text(frame, game_state.get("phase_label", "Wave incoming"), width // 2 - 170, 128, 0.62, (120, 230, 255), 2)

        return frame

    @staticmethod
    def _draw_setup_panel(frame, plane_registered, setup_hold_progress=0.0):
        height, _ = frame.shape[:2]
        w, h = 430, 92
        x, y = 10, 10
        HUD._panel(frame, x, y, w, h, alpha=0.56)
        if not plane_registered:
            title = "Board setup"
            detail = "Show the gate marker, then hold OK."
            color = (80, 190, 255)
        else:
            title = "Board registered"
            detail = "Keep holding OK to start."
            color = (90, 230, 130)
        HUD._text(frame, title, x + 14, y + 28, 0.58, color, 2)
        HUD._text(frame, detail, x + 14, y + 54, 0.44, (235, 235, 235), 1)
        HUD._bar(frame, x + 14, y + 68, w - 28, 10, setup_hold_progress, (70, 220, 255), "OK hold")
        HUD._text(frame, "[D] debug  [R] reset  [Q] quit", 18, height - 16, 0.42, (230, 230, 230), 1)
        return frame

    @staticmethod
    def draw_instructions(frame, game_state):
        return frame

    @staticmethod
    def draw_hp_bar(frame, player, enemy, x_offset=12, y_offset=62):
        return frame

    @staticmethod
    def draw_game_state(frame, game_state, x_offset=12, y_offset=205):
        return frame

    @staticmethod
    def draw_gesture_recognition(frame, gesture_info, x_offset=12, y_offset=350):
        return frame
