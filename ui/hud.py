import cv2

from game.battle_system import BattleState


class HUD:
    """Minimal OpenCV HUD for gameplay, setup guidance before battle."""

    @staticmethod
    def _panel(frame, x, y, w, h, color=(12, 12, 18), alpha=0.62, border=(92, 86, 68)):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)

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
    def draw_game_layer(frame, game_state, gesture_info, plane_registered, game_started):
        if game_started:
            return HUD._draw_minimal_game_hud(frame, game_state)
        return HUD._draw_setup_panel(frame, plane_registered)

    @staticmethod
    def _draw_minimal_game_hud(frame, game_state):
        height, width = frame.shape[:2]
        panel_w = min(460, width - 16)
        HUD._panel(frame, 8, 8, panel_w, 74, alpha=0.58)

        wave = game_state.get("wave", 0)
        best = game_state.get("best_wave", 0)
        phase = game_state.get("phase_label", "")
        state = game_state.get("battle_state")
        if state == BattleState.PLAYER_TURN and game_state.get("can_act"):
            phase = "Choose an action"
        elif state == BattleState.ENEMY_TURN:
            preview = game_state.get("enemy_preview", {})
            phase = f"Enemy: {preview.get('action', '...')}" if preview.get("active") else "Enemy turn"

        HUD._text(frame, f"Wave {wave}  Best {best}", 20, 30, 0.58, (255, 245, 210), 2)
        HUD._text(frame, phase, 190, 30, 0.52, (120, 230, 255), 1)

        player = game_state.get("player", {})
        enemy = game_state.get("enemy", {})
        HUD._bar(
            frame,
            20,
            50,
            180,
            14,
            player.get("hp", 0) / max(player.get("max_hp", 1), 1),
            (60, 185, 85),
            f"HP {player.get('hp', 0)}/{player.get('max_hp', 1)}",
        )
        HUD._bar(
            frame,
            218,
            50,
            218,
            14,
            enemy.get("hp", 0) / max(enemy.get("max_hp", 1), 1),
            enemy.get("color", (70, 70, 210)),
            f"{enemy.get('name', 'Enemy')} {enemy.get('hp', 0)}/{enemy.get('max_hp', 1)}",
        )

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
    def _draw_setup_panel(frame, plane_registered):
        height, _ = frame.shape[:2]
        w, h = 430, 74
        x, y = 10, 10
        HUD._panel(frame, x, y, w, h, alpha=0.56)
        if not plane_registered:
            title = "Board setup"
            detail = "Place marked A4 in view, then press SPACE."
            color = (80, 190, 255)
        else:
            title = "Board registered"
            detail = "Show OK sign or press SPACE to start."
            color = (90, 230, 130)
        HUD._text(frame, title, x + 14, y + 28, 0.58, color, 2)
        HUD._text(frame, detail, x + 14, y + 54, 0.44, (235, 235, 235), 1)
        HUD._text(frame, "[SPACE] start/register  [D] debug  [R] reset  [Q] quit", 18, height - 16, 0.42, (230, 230, 230), 1)
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
