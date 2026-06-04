import cv2


class HUD:
    """OpenCV HUD drawing helpers."""

    @staticmethod
    def draw_hp_bar(frame, player, enemy, x_offset=10, y_offset=70):
        bar_width = 200
        bar_height = 20
        gap = 52

        player_hp_ratio = player.hp_percentage
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + bar_width, y_offset + bar_height),
            (100, 100, 100),
            2,
        )
        cv2.rectangle(
            frame,
            (x_offset, y_offset),
            (x_offset + int(bar_width * player_hp_ratio), y_offset + bar_height),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame,
            f"Player HP: {player.hp}/{player.max_hp}",
            (x_offset, y_offset - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        enemy_y = y_offset + gap
        enemy_hp_ratio = enemy.hp_percentage
        cv2.rectangle(
            frame,
            (x_offset, enemy_y),
            (x_offset + bar_width, enemy_y + bar_height),
            (100, 100, 100),
            2,
        )
        cv2.rectangle(
            frame,
            (x_offset, enemy_y),
            (x_offset + int(bar_width * enemy_hp_ratio), enemy_y + bar_height),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            f"Enemy HP: {enemy.hp}/{enemy.max_hp}",
            (x_offset, enemy_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    @staticmethod
    def draw_game_state(frame, game_state, x_offset=10, y_offset=180):
        battle_state_names = {
            0: "Waiting",
            1: "Player turn",
            2: "Enemy turn",
            3: "Victory",
            4: "Defeat",
        }

        state = game_state["battle_state"].value
        state_name = battle_state_names.get(state, "Unknown")

        cv2.putText(
            frame,
            f"State: {state_name}",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if game_state["last_action"]:
            cv2.putText(
                frame,
                f"Last action: {game_state['last_action']}",
                (x_offset, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        if game_state["last_damage"] > 0:
            cv2.putText(
                frame,
                f"Damage: {game_state['last_damage']}",
                (x_offset, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return frame

    @staticmethod
    def draw_gesture_recognition(frame, gesture_info, x_offset=10, y_offset=265):
        gesture = gesture_info.get("gesture", "Unknown")
        confidence = gesture_info.get("confidence", 0.0)
        smoothed_gesture = gesture_info.get("smoothed_gesture", "Unknown")
        color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)

        cv2.putText(
            frame,
            f"Gesture: {gesture} ({confidence:.2f})",
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        cv2.putText(
            frame,
            f"Stable: {smoothed_gesture}",
            (x_offset, y_offset + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        return frame

    @staticmethod
    def draw_instructions(frame, game_state):
        height, width = frame.shape[:2]
        cv2.putText(
            frame,
            "VisionQuest - AR Gesture Battle",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "[SPACE] register/start | [R] reset | [Q] quit",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

        battle_state = game_state["battle_state"].value
        if battle_state == 3:
            cv2.putText(
                frame,
                "*** Victory ***",
                (width // 2 - 110, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        elif battle_state == 4:
            cv2.putText(
                frame,
                "*** Defeat ***",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        return frame
