import cv2
import numpy as np

from ar.ar_renderer import ARRenderer
from ar.plane_tracker import PlaneTracker
from game.battle_system import BattleState
from game.game_manager import GameManager
from network.frame_receiver import FrameReceiver
from network.websocket_server import WebSocketFrameServer
from ui.hud import HUD
from vision.gesture_detector import GestureDetector
from vision.hand_tracker import HandTracker


FRAME_STALE_SECONDS = 2.0
VISION_INTERVAL_FRAMES = 2
GESTURE_CONFIDENCE_THRESHOLD = 0.6


UNKNOWN_GESTURE = {
    "gesture": "Unknown",
    "confidence": 0.0,
    "smoothed_gesture": "Unknown",
}


EMPTY_HAND_DETECTION = {
    "left_hand": None,
    "right_hand": None,
    "handedness": [],
    "hand_landmarks": [],
}


def draw_wrapped_text(frame, text, origin, font_scale, color, thickness, max_chars=58, line_height=26):
    x, y = origin
    if not text:
        return
    for start in range(0, len(text), max_chars):
        line = text[start : start + max_chars]
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += line_height


def draw_waiting_frame(frame_receiver, page_url=None, qr_image=None):
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    age = frame_receiver.get_last_update_age()
    connected = frame_receiver.is_client_connected()

    if age is None:
        status = "Waiting for mobile camera..."
        detail = "Scan the QR code from your phone."
    elif connected:
        status = "Waiting for a fresh camera frame..."
        detail = f"Last frame received {age:.1f}s ago."
    else:
        status = "Mobile camera disconnected."
        detail = f"Last frame received {age:.1f}s ago."

    cv2.putText(frame, "VisionQuest Mobile Camera Setup", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, status, (30, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, detail, (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, "1. Connect phone and PC to the same network.", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, "2. Scan the QR code or type the URL below.", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(frame, "3. Allow camera access and wait for WebSocket: connected.", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if qr_image is not None:
        qr_size = 280
        qr_resized = cv2.resize(qr_image, (qr_size, qr_size), interpolation=cv2.INTER_AREA)
        y1 = 260
        x1 = 30
        frame[y1 : y1 + qr_size, x1 : x1 + qr_size] = qr_resized
        cv2.rectangle(frame, (x1, y1), (x1 + qr_size, y1 + qr_size), (255, 255, 255), 2)
    else:
        cv2.putText(frame, "QR code not available yet.", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    if page_url:
        cv2.putText(frame, "URL:", (340, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        draw_wrapped_text(frame, page_url, (340, 365), 0.52, (0, 255, 255), 1)
        cv2.putText(frame, "If the phone cannot open the page, check firewall and Wi-Fi isolation.", (340, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    cv2.putText(frame, "[Q] quit", (30, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return frame


def draw_runtime_diagnostics(frame, fps, hand_detection, plane_registered, game_started):
    height, width = frame.shape[:2]
    hand_count = len(hand_detection.get("hand_landmarks") or [])
    cv2.putText(
        frame,
        f"Camera: {width}x{height} | Hands: {hand_count} | FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    plane_text = "Plane registered" if plane_registered else "Press SPACE to register plane"
    game_text = "Game started" if game_started else "Game not started"
    cv2.putText(
        frame,
        f"{plane_text} | {game_text}",
        (10, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if plane_registered else (0, 0, 255),
        2,
    )


def choose_best_gesture(left_info, right_info):
    return right_info if right_info["confidence"] > left_info["confidence"] else left_info


def main():
    print("Starting VisionQuest...")

    frame_receiver = FrameReceiver()
    websocket_server = WebSocketFrameServer(frame_receiver)
    websocket_server.start()
    qr_image = None
    if websocket_server.qr_code_path is not None:
        qr_image = cv2.imread(str(websocket_server.qr_code_path))

    hand_tracker = HandTracker(max_num_hands=2)
    gesture_detector = GestureDetector()
    plane_tracker = PlaneTracker()
    ar_renderer = ARRenderer(plane_size=(400, 300))
    game_manager = GameManager()

    game_started = False
    plane_registered = False
    both_hands_counter = 0
    both_hands_threshold = 15
    require_open_palms = True

    frame_count = 0
    fps_clock = cv2.getTickCount()
    last_hand_detection = dict(EMPTY_HAND_DETECTION)
    last_gesture_info_left = dict(UNKNOWN_GESTURE)
    last_gesture_info_right = dict(UNKNOWN_GESTURE)
    last_gesture_info = dict(UNKNOWN_GESTURE)

    print("Ready.")
    print("Open the QR URL on your phone.")
    print("Controls: SPACE=register plane/start game, R=reset, Q=quit")

    try:
        while True:
            frame = frame_receiver.get_latest_frame()
            if frame is None or not frame_receiver.is_frame_fresh(FRAME_STALE_SECONDS):
                waiting_frame = draw_waiting_frame(frame_receiver, websocket_server.page_url, qr_image)
                cv2.imshow("VisionQuest", waiting_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - fps_clock, 1)
            fps_clock = cv2.getTickCount()

            if frame_count % VISION_INTERVAL_FRAMES == 0:
                last_hand_detection = hand_tracker.detect_hands(frame)
                last_gesture_info_left = gesture_detector.detect_gesture(last_hand_detection["left_hand"], "left")
                last_gesture_info_right = gesture_detector.detect_gesture(last_hand_detection["right_hand"], "right")
                last_gesture_info = choose_best_gesture(last_gesture_info_left, last_gesture_info_right)

            hand_detection = last_hand_detection
            gesture_info_left = last_gesture_info_left
            gesture_info_right = last_gesture_info_right
            gesture_info = last_gesture_info

            if plane_registered:
                tracking_result = plane_tracker.track_plane(frame)
                H = tracking_result["H"]
            else:
                H = None

            if plane_registered and not game_started:
                left_present = hand_detection.get("left_hand") is not None
                right_present = hand_detection.get("right_hand") is not None

                palms_ok = True
                if require_open_palms:
                    palms_ok = (
                        gesture_info_left.get("smoothed_gesture") == "Open_Palm"
                        and gesture_info_right.get("smoothed_gesture") == "Open_Palm"
                        and gesture_info_left.get("confidence", 0.0) >= GESTURE_CONFIDENCE_THRESHOLD
                        and gesture_info_right.get("confidence", 0.0) >= GESTURE_CONFIDENCE_THRESHOLD
                    )

                both_hands_counter = both_hands_counter + 1 if left_present and right_present and palms_ok else 0
                if both_hands_counter > 0:
                    cv2.putText(
                        frame,
                        f"Start pose: {both_hands_counter}/{both_hands_threshold}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                if both_hands_counter >= both_hands_threshold:
                    game_manager.start_game()
                    game_started = True
                    print("Game started by two-hand gesture.")

            if game_started:
                game_manager.process_gesture(gesture_info)
                game_manager.update()

            if H is not None and game_started:
                frame = ar_renderer.render_battlefield(
                    frame,
                    H,
                    game_manager.player_pos,
                    game_manager.enemy_pos,
                )

            frame = hand_tracker.draw_hands(frame, hand_detection)

            if game_started:
                game_state = game_manager.get_game_state()
                frame = HUD.draw_hp_bar(frame, game_manager.player, game_manager.enemy)
                frame = HUD.draw_game_state(frame, game_state)
                frame = HUD.draw_gesture_recognition(frame, gesture_info)

            current_state = game_manager.get_game_state() if game_started else {"battle_state": BattleState.WAITING}
            frame = HUD.draw_instructions(frame, current_state)
            draw_runtime_diagnostics(frame, fps, hand_detection, plane_registered, game_started)

            cv2.imshow("VisionQuest", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                if not plane_registered:
                    if plane_tracker.register_plane(frame):
                        plane_registered = True
                        print("Plane registered. Press SPACE again to start the game.")
                    else:
                        print("Plane registration failed. Try a more textured surface.")
                elif not game_started:
                    game_manager.start_game()
                    game_started = True
                    print("Game started.")

            if key == ord("r"):
                game_manager.reset_game()
                game_started = False
                plane_registered = False
                both_hands_counter = 0
                print("Game reset.")
    finally:
        websocket_server.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
