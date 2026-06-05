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
PLANE_PREVIEW_INTERVAL_FRAMES = 3
GESTURE_CONFIDENCE_THRESHOLD = 0.6
START_GESTURE = "OK_Sign"
START_GESTURE_HOLD_FRAMES = 12


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
    x = max(width - 470, 10)
    cv2.rectangle(frame, (x - 8, 8), (width - 8, 58), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Camera: {width}x{height} | Hands: {hand_count} | FPS: {fps:.1f}",
        (x, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    plane_text = "A4 board registered" if plane_registered else "Center an A4 sheet and press SPACE"
    game_text = "Game started" if game_started else "Game not started"
    cv2.rectangle(frame, (8, height - 76), (width - 8, height - 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{plane_text} | {game_text}",
        (16, height - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0) if plane_registered else (0, 0, 255),
        2,
    )


def draw_a4_detection_highlight(frame, tracking_result, plane_registered):
    if not tracking_result or not tracking_result.get("success"):
        cv2.rectangle(frame, (8, 88), (360, 124), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "A4 not detected",
            (16, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 120, 255),
            2,
            cv2.LINE_AA,
        )
        return

    corners = np.asarray(tracking_result["corners"], dtype=np.int32)
    overlay = frame.copy()
    stale = tracking_result.get("stale", False)
    if stale:
        color = (0, 165, 255)
    else:
        color = (0, 255, 80) if plane_registered else (0, 220, 255)
    cv2.fillConvexPoly(overlay, corners, color)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
    cv2.polylines(frame, [corners], True, color, 4, cv2.LINE_AA)

    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, corners):
        cv2.circle(frame, tuple(point), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (int(point[0]) + 7, int(point[1]) - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    marker_centers = tracking_result.get("marker_centers")
    if marker_centers is not None:
        for point in np.asarray(marker_centers, dtype=np.int32):
            cv2.circle(frame, tuple(point), 9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1, cv2.LINE_AA)

    if stale:
        text = "A4 temporarily occluded - holding last corners"
    elif tracking_result.get("tracking_method") == "corner_marks":
        text = "A4 corner marks detected"
    elif tracking_result.get("tracking_method") == "marker_track":
        score = tracking_result.get("track_score", 0.0)
        text = f"A4 marker tracking ({score:.2f})"
    elif tracking_result.get("tracking_method") == "patch_track":
        score = tracking_result.get("track_score", 0.0)
        text = f"A4 tracked from previous frame ({score:.2f})"
    elif plane_registered:
        text = "A4 detected / registered"
    else:
        text = "A4 detected - press SPACE"
    cv2.rectangle(frame, (8, 88), (560, 124), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


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
    ar_renderer = ARRenderer(plane_size=(210, 297))
    game_manager = GameManager()
    game_manager.player_pos = (105, 230)
    game_manager.enemy_pos = (105, 70)

    game_started = False
    plane_registered = False
    start_gesture_counter = 0

    frame_count = 0
    fps_clock = cv2.getTickCount()
    last_hand_detection = dict(EMPTY_HAND_DETECTION)
    last_gesture_info_left = dict(UNKNOWN_GESTURE)
    last_gesture_info_right = dict(UNKNOWN_GESTURE)
    last_gesture_info = dict(UNKNOWN_GESTURE)
    current_tracking_result = {"success": False, "H": None, "corners": None}

    print("Ready.")
    print("Open the QR URL on your phone.")
    print("Controls: SPACE=register centered A4 board/start fallback, R=reset, Q=quit")

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
            #frame = cv2.flip(frame, 1)
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

            should_update_plane = plane_registered or frame_count % PLANE_PREVIEW_INTERVAL_FRAMES == 0
            if should_update_plane:
                current_tracking_result = plane_tracker.track_plane(frame)
            H = current_tracking_result["H"] if current_tracking_result["success"] else None

            if plane_registered and not game_started:
                start_ok = (
                    gesture_info.get("smoothed_gesture") == START_GESTURE
                    and gesture_info.get("confidence", 0.0) >= GESTURE_CONFIDENCE_THRESHOLD
                )
                start_gesture_counter = start_gesture_counter + 1 if start_ok else 0
                cv2.rectangle(frame, (8, 42), (360, 82), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Show OK sign to start: {start_gesture_counter}/{START_GESTURE_HOLD_FRAMES}",
                    (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0) if start_ok else (0, 220, 255),
                    2,
                )

                if start_gesture_counter >= START_GESTURE_HOLD_FRAMES:
                    game_manager.start_game()
                    game_started = True
                    start_gesture_counter = 0
                    print("Game started by OK sign.")

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

            draw_a4_detection_highlight(frame, current_tracking_result, plane_registered)

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
                    if plane_tracker.register_tracking_result(current_tracking_result):
                        plane_registered = True
                        print("A4 board registered. Press SPACE again to start the game.")
                    else:
                        print("A4 board registration failed. Center a white A4 sheet in the camera view.")
                elif not game_started:
                    game_manager.start_game()
                    game_started = True
                    start_gesture_counter = 0
                    print("Game started by keyboard fallback.")

            if key == ord("r"):
                game_manager.reset_game()
                game_started = False
                plane_registered = False
                start_gesture_counter = 0
                print("Game reset.")
    finally:
        websocket_server.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
