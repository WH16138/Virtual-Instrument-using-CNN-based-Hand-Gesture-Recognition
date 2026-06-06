import cv2
import numpy as np
import time

from ar.ar_renderer import ARRenderer
from ar.homography import HomographyEstimator
from ar.plane_tracker import PlaneTracker
from game.game_manager import GameManager
from network.frame_receiver import FrameReceiver
from network.websocket_server import WebSocketFrameServer
from ui.action_cards import ActionCardRenderer
from ui.damage_text import FloatingTextManager
from ui.hud import HUD
from vision.gesture_detector import GestureDetector
from vision.hand_tracker import HandTracker


FRAME_STALE_SECONDS = 5.0
FRAME_STALE_GRACE_SECONDS = 2.5
VISION_INTERVAL_FRAMES = 2
PRE_REGISTRATION_VISION_INTERVAL_FRAMES = 2
PLANE_PREVIEW_INTERVAL_FRAMES = 4
HAND_DETECTION_MAX_DIM = 640
GESTURE_CONFIDENCE_THRESHOLD = 0.6
SETUP_GESTURE = "OK_Sign"
SETUP_GESTURE_HOLD_SECONDS = 2.0
WINDOW_NAME = "VisionQuest"
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 860
MOBILE_PREVIEW_WIDTH = 960


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


def configure_display_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)


def get_display_size(window_name):
    try:
        _, _, width, height = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return DISPLAY_WIDTH, DISPLAY_HEIGHT
    if width <= 0 or height <= 0:
        return DISPLAY_WIDTH, DISPLAY_HEIGHT
    return width, height


def display_frame_layout(frame_shape, target_width=None, target_height=None):
    target_width = DISPLAY_WIDTH if target_width is None else int(target_width)
    target_height = DISPLAY_HEIGHT if target_height is None else int(target_height)
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0 or target_width <= 0 or target_height <= 0:
        return 1.0, 0, 0, frame_width, frame_height, target_width, target_height

    scale = min(target_width / float(frame_width), target_height / float(frame_height))
    resized_width = int(round(frame_width * scale))
    resized_height = int(round(frame_height * scale))
    x1 = max((target_width - resized_width) // 2, 0)
    y1 = max((target_height - resized_height) // 2, 0)
    return scale, x1, y1, resized_width, resized_height, target_width, target_height


def prepare_display_frame(frame, target_width=None, target_height=None):
    scale, x1, y1, resized_width, resized_height, target_width, target_height = display_frame_layout(
        frame.shape,
        target_width,
        target_height,
    )
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= 0 or frame_height <= 0 or target_width <= 0 or target_height <= 0:
        return frame

    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[y1 : y1 + resized_height, x1 : x1 + resized_width] = resized
    return canvas


def homography_for_display(H, frame_shape, target_width, target_height):
    if H is None:
        return None
    scale, x1, y1, _, _, _, _ = display_frame_layout(frame_shape, target_width, target_height)
    transform = np.asarray(
        [
            [scale, 0.0, float(x1)],
            [0.0, scale, float(y1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform @ H


def scaled_frame_and_homography(frame, H, target_width):
    frame_height, frame_width = frame.shape[:2]
    target_width = int(target_width)
    if frame_width <= 0 or frame_height <= 0 or target_width <= 0:
        return frame, H

    scale = target_width / float(frame_width)
    if abs(scale - 1.0) < 1e-3:
        return frame.copy(), None if H is None else H.copy()

    target_height = max(1, int(round(frame_height * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
    if H is None:
        return scaled, None

    transform = np.asarray(
        [
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scaled, transform @ H


def draw_sharp_ui_overlay(
    target_frame,
    target_H,
    game_started,
    game_state,
    events,
    gesture_info,
    action_cards,
    floating_text,
    ar_renderer,
    game_manager,
    setup_hold_progress,
    plane_registered,
    debug_mode,
    fps,
    hand_detection,
):
    if target_H is not None and game_started:
        target_frame = action_cards.draw(
            target_frame,
            target_H,
            (ar_renderer.plane_width, ar_renderer.plane_height),
            game_state,
            events,
            game_manager.enemy_pos,
            gesture_info=gesture_info,
        )
        target_frame = floating_text.draw(target_frame, target_H, HomographyEstimator.transform_point)

    target_frame = HUD.draw_game_layer(
        target_frame,
        game_state,
        gesture_info,
        plane_registered,
        game_started,
        setup_hold_progress,
    )
    if debug_mode or not plane_registered:
        draw_runtime_diagnostics(target_frame, fps, hand_detection, plane_registered, game_started)
    return target_frame


def resize_for_vision(frame, max_dim):
    height, width = frame.shape[:2]
    max_current = max(width, height)
    if max_current <= max_dim:
        return frame
    scale = float(max_dim) / float(max_current)
    return cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )

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
    x = max(width - 390, 10)
    cv2.rectangle(frame, (x - 8, height - 64), (width - 8, height - 10), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{width}x{height} | Hands {hand_count} | FPS {fps:.1f}",
        (x, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0),
        1,
    )

    plane_text = "Gate board registered" if plane_registered else "Show the gate marker and hold OK"
    game_text = "Game started" if game_started else "Game not started"
    cv2.putText(
        frame,
        f"{plane_text} | {game_text}",
        (x, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0) if plane_registered else (0, 0, 255),
        1,
    )


def draw_polygon_progress(frame, points, progress, color, thickness=7):
    if progress <= 0.0:
        return
    points = np.asarray(points, dtype=np.int32)
    if points.shape != (4, 2):
        return
    progress = max(0.0, min(1.0, float(progress)))
    segments = list(zip(points, np.roll(points, -1, axis=0)))
    lengths = [float(np.linalg.norm(end - start)) for start, end in segments]
    remaining = sum(lengths) * progress
    for (start, end), length in zip(segments, lengths):
        if remaining <= 0.0:
            break
        if remaining >= length:
            cv2.line(frame, tuple(start), tuple(end), color, thickness, cv2.LINE_AA)
            remaining -= length
            continue
        ratio = remaining / max(length, 1e-6)
        partial_end = start.astype(np.float32) + (end - start).astype(np.float32) * ratio
        cv2.line(frame, tuple(start), tuple(np.round(partial_end).astype(np.int32)), color, thickness, cv2.LINE_AA)
        break


def draw_a4_detection_highlight(frame, tracking_result, plane_registered, setup_hold_progress=0.0):
    if not tracking_result or not tracking_result.get("success"):
        marker_candidates = (tracking_result or {}).get("marker_candidates") or []
        for candidate in marker_candidates:
            point = np.asarray(candidate.get("point"), dtype=np.int32)
            if point.shape != (2,):
                continue
            color = (0, 0, 255) if candidate.get("accepted") else (0, 165, 255)
            if candidate.get("stale_debug"):
                color = tuple(int(channel * 0.45) for channel in color)
            cv2.circle(frame, tuple(point), 7, color, 2, cv2.LINE_AA)
            open_vector = np.asarray(candidate.get("open_vector"), dtype=np.float32)
            if candidate.get("accepted") and open_vector.shape == (2,):
                norm = float(np.linalg.norm(open_vector))
                if norm > 1e-6:
                    direction = open_vector / norm
                    arrow_end = point.astype(np.float32) + direction * 28.0
                    cv2.arrowedLine(
                        frame,
                        tuple(point),
                        tuple(np.round(arrow_end).astype(np.int32)),
                        color,
                        2,
                        cv2.LINE_AA,
                        tipLength=0.35,
                    )
            slot = candidate.get("slot", "")
            if slot:
                cv2.putText(
                    frame,
                    slot.upper(),
                    (int(point[0]) + 8, int(point[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        selected_markers = (tracking_result or {}).get("selected_marker_centers")
        if selected_markers is not None:
            selected_points = np.asarray(selected_markers, dtype=np.int32)
            if selected_points.shape == (4, 2):
                cv2.polylines(frame, [selected_points], True, (0, 255, 255), 2, cv2.LINE_AA)
                for label, point in zip(("TL", "TR", "BR", "BL"), selected_points):
                    cv2.circle(frame, tuple(point), 9, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(
                        frame,
                        label,
                        (int(point[0]) + 9, int(point[1]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        cv2.rectangle(frame, (8, 88), (360, 124), (0, 0, 0), -1)
        reject_reason = (tracking_result or {}).get("reject_reason")
        detail = f" | {reject_reason}" if reject_reason else ""
        candidate_count = (tracking_result or {}).get("candidate_count", len(marker_candidates))
        cv2.putText(
            frame,
            (f"Gate marker not detected | candidates: {candidate_count}{detail}" if (tracking_result or {}).get("detector_mode") == "door_marker" else f"A4 not detected | markers: {candidate_count}{detail}"),
            (16, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
        return

    corners = np.asarray(tracking_result["corners"], dtype=np.int32)
    overlay = frame.copy()
    stale = tracking_result.get("stale", False)
    registering = (not plane_registered) and setup_hold_progress > 0.0
    if stale:
        color = (0, 165, 255)
    elif registering:
        color = (40, 230, 255)
    else:
        color = (0, 255, 80) if plane_registered else (0, 220, 255)
    cv2.fillConvexPoly(overlay, corners, color)
    fill_alpha = 0.24 if registering else 0.16
    cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

    if registering:
        pulse = 0.5 + 0.5 * np.sin(time.monotonic() * 8.0)
        glow_color = (
            int(35 + 45 * pulse),
            int(190 + 55 * pulse),
            255,
        )
        for thickness, alpha in ((18, 0.20), (11, 0.28)):
            glow = frame.copy()
            cv2.polylines(glow, [corners], True, glow_color, thickness, cv2.LINE_AA)
            cv2.addWeighted(glow, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.polylines(frame, [corners], True, (30, 245, 255), 5, cv2.LINE_AA)
        draw_polygon_progress(frame, corners, setup_hold_progress, (255, 255, 255), 9)

        bx, by, bw, _ = cv2.boundingRect(corners)
        label = f"OK REGISTERING {int(setup_hold_progress * 100):02d}%"
        label_x = max(10, min(frame.shape[1] - 265, bx + bw // 2 - 132))
        label_y = max(38, by - 14)
        cv2.rectangle(frame, (label_x - 8, label_y - 25), (label_x + 265, label_y + 8), (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    else:
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
    marker_observed = tracking_result.get("marker_observed")
    if marker_centers is not None:
        marker_points = np.asarray(marker_centers, dtype=np.int32)
        if marker_observed is None:
            marker_observed = np.ones(len(marker_points), dtype=bool)
        else:
            marker_observed = np.asarray(marker_observed, dtype=bool)
        for point, observed in zip(marker_points, marker_observed):
            marker_color = (0, 0, 255) if observed else (0, 165, 255)
            if observed:
                cv2.circle(frame, tuple(point), 9, marker_color, 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.line(frame, (point[0] - 6, point[1] - 6), (point[0] + 6, point[1] + 6), marker_color, 1, cv2.LINE_AA)
                cv2.line(frame, (point[0] - 6, point[1] + 6), (point[0] + 6, point[1] - 6), marker_color, 1, cv2.LINE_AA)

    confidence = tracking_result.get("homography_confidence", tracking_result.get("track_score", 0.0))
    reprojection_error = tracking_result.get("reprojection_error")
    error_text = f", err {reprojection_error:.1f}px" if reprojection_error is not None else ""
    plane_size = tracking_result.get("plane_size")
    size_text = ""
    if plane_size is not None:
        size_text = f", {float(plane_size[0]):.0f}x{float(plane_size[1]):.0f}mm"
    matched_points = tracking_result.get("matched_points", 0)
    if stale:
        text = "Board temporarily occluded - holding last pose"
    elif tracking_result.get("tracking_method") in ("door_marker", "door_redetect"):
        symbol_score = tracking_result.get("door_symbol_score")
        direction_score = tracking_result.get("door_direction_score")
        symbol_text = f", symbol {symbol_score:.2f}" if symbol_score is not None else ""
        direction_text = f", direction {direction_score:.2f}" if direction_score is not None else ""
        if not plane_registered and setup_hold_progress > 0.0:
            text = f"OK detected - registering gate board {int(setup_hold_progress * 100):02d}%"
        else:
            text = f"Gate board detected (H {confidence:.2f}{symbol_text}{direction_text}{size_text})"
    elif tracking_result.get("tracking_method") == "door_flow":
        text = f"Gate board tracking ({matched_points} points, H {confidence:.2f}{error_text})"
    elif tracking_result.get("tracking_method") in ("corner_marks", "corner_marks_partial"):
        white_score = tracking_result.get("white_validation_score")
        white_text = f", white {white_score:.2f}" if white_score is not None else ""
        text = f"A4 L markers detected ({matched_points}/4, H {confidence:.2f}{white_text}{error_text}{size_text})"
    elif tracking_result.get("tracking_method") == "white_boundary_marker_assist":
        text = f"A4 boundary + markers ({matched_points}/4, H {confidence:.2f}{size_text})"
    elif tracking_result.get("tracking_method") == "redetect_corner_marks":
        text = f"A4 re-detected ({matched_points}/4, H {confidence:.2f}{error_text}{size_text})"
    elif tracking_result.get("tracking_method") == "marker_track":
        score = tracking_result.get("track_score", 0.0)
        text = f"A4 marker tracking ({score:.2f}, H {confidence:.2f})"
    elif tracking_result.get("tracking_method") == "optical_flow":
        score = tracking_result.get("track_score", 0.0)
        text = f"A4 optical flow tracking ({matched_points}/4, H {confidence:.2f}{error_text})"
    elif tracking_result.get("tracking_method") == "patch_track":
        score = tracking_result.get("track_score", 0.0)
        text = f"A4 tracked from previous frame ({score:.2f}, H {confidence:.2f})"
    elif plane_registered:
        text = "Board detected / registered"
    else:
        text = "Gate board detected - hold OK"
    cv2.rectangle(frame, (8, 88), (560, 124), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def choose_best_gesture(left_info, right_info):
    return right_info if right_info["confidence"] > left_info["confidence"] else left_info


def mirror_detection_to_original_frame(detection_result):
    """Map hand landmarks detected on a horizontally flipped frame back to the display frame."""
    if not detection_result:
        return dict(EMPTY_HAND_DETECTION)

    def mirror_landmarks(landmarks):
        if landmarks is None:
            return None
        mirrored = np.asarray(landmarks, dtype=np.float32).copy()
        if mirrored.ndim == 2 and mirrored.shape[1] >= 2:
            mirrored[:, 0] = 1.0 - mirrored[:, 0]
        return mirrored

    return {
        "left_hand": mirror_landmarks(detection_result.get("left_hand")),
        "right_hand": mirror_landmarks(detection_result.get("right_hand")),
        "handedness": list(detection_result.get("handedness") or []),
        "hand_landmarks": [
            mirrored for mirrored in (mirror_landmarks(item) for item in detection_result.get("hand_landmarks") or [])
            if mirrored is not None
        ],
    }

def should_show_tracking_attention(tracking_result):
    if not tracking_result or not tracking_result.get("success"):
        return True
    if tracking_result.get("stale", False):
        return True
    if int(tracking_result.get("matched_points", 0) or 0) < 4:
        return True

    marker_observed = tracking_result.get("marker_observed")
    if marker_observed is not None and not bool(np.all(np.asarray(marker_observed, dtype=bool))):
        return True

    confidence = float(tracking_result.get("homography_confidence", tracking_result.get("track_score", 1.0)) or 0.0)
    if confidence < 0.55:
        return True

    reprojection_error = tracking_result.get("reprojection_error")
    if reprojection_error is not None and float(reprojection_error) > 8.0:
        return True

    return False


def is_setup_ok_gesture(gesture_info):
    return (
        gesture_info.get("smoothed_gesture") == SETUP_GESTURE
        and gesture_info.get("confidence", 0.0) >= GESTURE_CONFIDENCE_THRESHOLD
    )


def collect_game_model_paths(game_manager):
    model_paths = set()
    for enemy_type in game_manager.wave_manager.enemy_types:
        for attr in ("model_path", "ground_model_path"):
            model_path = getattr(enemy_type, attr, None)
            if model_path:
                model_paths.add(model_path)
    return sorted(model_paths)



def collect_ground_model_paths(game_manager):
    model_paths = set()
    for enemy_type in game_manager.wave_manager.enemy_types:
        model_path = getattr(enemy_type, "ground_model_path", None)
        if model_path:
            model_paths.add(model_path)
    return sorted(model_paths)

def main():
    print("Starting VisionQuest...")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    configure_display_window(WINDOW_NAME)

    frame_receiver = FrameReceiver()
    websocket_server = WebSocketFrameServer(frame_receiver)
    websocket_server.start()
    qr_image = None
    if websocket_server.qr_code_path is not None:
        qr_image = cv2.imread(str(websocket_server.qr_code_path))

    hand_tracker = HandTracker(max_num_hands=2)
    gesture_detector = GestureDetector()
    plane_tracker = PlaneTracker()
    ar_renderer = ARRenderer(plane_size=(150, 150))
    game_manager = GameManager()
    floating_text = FloatingTextManager()
    action_cards = ActionCardRenderer()
    game_manager.player_pos = (105, 230)
    game_manager.enemy_pos = (105, 70)
    ar_renderer.preload_models(collect_game_model_paths(game_manager))
    ar_renderer.prewarm_ground_textures(collect_ground_model_paths(game_manager))

    game_started = False
    plane_registered = False
    debug_mode = False
    setup_gesture_started_at = None
    viewport_prepared = False
    last_frame_shape = None

    frame_count = 0
    fps_clock = cv2.getTickCount()
    last_hand_detection = dict(EMPTY_HAND_DETECTION)
    last_gesture_info_left = dict(UNKNOWN_GESTURE)
    last_gesture_info_right = dict(UNKNOWN_GESTURE)
    last_gesture_info = dict(UNKNOWN_GESTURE)
    current_tracking_result = {"success": False, "H": None, "corners": None}
    freshness_grace_until = 0.0

    print("Ready.")
    print("Open the QR URL on your phone.")
    print("Controls: hold OK sign=register gate board/start, D=debug overlays, R=reset, Q=quit")

    try:
        while True:
            now = time.monotonic()
            frame = frame_receiver.get_latest_frame()
            frame_is_fresh = frame_receiver.is_frame_fresh(FRAME_STALE_SECONDS)
            in_freshness_grace = frame is not None and now < freshness_grace_until
            if frame is None or (not frame_is_fresh and not in_freshness_grace):
                waiting_frame = draw_waiting_frame(frame_receiver, websocket_server.page_url, qr_image)
                websocket_server.publish_rendered_frame(waiting_frame)
                display_width, display_height = get_display_size(WINDOW_NAME)
                cv2.imshow(WINDOW_NAME, prepare_display_frame(waiting_frame, display_width, display_height))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame_count += 1
            current_frame_shape = frame.shape[:2]
            if last_frame_shape is not None and last_frame_shape != current_frame_shape:
                viewport_prepared = False
            last_frame_shape = current_frame_shape
            #frame = cv2.flip(frame, 1)
            fps = cv2.getTickFrequency() / max(cv2.getTickCount() - fps_clock, 1)
            fps_clock = cv2.getTickCount()

            vision_interval = VISION_INTERVAL_FRAMES if plane_registered else PRE_REGISTRATION_VISION_INTERVAL_FRAMES
            if frame_count % vision_interval == 0:
                recognition_frame = cv2.flip(frame, 1)
                vision_frame = resize_for_vision(recognition_frame, HAND_DETECTION_MAX_DIM)
                recognition_hand_detection = hand_tracker.detect_hands(vision_frame)
                last_gesture_info_left = gesture_detector.detect_gesture(recognition_hand_detection["left_hand"], "left")
                last_gesture_info_right = gesture_detector.detect_gesture(recognition_hand_detection["right_hand"], "right")
                last_gesture_info = choose_best_gesture(last_gesture_info_left, last_gesture_info_right)
                last_hand_detection = mirror_detection_to_original_frame(recognition_hand_detection)
            hand_detection = last_hand_detection
            gesture_info = last_gesture_info

            should_update_plane = plane_registered or frame_count % PLANE_PREVIEW_INTERVAL_FRAMES == 0
            if should_update_plane:
                current_tracking_result = plane_tracker.track_plane(
                    frame,
                    hand_landmarks=hand_detection.get("hand_landmarks"),
                    debug=debug_mode or not plane_registered,
                )
            H = current_tracking_result["H"] if current_tracking_result["success"] else None
            if H is not None:
                plane_size = current_tracking_result.get("plane_size")
                if plane_size is not None:
                    ar_renderer.set_plane_size(plane_size)
                    game_manager.player_pos = (ar_renderer.plane_width * 0.50, ar_renderer.plane_height * 0.79)
                    game_manager.enemy_pos = (ar_renderer.plane_width * 0.50, ar_renderer.plane_height * 0.44)
                if not viewport_prepared:
                    viewport_prepared = ar_renderer.prepare_viewport(frame.shape)

            if not game_started:
                setup_ok = is_setup_ok_gesture(gesture_info)
                can_complete_setup = current_tracking_result.get("success") and H is not None
                if setup_ok and can_complete_setup:
                    if setup_gesture_started_at is None:
                        setup_gesture_started_at = now
                    setup_hold_elapsed = now - setup_gesture_started_at
                else:
                    setup_gesture_started_at = None
                    setup_hold_elapsed = 0.0

                if setup_hold_elapsed >= SETUP_GESTURE_HOLD_SECONDS:
                    if not plane_registered and not plane_tracker.register_tracking_result(current_tracking_result):
                        setup_gesture_started_at = None
                        print("Gate board registration failed. Show the full square gate marker in the camera view.")
                    else:
                        plane_registered = True
                        game_manager.start_game()
                        game_started = True
                        floating_text.reset()
                        action_cards.reset()
                        setup_gesture_started_at = None
                        freshness_grace_until = time.monotonic() + FRAME_STALE_GRACE_SECONDS
                        print("Gate board registered and game started by OK sign.")
            if game_started:
                action_performed = game_manager.process_gesture(gesture_info)
                if action_performed:
                    gesture_detector.reset()
                    last_gesture_info_left = dict(UNKNOWN_GESTURE)
                    last_gesture_info_right = dict(UNKNOWN_GESTURE)
                    last_gesture_info = dict(UNKNOWN_GESTURE)
                    gesture_info = last_gesture_info
                game_manager.update()

            game_state = game_manager.get_game_state()
            events = game_manager.consume_events() if game_started else []
            player_feedback_pos = (
                (ar_renderer.plane_width * 0.34, ar_renderer.plane_height + 31.0)
                if H is not None and game_started
                else game_manager.player_pos
            )
            floating_text.add_from_events(events, game_manager.player_pos, game_manager.enemy_pos, player_feedback_pos)
            tracking_needs_attention = should_show_tracking_attention(current_tracking_result)
            setup_hold_progress = 0.0
            if setup_gesture_started_at is not None and not game_started:
                setup_hold_progress = min(1.0, (now - setup_gesture_started_at) / SETUP_GESTURE_HOLD_SECONDS)

            if H is not None and game_started:
                frame = ar_renderer.render_battlefield(
                    frame,
                    H,
                    game_manager.player_pos,
                    game_manager.enemy_pos,
                    game_state=game_state,
                    show_floor_mesh=debug_mode or tracking_needs_attention,
                )

            show_tracking_overlay = (
                debug_mode
                or not plane_registered
                or tracking_needs_attention
            )
            if show_tracking_overlay:
                draw_a4_detection_highlight(frame, current_tracking_result, plane_registered, setup_hold_progress)

            if debug_mode or not game_started:
                frame = hand_tracker.draw_hands(frame, hand_detection)

            preview_frame, preview_H = scaled_frame_and_homography(frame, H, MOBILE_PREVIEW_WIDTH)
            preview_frame = draw_sharp_ui_overlay(
                preview_frame,
                preview_H,
                game_started,
                game_state,
                events,
                gesture_info,
                action_cards,
                floating_text,
                ar_renderer,
                game_manager,
                setup_hold_progress,
                plane_registered,
                debug_mode,
                fps,
                hand_detection,
            )
            websocket_server.publish_rendered_frame(preview_frame)

            display_width, display_height = get_display_size(WINDOW_NAME)
            display_frame = prepare_display_frame(frame, display_width, display_height)
            display_H = homography_for_display(H, frame.shape, display_width, display_height)
            display_frame = draw_sharp_ui_overlay(
                display_frame,
                display_H,
                game_started,
                game_state,
                events,
                gesture_info,
                action_cards,
                floating_text,
                ar_renderer,
                game_manager,
                setup_hold_progress,
                plane_registered,
                debug_mode,
                fps,
                hand_detection,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("d"):
                debug_mode = not debug_mode
                print(f"Debug overlays {'enabled' if debug_mode else 'disabled'}.")

            if key == ord("r"):
                game_manager.reset_game()
                plane_tracker = PlaneTracker()
                floating_text.reset()
                action_cards.reset()
                game_started = False
                plane_registered = False
                debug_mode = False
                setup_gesture_started_at = None
                viewport_prepared = False
                last_frame_shape = None
                freshness_grace_until = 0.0
                print("Game reset.")
    finally:
        ar_renderer.close()
        websocket_server.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
