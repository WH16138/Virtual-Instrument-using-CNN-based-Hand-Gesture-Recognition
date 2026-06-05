import time
from pathlib import Path

import cv2
import numpy as np

from ar.homography import HomographyEstimator
from game.battle_system import BattleState


class ActionCardRenderer:
    """AR-space action card and short combat effect renderer."""

    ACTIONS = [
        {
            "action": "Strike",
            "gesture": "Fist",
            "label": "STRIKE",
            "path": Path("assets") / "cards" / "strike.png",
            "color": (70, 150, 255),
        },
        {
            "action": "Guard",
            "gesture": "Open_Palm",
            "label": "GUARD",
            "path": Path("assets") / "cards" / "guard.png",
            "color": (80, 220, 255),
        },
        {
            "action": "Shot",
            "gestures": {"V_Sign", "Gun_Sign"},
            "label": "SHOT",
            "path": Path("assets") / "cards" / "shot.png",
            "color": (255, 120, 230),
        },
    ]

    ENEMY_CARD_PATHS = {
        "Attack": Path("assets") / "cards" / "enemy_attack.png",
        "Defend": Path("assets") / "cards" / "enemy_defend.png",
        "Skill": Path("assets") / "cards" / "enemy_skill.png",
    }

    def __init__(self):
        self.card_cache = {}
        self.projectiles = []
        self.guard_effects = []
        self.seen_event_ids = set()

    def reset(self):
        self.projectiles.clear()
        self.guard_effects.clear()
        self.seen_event_ids.clear()

    def draw(self, frame, H, plane_size, game_state, events, enemy_pos):
        if H is None or not game_state:
            return frame

        selection = game_state.get("action_selection", {})
        battle_state = game_state.get("battle_state")
        can_select = battle_state == BattleState.PLAYER_TURN and game_state.get("can_act", False)
        slots = self._card_slots(plane_size, selection if can_select else {})

        if can_select:
            for action in self.ACTIONS:
                slot = slots[action["action"]]
                self._draw_board_card(frame, H, slot, action, selection)

        self._add_effects_from_events(events, slots, enemy_pos)
        self._draw_enemy_preview(frame, H, game_state, enemy_pos)
        self._draw_effects(frame, H)
        return frame

    def _card_slots(self, plane_size, selection):
        plane_width, plane_height = float(plane_size[0]), float(plane_size[1])
        card_w = min(56.0, plane_width * 0.24)
        card_h = card_w * 1.34
        gap = card_w * 0.20
        total_w = card_w * 3 + gap * 2
        start_x = plane_width * 0.5 - total_w * 0.5
        y = plane_height + card_h * 0.18
        selected_action = selection.get("action") if selection.get("active") else None

        slots = {}
        for index, action in enumerate(self.ACTIONS):
            scale = 1.14 if action["action"] == selected_action else 1.0
            width = card_w * scale
            height = card_h * scale
            center_x = start_x + card_w * 0.5 + index * (card_w + gap)
            center_y = y + card_h * 0.5
            slots[action["action"]] = {
                "center": (center_x, center_y),
                "rect": [
                    (center_x - width * 0.5, center_y - height * 0.5),
                    (center_x + width * 0.5, center_y - height * 0.5),
                    (center_x + width * 0.5, center_y + height * 0.5),
                    (center_x - width * 0.5, center_y + height * 0.5),
                ],
                "color": action["color"],
            }
        return slots

    def _draw_board_card(self, frame, H, slot, action, selection):
        image = self._load_card(action["path"], action["label"], action["color"])
        dst = self._project_points(slot["rect"], H)
        if dst is None:
            return

        self._warp_rgba(frame, image, dst)
        selected = selection.get("active") and selection.get("action") == action["action"]
        border = (255, 245, 120) if selected else (160, 150, 120)
        thickness = 3 if selected else 1
        cv2.polylines(frame, [dst.astype(np.int32)], True, border, thickness, cv2.LINE_AA)

        if selected:
            self._draw_progress_on_card(frame, dst, float(selection.get("progress", 0.0)), action["color"])

    def _draw_progress_on_card(self, frame, quad, progress, color):
        progress = max(0.0, min(1.0, progress))
        left = quad[3].astype(np.float32)
        right = quad[2].astype(np.float32)
        end = left + (right - left) * progress
        cv2.line(frame, tuple(left.astype(int)), tuple(right.astype(int)), (45, 45, 55), 5, cv2.LINE_AA)
        cv2.line(frame, tuple(left.astype(int)), tuple(end.astype(int)), color, 5, cv2.LINE_AA)

    def _draw_enemy_preview(self, frame, H, game_state, enemy_pos):
        preview = game_state.get("enemy_preview", {})
        if not preview.get("active"):
            return

        action = preview.get("action")
        point = HomographyEstimator.transform_point(enemy_pos, H)
        if point is None:
            return

        image = self._load_card(self.ENEMY_CARD_PATHS.get(action), str(action or "ACTION").upper(), (80, 80, 255))
        center = (int(point[0]), int(point[1] - 120))
        card_w, card_h = 92, 124
        x1 = center[0] - card_w // 2
        y1 = center[1] - card_h // 2
        dst = np.asarray(
            [
                [x1, y1],
                [x1 + card_w, y1],
                [x1 + card_w, y1 + card_h],
                [x1, y1 + card_h],
            ],
            dtype=np.float32,
        )
        self._warp_rgba(frame, image, dst)
        cv2.polylines(frame, [dst.astype(np.int32)], True, (90, 120, 255), 2, cv2.LINE_AA)
        self._draw_progress_on_card(frame, dst, float(preview.get("progress", 0.0)), (90, 120, 255))

    def _add_effects_from_events(self, events, slots, enemy_pos):
        now = time.monotonic()
        for event in events:
            event_id = (event.get("time"), event.get("source"), event.get("kind"), event.get("target"), event.get("label"))
            if event_id in self.seen_event_ids:
                continue
            self.seen_event_ids.add(event_id)

            if event.get("source") != "player":
                continue

            kind = event.get("kind")
            if kind in ("Strike", "Shot"):
                start = slots.get(kind, {}).get("center")
                if start is None:
                    continue
                self.projectiles.append(
                    {
                        "start": start,
                        "end": enemy_pos,
                        "kind": kind,
                        "start_time": now,
                        "duration": 0.55 if kind == "Shot" else 0.42,
                        "color": (255, 120, 230) if kind == "Shot" else (70, 150, 255),
                    }
                )
            elif kind == "Guard":
                center = slots.get("Guard", {}).get("center")
                if center is not None:
                    self.guard_effects.append({"center": center, "start_time": now, "duration": 0.85})

    def _draw_effects(self, frame, H):
        now = time.monotonic()
        kept_projectiles = []
        for projectile in self.projectiles:
            age = now - projectile["start_time"]
            if age > projectile["duration"]:
                continue
            progress = age / projectile["duration"]
            start = np.asarray(HomographyEstimator.transform_point(projectile["start"], H), dtype=np.float32)
            end = np.asarray(HomographyEstimator.transform_point(projectile["end"], H), dtype=np.float32)
            if start.shape != (2,) or end.shape != (2,):
                continue
            pos = start + (end - start) * progress
            radius = 8 if projectile["kind"] == "Shot" else 6
            cv2.circle(frame, tuple(pos.astype(int)), radius, projectile["color"], -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pos.astype(int)), radius + 4, (255, 255, 255), 1, cv2.LINE_AA)
            kept_projectiles.append(projectile)
        self.projectiles = kept_projectiles

        kept_guards = []
        for guard in self.guard_effects:
            age = now - guard["start_time"]
            if age > guard["duration"]:
                continue
            progress = age / guard["duration"]
            center = HomographyEstimator.transform_point(guard["center"], H)
            if center is None:
                continue
            radius = int(18 + progress * 34)
            alpha_color = int(255 * (1.0 - progress))
            cv2.circle(frame, tuple(np.asarray(center, dtype=int)), radius, (80, 220, alpha_color), 2, cv2.LINE_AA)
            kept_guards.append(guard)
        self.guard_effects = kept_guards

    def _load_card(self, path, label, color):
        key = str(path) if path else f"fallback:{label}"
        if key in self.card_cache:
            return self.card_cache[key]

        image = None
        if path and Path(path).exists():
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            image = self._fallback_card(label, color)
        elif image.shape[2] == 3:
            alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
            image = np.concatenate([image, alpha], axis=2)

        self.card_cache[key] = image
        return image

    def _fallback_card(self, label, color):
        width, height = 180, 244
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[:, :, 3] = 218
        image[:, :, :3] = (28, 25, 34)
        cv2.rectangle(image, (7, 7), (width - 8, height - 8), color, 3, cv2.LINE_AA)
        cv2.rectangle(image, (22, 26), (width - 23, 130), tuple(max(0, int(c * 0.55)) for c in color), -1)
        cv2.putText(image, label[:10], (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 240, 225, 255), 2, cv2.LINE_AA)
        return image

    def _project_points(self, points, H):
        projected = [HomographyEstimator.transform_point(point, H) for point in points]
        if any(point is None for point in projected):
            return None
        return np.asarray(projected, dtype=np.float32)

    def _warp_rgba(self, frame, rgba, dst):
        src_h, src_w = rgba.shape[:2]
        src = np.asarray([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        warped = cv2.warpPerspective(rgba, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        if np.max(alpha) <= 0.0:
            return
        color = warped[:, :, :3].astype(np.float32)
        blended = color * alpha + frame.astype(np.float32) * (1.0 - alpha)
        np.copyto(frame, blended.astype(np.uint8))
