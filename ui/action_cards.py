import time
from pathlib import Path

import cv2
import numpy as np

from ar.homography import HomographyEstimator
from game.battle_system import BattleState

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


class ActionCardRenderer:
    """AR-space action cards, attached status panels, and combat effects."""

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
        self.panel_cache = {}
        self.font_cache = {}
        self.projectiles = []
        self.guard_effects = []
        self.seen_event_ids = set()

    def reset(self):
        self.projectiles.clear()
        self.guard_effects.clear()
        self.seen_event_ids.clear()

    def draw(self, frame, H, plane_size, game_state, events, enemy_pos, gesture_info=None):
        if H is None or not game_state:
            return frame

        selection = game_state.get("action_selection", {})
        reward_selection = game_state.get("reward_selection", {})
        battle_state = game_state.get("battle_state")
        round_reveal = game_state.get("round_reveal", {})
        reveal_active = bool(round_reveal.get("active"))
        reward_active = battle_state == BattleState.REWARD_SELECT and bool(reward_selection.get("active"))
        can_select = battle_state == BattleState.PLAYER_TURN and game_state.get("can_act", False)
        reveal_selection = {
            "active": reveal_active,
            "action": round_reveal.get("player_action"),
            "progress": round_reveal.get("progress", 0.0),
        }
        slots = self._card_slots(plane_size, selection if can_select else reveal_selection)

        self._draw_player_info(frame, H, plane_size, game_state)
        if not reward_active:
            self._draw_enemy_hp(frame, H, plane_size, game_state, enemy_pos)
            self._draw_enemy_action_hint(frame, H, game_state, enemy_pos)
        self._draw_gesture_probability_panel(frame, H, plane_size, gesture_info)
        self._draw_augment_badges(frame, H, plane_size, game_state)

        if reward_active:
            reward_slots = self._reward_card_slots(H, plane_size, enemy_pos, reward_selection)
            self._draw_reward_cards(frame, H, reward_slots, reward_selection)
        elif can_select:
            for action in self.ACTIONS:
                slot = slots[action["action"]]
                self._draw_board_card(frame, H, slot, action, selection)
        elif reveal_active:
            selected_action = round_reveal.get("player_action")
            for action in self.ACTIONS:
                if action["action"] == selected_action:
                    self._draw_board_card(frame, H, slots[action["action"]], action, reveal_selection)
                    break

        self._add_effects_from_events(events, slots, enemy_pos)
        self._draw_round_reveal(frame, H, game_state, enemy_pos)
        self._draw_effects(frame, H)
        return frame

    def _card_slots(self, plane_size, selection):
        plane_width, plane_height = float(plane_size[0]), float(plane_size[1])
        card_w = min(62.0, plane_width * 0.275)
        card_h = card_w * 1.34
        gap = card_w * 0.16
        total_w = card_w * 3 + gap * 2
        start_x = plane_width * 0.5 - total_w * 0.5
        y = plane_height + 68.0
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

    def _draw_player_info(self, frame, H, plane_size, game_state):
        player = game_state.get("player", {})
        max_hp = max(int(player.get("max_hp", 1)), 1)
        hp = max(0, int(player.get("hp", 0)))
        ratio = hp / float(max_hp)
        growth = player.get("growth", {}) or {}
        image = self._player_info_image(hp, max_hp, ratio, growth)

        plane_width, plane_height = float(plane_size[0]), float(plane_size[1])
        rect = [
            (0.0, plane_height + 8.0),
            (plane_width, plane_height + 8.0),
            (plane_width, plane_height + 58.0),
            (0.0, plane_height + 58.0),
        ]
        dst = self._project_points(rect, H)
        if dst is None:
            return
        self._warp_rgba(frame, image, dst)
        cv2.polylines(frame, [dst.astype(np.int32)], True, (210, 190, 120), 1, cv2.LINE_AA)

    def _player_info_image(self, hp, max_hp, ratio, growth=None):
        stat_line = self._player_stat_line(hp, max_hp, growth or {})
        key = (int(hp), int(max_hp), int(round(ratio * 100)), stat_line)
        cached = self.panel_cache.get(key)
        if cached is not None:
            return cached

        width, height = 620, 170
        if Image is None:
            image = np.zeros((height, width, 4), dtype=np.uint8)
            image[:, :, :3] = (24, 20, 18)
            image[:, :, 3] = 210
            cv2.putText(image, "PLAYER", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (245, 238, 210, 255), 2, cv2.LINE_AA)
            cv2.putText(image, stat_line, (28, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (230, 218, 178, 255), 1, cv2.LINE_AA)
            self._draw_bgra_bar(image, 28, 120, width - 56, 28, ratio, (60, 190, 90), f"HP {hp}/{max_hp}")
            self.panel_cache[key] = image
            return image

        canvas = Image.new("RGBA", (width, height), (12, 11, 18, 208))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((4, 4, width - 5, height - 5), outline=(205, 185, 118, 235), width=3)
        draw.rectangle((14, 14, width - 15, height - 15), outline=(80, 66, 46, 180), width=1)

        title_font = self._font(38, bold=True)
        value_font = self._font(21, bold=False)
        stat_font = self._font(19, bold=False)
        draw.text((28, 18), "\uD50C\uB808\uC774\uC5B4", font=title_font, fill=(248, 238, 205, 255))
        draw.text((28, 70), stat_line, font=stat_font, fill=(230, 218, 178, 255))

        bar_x, bar_y, bar_w, bar_h = 28, 120, width - 56, 28
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=(34, 32, 40, 245), outline=(210, 190, 140, 255), width=2)
        draw.rectangle((bar_x + 2, bar_y + 2, bar_x + 2 + int((bar_w - 4) * ratio), bar_y + bar_h - 2), fill=(72, 196, 92, 255))
        draw.text((bar_x + 10, bar_y + 1), f"HP {hp}/{max_hp}", font=value_font, fill=(255, 252, 235, 255))

        rgba = np.asarray(canvas, dtype=np.uint8)
        bgra = rgba[:, :, [2, 1, 0, 3]].copy()
        self.panel_cache[key] = bgra
        return bgra

    @staticmethod
    def _player_stat_line(hp, max_hp, growth):
        attack_power = int(growth.get("attack_power", 15) or 15)
        strike_bonus = int(growth.get("strike_bonus", 0) or 0)
        guard_bonus = int(growth.get("guard_bonus", 0) or 0)
        guard_heal_ratio_bonus = float(growth.get("guard_heal_ratio_bonus", 0.0) or 0.0)
        shot_bonus = int(growth.get("shot_bonus", 0) or 0)
        damage_multiplier = float(growth.get("damage_multiplier", 1.0) or 1.0)
        heal_multiplier = float(growth.get("heal_multiplier", 1.0) or 1.0)

        strike = max(0, int(round((attack_power + strike_bonus) * damage_multiplier)))
        shot = max(0, int(round((attack_power + shot_bonus) * 2 * damage_multiplier)))
        missing_hp = max(0, int(max_hp) - int(hp))
        base_heal = max(5, int(round(missing_hp * (0.10 + guard_heal_ratio_bonus))))
        guard_heal = max(0, int(round((base_heal + guard_bonus) * heal_multiplier)))
        return f"ATK {attack_power}  STR {strike}  SHOT {shot}  HEAL {guard_heal}"

    def _font(self, size, bold=False):
        key = (size, bold)
        cached = self.font_cache.get(key)
        if cached is not None:
            return cached
        candidates = []
        if bold:
            candidates.append(Path("C:/Windows/Fonts/malgunbd.ttf"))
        candidates.extend(
            [
                Path("C:/Windows/Fonts/malgun.ttf"),
                Path("C:/Windows/Fonts/gulim.ttc"),
            ]
        )
        for path in candidates:
            if path.exists():
                try:
                    font = ImageFont.truetype(str(path), size)
                    self.font_cache[key] = font
                    return font
                except Exception:
                    pass
        font = ImageFont.load_default()
        self.font_cache[key] = font
        return font

    def _draw_bgra_bar(self, image, x, y, w, h, ratio, color, label):
        ratio = max(0.0, min(1.0, ratio))
        cv2.rectangle(image, (x, y), (x + w, y + h), (34, 32, 40, 245), -1)
        cv2.rectangle(image, (x + 2, y + 2), (x + 2 + int((w - 4) * ratio), y + h - 2), color + (255,), -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (210, 190, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label, (x + 10, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 252, 235, 255), 1, cv2.LINE_AA)

    def _draw_enemy_hp(self, frame, H, plane_size, game_state, enemy_pos):
        enemy = game_state.get("enemy", {})
        max_hp = max(int(enemy.get("max_hp", 1)), 1)
        hp = max(0, int(enemy.get("hp", 0)))
        ratio = hp / float(max_hp)
        point = HomographyEstimator.transform_point(enemy_pos, H)
        if point is None:
            return

        scale = self._local_pixel_scale(H, enemy_pos)
        bar_w = int(np.clip(scale * float(plane_size[0]) * 0.58, 84, 168))
        bar_h = int(np.clip(scale * 8.0, 9, 15))
        y_offset = int(np.clip(scale * 82.0, 78, 168))
        cx = int(round(point[0]))
        cy = int(round(point[1])) - y_offset
        x1 = cx - bar_w // 2
        y1 = cy - bar_h // 2
        x2 = x1 + bar_w
        y2 = y1 + bar_h

        x1 = max(4, min(frame.shape[1] - bar_w - 4, x1))
        x2 = x1 + bar_w
        y1 = max(22, min(frame.shape[0] - bar_h - 4, y1))
        y2 = y1 + bar_h

        bg_x1 = max(0, x1 - 8)
        bg_y1 = max(0, y1 - 22)
        bg_x2 = min(frame.shape[1], x2 + 8)
        bg_y2 = min(frame.shape[0], y2 + 7)
        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
            roi = frame[bg_y1:bg_y2, bg_x1:bg_x2]
            overlay = roi.copy()
            cv2.rectangle(overlay, (0, 0), (bg_x2 - bg_x1 - 1, bg_y2 - bg_y1 - 1), (8, 8, 14), -1)
            cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)
        name = str(enemy.get("name", "Enemy"))[:18]
        cv2.putText(frame, name, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 235, 210), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (38, 36, 44), -1)
        cv2.rectangle(frame, (x1, y1), (x1 + int(bar_w * ratio), y2), tuple(enemy.get("color", (70, 70, 210))), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 205, 150), 1, cv2.LINE_AA)

    def _draw_enemy_action_hint(self, frame, H, game_state, enemy_pos):
        round_reveal = game_state.get("round_reveal", {})
        if round_reveal.get("active"):
            return
        hint = game_state.get("enemy_action_hint") or {}
        if not hint:
            return
        point = HomographyEstimator.transform_point(enemy_pos, H)
        if point is None:
            return

        scale = self._local_pixel_scale(H, enemy_pos)
        cx = int(round(point[0]))
        cy = int(round(point[1] - np.clip(scale * 128.0, 116, 236)))
        text = "  ".join(
            [
                f"ATK {self._format_probability_percent(hint.get('Attack', 0.0))}",
                f"DEF {self._format_probability_percent(hint.get('Defend', 0.0))}",
                f"SKL {self._format_probability_percent(hint.get('Skill', 0.0))}",
            ]
        )
        width = 242
        height = 25
        x1 = max(4, min(frame.shape[1] - width - 4, cx - width // 2))
        y1 = max(4, min(frame.shape[0] - height - 4, cy - height // 2))
        roi = frame[y1 : y1 + height, x1 : x1 + width]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (12, 12, 18), -1)
        cv2.addWeighted(overlay, 0.58, roi, 0.42, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (96, 88, 64), 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x1 + 8, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 222, 190), 1, cv2.LINE_AA)

    @staticmethod
    def _format_probability_percent(value):
        percent = max(0.0, min(100.0, float(value) * 100.0))
        truncated = np.floor(percent * 10.0) / 10.0
        return f"{truncated:.1f}%"

    def _draw_gesture_probability_panel(self, frame, H, plane_size, gesture_info):
        if not gesture_info:
            return
        probabilities = gesture_info.get("class_probabilities") or {}
        rows = [
            ("FIST", float(probabilities.get("Fist", 0.0)), (70, 150, 255), {"Fist"}),
            ("PALM", float(probabilities.get("Open_Palm", 0.0)), (80, 220, 255), {"Open_Palm"}),
            (
                "SCIS",
                max(float(probabilities.get("V_Sign", 0.0)), float(probabilities.get("Gun_Sign", 0.0))),
                (255, 120, 230),
                {"V_Sign", "Gun_Sign"},
            ),
        ]
        if max(row[1] for row in rows) <= 0.0:
            return

        active = gesture_info.get("smoothed_gesture")
        if active == "Unknown":
            active = None
        image = self._gesture_probability_image(rows, active)

        plane_width, plane_height = float(plane_size[0]), float(plane_size[1])
        panel_w = min(92.0, plane_width * 0.62)
        panel_h = panel_w * 1.02
        x1 = plane_width + plane_width * 0.045
        y1 = plane_height * 0.26
        rect = [
            (x1, y1),
            (x1 + panel_w, y1),
            (x1 + panel_w, y1 + panel_h),
            (x1, y1 + panel_h),
        ]
        dst = self._project_points(rect, H)
        if dst is None:
            return
        self._warp_rgba(frame, image, dst)
        cv2.polylines(frame, [dst.astype(np.int32)], True, (150, 138, 96), 1, cv2.LINE_AA)

    def _gesture_probability_image(self, rows, active):
        key = ("gesture_probs", active) + tuple(int(round(row[1] * 100.0)) for row in rows)
        cached = self.panel_cache.get(key)
        if cached is not None:
            return cached

        width, height = 300, 286
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[:, :, :3] = (14, 13, 20)
        image[:, :, 3] = 214
        cv2.rectangle(image, (5, 5), (width - 6, height - 6), (86, 80, 58, 235), 2, cv2.LINE_AA)
        cv2.putText(image, "GESTURE", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (244, 236, 202, 255), 2, cv2.LINE_AA)

        bar_x, bar_w, bar_h = 106, 142, 24
        for index, (label, probability, color, aliases) in enumerate(rows):
            y = 88 + index * 58
            is_active = active in aliases
            label_color = (255, 252, 218, 255) if is_active else (205, 198, 174, 255)
            cv2.putText(image, label, (22, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
            cv2.rectangle(image, (bar_x, y), (bar_x + bar_w, y + bar_h), (38, 36, 46, 245), -1)
            fill_w = int(round(bar_w * max(0.0, min(1.0, probability))))
            if fill_w > 0:
                cv2.rectangle(image, (bar_x, y), (bar_x + fill_w, y + bar_h), color + (255,), -1)
            cv2.rectangle(image, (bar_x, y), (bar_x + bar_w, y + bar_h), (150, 140, 104, 255), 1, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{int(round(probability * 100.0)):02d}",
                (bar_x + bar_w + 12, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                label_color,
                1,
                cv2.LINE_AA,
            )
            if is_active:
                cv2.rectangle(image, (13, y - 8), (width - 14, y + 35), (95, 235, 255, 255), 1, cv2.LINE_AA)

        self.panel_cache[key] = image
        return image


    def _draw_augment_badges(self, frame, H, plane_size, game_state):
        player = game_state.get("player", {}) or {}
        growth = player.get("growth", {}) or {}
        augments = growth.get("augments") or []
        if not augments:
            return

        plane_width, plane_height = float(plane_size[0]), float(plane_size[1])
        panel_w = min(92.0, plane_width * 0.62)
        panel_h = panel_w * 1.02
        badge_w = panel_w
        badge_h = 16.5
        gap = 4.4
        x1 = plane_width + plane_width * 0.045
        y1 = plane_height * 0.26 + panel_h + 9.0
        max_visible = 6

        visible = augments[:max_visible]
        if len(augments) > max_visible:
            visible = list(visible) + [
                {
                    "flag": "more",
                    "label": f"+{len(augments) - max_visible}",
                    "short_label": f"+{len(augments) - max_visible}",
                    "description": "More augments",
                }
            ]

        for index, augment in enumerate(visible):
            top = y1 + index * (badge_h + gap)
            rect = [
                (x1, top),
                (x1 + badge_w, top),
                (x1 + badge_w, top + badge_h),
                (x1, top + badge_h),
            ]
            dst = self._project_points(rect, H)
            if dst is None:
                continue
            image = self._augment_badge_image(augment)
            self._warp_rgba(frame, image, dst)
            cv2.polylines(frame, [dst.astype(np.int32)], True, (215, 155, 225), 1, cv2.LINE_AA)

    def _augment_badge_image(self, augment):
        label = str(augment.get("label") or augment.get("flag") or "Augment")
        short_label = str(augment.get("short_label") or label[:10])
        key = ("augment_badge", label, short_label)
        cached = self.panel_cache.get(key)
        if cached is not None:
            return cached

        width, height = 300, 64
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[:, :, :3] = (20, 16, 28)
        image[:, :, 3] = 218
        cv2.rectangle(image, (4, 4), (width - 5, height - 5), (225, 130, 235, 245), 2, cv2.LINE_AA)
        cv2.rectangle(image, (14, 14), (45, height - 15), (95, 48, 108, 245), -1)
        cv2.circle(image, (30, height // 2), 8, (245, 188, 255, 255), -1, cv2.LINE_AA)

        if Image is not None:
            rgba = image[:, :, [2, 1, 0, 3]].copy()
            canvas = Image.fromarray(rgba, "RGBA")
            draw = ImageDraw.Draw(canvas)
            font = self._font(22, bold=True)
            draw.text((58, 16), label[:14], font=font, fill=(252, 235, 255, 255))
            image = np.asarray(canvas, dtype=np.uint8)[:, :, [2, 1, 0, 3]].copy()
        else:
            cv2.putText(image, short_label[:10], (58, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (252, 235, 255, 255), 1, cv2.LINE_AA)

        self.panel_cache[key] = image
        return image

    def _reward_card_slots(self, H, plane_size, enemy_pos, selection):
        center = HomographyEstimator.transform_point(enemy_pos, H)
        if center is None:
            return {}
        center = np.asarray(center, dtype=np.float32)

        scale = self._local_pixel_scale(H, enemy_pos)
        plane_width = float(plane_size[0])
        reward_card_scale = 2.45
        card_w = float(np.clip(scale * plane_width * 0.28 * reward_card_scale, 62.0 * reward_card_scale, 122.0 * reward_card_scale))
        card_h = card_w * 1.34
        gap = card_w * 0.10
        selected_action = selection.get("action") if selection.get("active") else None

        x_axis = self._projected_axis(H, enemy_pos, (10.0, 0.0), fallback=(1.0, 0.0))
        y_axis = self._projected_axis(H, enemy_pos, (0.0, 10.0), fallback=(0.0, 1.0))
        screen_up = np.asarray([0.0, -1.0], dtype=np.float32)

        tilt_radians = np.deg2rad(60.0)
        ground_depth = np.cos(tilt_radians) * card_h * 0.54
        screen_lift = np.sin(tilt_radians) * card_h * 0.54
        top_vector = -y_axis * ground_depth + screen_up * screen_lift

        row_width = card_w * 3.0 + gap * 2.0
        row_origin = center - x_axis * (row_width * 0.5 - card_w * 0.5)
        slots = {}
        for index, action in enumerate(self.ACTIONS):
            scale_selected = 1.12 if action["action"] == selected_action else 1.0
            width = card_w * scale_selected
            bottom_center = row_origin + x_axis * index * (card_w + gap) + y_axis * (card_h * 0.10)
            top_center = bottom_center + top_vector * scale_selected
            half_width = x_axis * (width * 0.5)
            dst = np.asarray(
                [
                    top_center - half_width,
                    top_center + half_width,
                    bottom_center + half_width,
                    bottom_center - half_width,
                ],
                dtype=np.float32,
            )
            slots[action["action"]] = {
                "center": tuple(bottom_center.tolist()),
                "dst": dst,
                "color": action["color"],
            }
        return slots

    def _projected_axis(self, H, center, delta, fallback):
        base = HomographyEstimator.transform_point(center, H)
        shifted = HomographyEstimator.transform_point((center[0] + delta[0], center[1] + delta[1]), H)
        if base is None or shifted is None:
            axis = np.asarray(fallback, dtype=np.float32)
        else:
            axis = np.asarray(shifted, dtype=np.float32) - np.asarray(base, dtype=np.float32)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-5:
            axis = np.asarray(fallback, dtype=np.float32)
            norm = float(np.linalg.norm(axis))
        return axis / max(norm, 1e-5)

    def _draw_reward_cards(self, frame, H, slots, reward_selection):
        choices = reward_selection.get("choices") or []
        for index, choice in enumerate(choices[:3]):
            slot_action = choice.get("slot_action") or ("Strike", "Guard", "Shot")[min(index, 2)]
            slot = slots.get(slot_action)
            if slot is None:
                continue
            self._draw_reward_card(frame, H, slot, choice, reward_selection)

    def _draw_reward_card(self, frame, H, slot, choice, reward_selection):
        image = self._reward_card_image(choice)
        if "dst" in slot:
            dst = np.asarray(slot["dst"], dtype=np.float32)
        else:
            dst = self._project_points(slot["rect"], H)
        if dst is None:
            return

        self._warp_rgba(frame, image, dst)
        selected = (
            reward_selection.get("active")
            and reward_selection.get("selected_index") == choice.get("slot_index")
        )
        color = self._reward_color(choice.get("category"))
        border = (255, 245, 120) if selected else (160, 150, 120)
        thickness = 3 if selected else 1
        cv2.polylines(frame, [dst.astype(np.int32)], True, border, thickness, cv2.LINE_AA)
        if selected:
            progress = float(reward_selection.get("hold_progress", reward_selection.get("progress", 0.0)) or 0.0)
            self._draw_progress_around_card(frame, dst, progress)

    def _reward_card_image(self, choice):
        category = str(choice.get("category", "reward"))
        title = str(choice.get("title", "Reward"))
        description = str(choice.get("description", "Details pending"))
        key = ("reward", choice.get("id"), title, category, description)
        cached = self.card_cache.get(key)
        if cached is not None:
            return cached

        width, height = 180, 244
        color = self._reward_color(category)
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[:, :, 3] = 224
        image[:, :, :3] = (24, 22, 32)
        cv2.rectangle(image, (7, 7), (width - 8, height - 8), color, 3, cv2.LINE_AA)
        cv2.rectangle(image, (18, 22), (width - 19, 96), tuple(max(0, int(c * 0.48)) for c in color), -1)

        if Image is not None:
            rgba = image[:, :, [2, 1, 0, 3]].copy()
            canvas = Image.fromarray(rgba, "RGBA")
            draw = ImageDraw.Draw(canvas)
            title_font = self._font(19, bold=True)
            small_font = self._font(12, bold=False)
            accent = (color[2], color[1], color[0], 255)
            title_lines = self._wrap_text(title, 12, 2)
            title_start_y = 38 if len(title_lines) == 1 else 28
            for line_index, line in enumerate(title_lines):
                bbox = draw.textbbox((0, 0), line, font=title_font)
                line_width = bbox[2] - bbox[0]
                draw.text(((width - line_width) * 0.5, title_start_y + line_index * 22), line, font=title_font, fill=(255, 247, 220, 255))
            draw.text((18, 111), category.upper()[:18], font=small_font, fill=accent)
            for line_index, line in enumerate(self._wrap_text(description, 18, 4)):
                draw.text((18, 142 + line_index * 16), line, font=small_font, fill=(214, 207, 188, 255))
            image = np.asarray(canvas, dtype=np.uint8)[:, :, [2, 1, 0, 3]].copy()
        else:
            cv2.putText(image, title[:10].upper(), (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 240, 225, 255), 2, cv2.LINE_AA)
            cv2.putText(image, category.upper()[:12], (18, 123), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color + (255,), 1, cv2.LINE_AA)
            for line_index, line in enumerate(self._wrap_text(description, 18, 4)):
                cv2.putText(image, line, (18, 150 + line_index * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (214, 207, 188, 255), 1, cv2.LINE_AA)

        self.card_cache[key] = image
        return image

    @staticmethod
    def _reward_color(category):
        colors = {
            "stat": (70, 150, 255),
            "heal": (80, 220, 120),
            "card_upgrade": (255, 160, 80),
            "augment": (255, 120, 230),
        }
        return colors.get(str(category), (190, 170, 110))

    @staticmethod
    def _wrap_text(text, max_chars, max_lines):
        words = str(text).split()
        if not words:
            return [""]
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                lines.append(current[:max_chars])
                current = word
                if len(lines) >= max_lines:
                    break
        if current and len(lines) < max_lines:
            lines.append(current[:max_chars])
        return lines[:max_lines]

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
        left = quad[0].astype(np.float32)
        right = quad[1].astype(np.float32)
        end = left + (right - left) * progress
        cv2.line(frame, tuple(left.astype(int)), tuple(right.astype(int)), (45, 45, 55), 5, cv2.LINE_AA)
        cv2.line(frame, tuple(left.astype(int)), tuple(end.astype(int)), color, 5, cv2.LINE_AA)

    def _draw_progress_around_card(self, frame, quad, progress):
        progress = max(0.0, min(1.0, float(progress)))
        if progress <= 0.0:
            return
        points = np.asarray(quad, dtype=np.float32)
        if points.shape != (4, 2):
            return

        glow = frame.copy()
        cv2.polylines(glow, [points.astype(np.int32)], True, (255, 255, 255), 9, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.24, frame, 0.76, 0, frame)

        segments = list(zip(points, np.roll(points, -1, axis=0)))
        lengths = [float(np.linalg.norm(end - start)) for start, end in segments]
        remaining = sum(lengths) * progress
        for (start, end), length in zip(segments, lengths):
            if remaining <= 0.0:
                break
            if remaining >= length:
                cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (255, 255, 255), 5, cv2.LINE_AA)
                remaining -= length
                continue
            ratio = remaining / max(length, 1e-6)
            partial_end = start + (end - start) * ratio
            cv2.line(frame, tuple(start.astype(int)), tuple(partial_end.astype(int)), (255, 255, 255), 5, cv2.LINE_AA)
            break

    def _draw_round_reveal(self, frame, H, game_state, enemy_pos):
        reveal = game_state.get("round_reveal", {})
        if not reveal.get("active"):
            return

        action = reveal.get("enemy_action")
        point = HomographyEstimator.transform_point(enemy_pos, H)
        if point is None:
            return

        image = self._load_card(self.ENEMY_CARD_PATHS.get(action), str(action or "ACTION").upper(), (80, 80, 255))
        scale = self._local_pixel_scale(H, enemy_pos)
        center = (int(point[0]), int(point[1] - np.clip(scale * 92.0, 96, 172)))
        card_w, card_h = 100, 135
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
        cv2.polylines(frame, [dst.astype(np.int32)], True, (90, 120, 255), 3, cv2.LINE_AA)
        self._draw_progress_on_card(frame, dst, float(reveal.get("progress", 0.0)), (90, 120, 255))

        player_action = reveal.get("player_action", "?")
        label = f"{player_action}  VS  {action}"
        label_x = max(8, min(frame.shape[1] - 260, center[0] - 130))
        label_y = max(28, y1 - 12)
        cv2.rectangle(frame, (label_x - 8, label_y - 23), (label_x + 260, label_y + 6), (8, 8, 14), -1)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 235, 190), 2, cv2.LINE_AA)

    def _add_effects_from_events(self, events, slots, enemy_pos):
        now = time.monotonic()
        for event in events:
            event_id = (event.get("time"), event.get("source"), event.get("kind"), event.get("target"), event.get("label"), event.get("result"))
            if event_id in self.seen_event_ids:
                continue
            self.seen_event_ids.add(event_id)

            if event.get("source") != "player":
                continue

            kind = event.get("kind")
            damage = int(event.get("damage", 0) or 0)
            result = event.get("result")
            if kind in ("Strike", "Shot") and damage > 0:
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
            elif kind == "Guard" and result in ("heal", "block"):
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

    def _local_pixel_scale(self, H, center):
        cx, cy = center
        p0 = np.asarray(HomographyEstimator.transform_point((cx, cy), H), dtype=np.float32)
        px = np.asarray(HomographyEstimator.transform_point((cx + 10.0, cy), H), dtype=np.float32)
        py = np.asarray(HomographyEstimator.transform_point((cx, cy + 10.0), H), dtype=np.float32)
        if p0.shape != (2,) or px.shape != (2,) or py.shape != (2,):
            return 1.0
        scale = (np.linalg.norm(px - p0) + np.linalg.norm(py - p0)) / 20.0
        return float(max(0.45, min(3.5, scale)))

    def _warp_rgba(self, frame, rgba, dst):
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            return
        dst = np.asarray(dst, dtype=np.float32)
        x1 = max(0, int(np.floor(np.min(dst[:, 0]))))
        y1 = max(0, int(np.floor(np.min(dst[:, 1]))))
        x2 = min(frame.shape[1], int(np.ceil(np.max(dst[:, 0]))) + 1)
        y2 = min(frame.shape[0], int(np.ceil(np.max(dst[:, 1]))) + 1)
        if x2 <= x1 or y2 <= y1:
            return

        src_h, src_w = rgba.shape[:2]
        src = np.asarray([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], dtype=np.float32)
        local_dst = dst - np.asarray([x1, y1], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, local_dst)
        warped = cv2.warpPerspective(
            rgba,
            matrix,
            (x2 - x1, y2 - y1),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        if np.max(alpha) <= 0.0:
            return
        roi = frame[y1:y2, x1:x2]
        color = warped[:, :, :3].astype(np.float32)
        blended = color * alpha + roi.astype(np.float32) * (1.0 - alpha)
        np.copyto(roi, blended.astype(np.uint8))
