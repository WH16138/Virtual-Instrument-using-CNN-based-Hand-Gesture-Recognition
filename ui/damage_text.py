import random
import time

import cv2


class FloatingTextManager:
    """Small time-based combat feedback layer."""

    def __init__(self):
        self.items = []
        self.seen_event_ids = set()

    def reset(self):
        self.items.clear()
        self.seen_event_ids.clear()

    def add_from_events(self, events, player_pos, enemy_pos, player_feedback_pos=None):
        target_counts = {}
        for event in events:
            event_id = (event.get("time"), event.get("kind"), event.get("label"), event.get("target"), event.get("result"))
            if event_id in self.seen_event_ids:
                continue
            self.seen_event_ids.add(event_id)

            target = event.get("target", "center")
            result = event.get("result")
            if target == "enemy":
                origin = enemy_pos
                color = (80, 220, 255)
            elif target == "player":
                origin = player_feedback_pos if player_feedback_pos is not None else player_pos
                color = (80, 80, 255)
            else:
                origin = None
                color = (110, 230, 255)

            damage = int(event.get("damage", 0) or 0)
            heal = int(event.get("heal", 0) or 0)
            if damage > 0 and (event.get("critical") or result == "critical"):
                label = f"CRIT -{damage}"
                color = (80, 180, 255)
            elif damage > 0:
                label = f"-{damage}"
            elif heal > 0:
                label = f"+{heal}"
                color = (90, 230, 130)
            elif result == "block":
                label = "BLOCK"
                color = (95, 235, 255)
            elif result == "miss":
                label = "MISS"
                color = (185, 185, 185)
            elif result == "heal_failed":
                label = "FAIL"
                color = (95, 95, 255)
            elif result == "heal":
                label = event.get("label", "FULL")
                color = (90, 230, 130)
            else:
                label = event.get("label", "")

            if label:
                stack_index = target_counts.get(target, 0)
                target_counts[target] = stack_index + 1
                if target == "center":
                    screen_offset = (
                        random.uniform(-42.0, 42.0),
                        random.uniform(-10.0, 10.0) + stack_index * 20.0,
                    )
                else:
                    screen_offset = (
                        random.uniform(-24.0, 24.0),
                        random.uniform(-8.0, 8.0) - stack_index * 17.0,
                    )
                self.items.append(
                    {
                        "label": label,
                        "origin": origin,
                        "color": color,
                        "start": time.monotonic(),
                        "duration": 1.25 if target != "center" else 1.6,
                        "target": target,
                        "screen_offset": screen_offset,
                    }
                )

    def draw(self, frame, H, projector):
        now = time.monotonic()
        kept = []
        height, width = frame.shape[:2]
        for item in self.items:
            age = now - item["start"]
            if age > item["duration"]:
                continue
            progress = age / item["duration"]
            color = item["color"]
            alpha = 1.0 - progress
            offset_x, offset_y = item.get("screen_offset", (0.0, 0.0))

            if item["origin"] is None or H is None:
                x = int(width // 2 - 120 + offset_x)
                y = int(height * 0.28 - progress * 24 + offset_y)
                scale = 0.95
                thickness = 3
            else:
                x, y = projector(item["origin"], H)
                x = int(x - 32 + offset_x)
                y = int(y - 22 - progress * 48 + offset_y)
                scale = 0.72
                thickness = 2

            draw_color = tuple(max(0, min(255, int(channel * alpha + 40 * (1.0 - alpha)))) for channel in color)
            cv2.putText(frame, item["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, draw_color, thickness, cv2.LINE_AA)
            kept.append(item)
        self.items = kept
        return frame
