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

    def add_from_events(self, events, player_pos, enemy_pos):
        for event in events:
            event_id = (event.get("time"), event.get("kind"), event.get("label"), event.get("target"))
            if event_id in self.seen_event_ids:
                continue
            self.seen_event_ids.add(event_id)

            target = event.get("target", "center")
            if target == "enemy":
                origin = enemy_pos
                color = (80, 220, 255)
            elif target == "player":
                origin = player_pos
                color = (80, 80, 255)
            else:
                origin = None
                color = (110, 230, 255)

            damage = int(event.get("damage", 0) or 0)
            if damage > 0:
                label = f"-{damage}"
            elif event.get("kind") in ("Guard", "Defend"):
                label = "BLOCK"
            else:
                label = event.get("label", "")

            if label:
                self.items.append(
                    {
                        "label": label,
                        "origin": origin,
                        "color": color,
                        "start": time.monotonic(),
                        "duration": 1.25 if target != "center" else 1.6,
                        "target": target,
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

            if item["origin"] is None or H is None:
                x = width // 2 - 120
                y = int(height * 0.28 - progress * 24)
                scale = 0.95
                thickness = 3
            else:
                x, y = projector(item["origin"], H)
                x -= 24
                y -= int(22 + progress * 48)
                scale = 0.72
                thickness = 2

            draw_color = tuple(max(0, min(255, int(channel * alpha + 40 * (1.0 - alpha)))) for channel in color)
            cv2.putText(frame, item["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, draw_color, thickness, cv2.LINE_AA)
            kept.append(item)
        self.items = kept
        return frame
