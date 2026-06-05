import cv2
import numpy as np

from ar.homography import HomographyEstimator


class ARRenderer:
    """Render a simple 3D battlefield rising from the detected A4 board."""

    def __init__(self, plane_size=(210, 297)):
        self.plane_width, self.plane_height = plane_size

    def render_battlefield(self, frame, H, player_pos, enemy_pos, game_state=None):
        if H is None:
            return frame

        pose = self._pose_from_homography(H, frame.shape)
        if pose is None:
            return self._render_flat_fallback(frame, H, player_pos, enemy_pos)

        self._draw_floor(frame, pose)
        self._draw_3d_grid(frame, pose, grid_size=30)
        self._draw_corner_pillars(frame, pose)

        units = [
            {"pos": enemy_pos, "label": "E", "color": (30, 30, 230), "size": 32},
            {"pos": player_pos, "label": "P", "color": (40, 210, 70), "size": 32},
        ]
        units.sort(key=lambda item: item["pos"][1])
        for unit in units:
            self._draw_unit_cube(frame, pose, **unit)

        return frame

    def _render_flat_fallback(self, frame, H, player_pos, enemy_pos):
        frame = HomographyEstimator.draw_grid_on_plane(
            frame,
            H,
            plane_size=(self.plane_width, self.plane_height),
            grid_size=30,
            color=(50, 50, 50),
        )
        self._draw_board_outline_2d(frame, H)
        self._draw_token_2d(frame, H, player_pos, "P", (40, 210, 70))
        self._draw_token_2d(frame, H, enemy_pos, "E", (30, 30, 230))
        return frame

    def _camera_matrix(self, frame_shape):
        height, width = frame_shape[:2]
        focal = 0.95 * max(width, height)
        return np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _pose_from_homography(self, H, frame_shape):
        K = self._camera_matrix(frame_shape)
        K_inv = np.linalg.inv(K)
        normalized = K_inv @ np.asarray(H, dtype=np.float64)

        h1 = normalized[:, 0]
        h2 = normalized[:, 1]
        h3 = normalized[:, 2]
        scale = 2.0 / max(np.linalg.norm(h1) + np.linalg.norm(h2), 1e-9)

        r1 = scale * h1
        r2 = scale * h2
        t = scale * h3
        r3 = np.cross(r1, r2)

        R_approx = np.column_stack([r1, r2, r3])
        try:
            u, _, vh = np.linalg.svd(R_approx)
        except np.linalg.LinAlgError:
            return None
        R = u @ vh
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0

        pose = {"K": K, "R": R, "t": t, "height_sign": 1.0}
        center = (self.plane_width / 2.0, self.plane_height / 2.0)
        base = self._project_point_3d(pose, center[0], center[1], 0.0)
        up_plus = self._project_point_3d(pose, center[0], center[1], 40.0)
        up_minus = self._project_point_3d(pose, center[0], center[1], -40.0)
        if base is None or up_plus is None or up_minus is None:
            return None

        pose["height_sign"] = 1.0 if up_plus[1] < up_minus[1] else -1.0
        return pose

    def _project_point_3d(self, pose, x, y, z):
        point = np.array([x, y, z * pose.get("height_sign", 1.0)], dtype=np.float64)
        camera_point = pose["R"] @ point + pose["t"]
        if camera_point[2] <= 1e-6:
            return None
        projected = pose["K"] @ camera_point
        return (
            int(round(projected[0] / projected[2])),
            int(round(projected[1] / projected[2])),
        )

    def _project_polyline(self, pose, points):
        projected = [self._project_point_3d(pose, x, y, z) for x, y, z in points]
        if any(point is None for point in projected):
            return None
        return np.asarray(projected, dtype=np.int32)

    def _draw_floor(self, frame, pose):
        corners = self._project_polyline(
            pose,
            [
                (0, 0, 0),
                (self.plane_width, 0, 0),
                (self.plane_width, self.plane_height, 0),
                (0, self.plane_height, 0),
            ],
        )
        if corners is None:
            return

        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, corners, (34, 44, 46))
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        cv2.polylines(frame, [corners], True, (0, 235, 255), 2, cv2.LINE_AA)

    def _draw_3d_grid(self, frame, pose, grid_size=30):
        width = int(self.plane_width)
        height = int(self.plane_height)
        for x in range(0, width + 1, grid_size):
            line = self._project_polyline(pose, [(x, 0, 0), (x, height, 0)])
            if line is not None:
                cv2.line(frame, tuple(line[0]), tuple(line[1]), (70, 120, 120), 1, cv2.LINE_AA)

        for y in range(0, height + 1, grid_size):
            line = self._project_polyline(pose, [(0, y, 0), (width, y, 0)])
            if line is not None:
                cv2.line(frame, tuple(line[0]), tuple(line[1]), (70, 120, 120), 1, cv2.LINE_AA)

    def _draw_corner_pillars(self, frame, pose):
        pillar_size = 18
        pillar_height = 42
        half = pillar_size / 2.0
        corners = [
            (half, half),
            (self.plane_width - half, half),
            (self.plane_width - half, self.plane_height - half),
            (half, self.plane_height - half),
        ]
        for center in corners:
            self._draw_box(frame, pose, center, pillar_size, pillar_size, pillar_height, (10, 10, 10), label=None)

    def _draw_unit_cube(self, frame, pose, pos, label, color, size):
        self._draw_box(frame, pose, pos, size, size, size, color, label=label)

    def _draw_box(self, frame, pose, center, width, depth, height, color, label=None):
        x, y = center
        hx = width / 2.0
        hy = depth / 2.0

        points = [
            (x - hx, y - hy, 0.0),
            (x + hx, y - hy, 0.0),
            (x + hx, y + hy, 0.0),
            (x - hx, y + hy, 0.0),
            (x - hx, y - hy, height),
            (x + hx, y - hy, height),
            (x + hx, y + hy, height),
            (x - hx, y + hy, height),
        ]
        projected = [self._project_point_3d(pose, *point) for point in points]
        if any(point is None for point in projected):
            return

        projected = np.asarray(projected, dtype=np.int32)
        faces = [
            ([0, 1, 2, 3], self._shade_color(color, 0.35)),
            ([0, 1, 5, 4], self._shade_color(color, 0.62)),
            ([1, 2, 6, 5], self._shade_color(color, 0.78)),
            ([2, 3, 7, 6], self._shade_color(color, 0.50)),
            ([3, 0, 4, 7], self._shade_color(color, 0.70)),
            ([4, 5, 6, 7], self._shade_color(color, 1.12)),
        ]

        face_items = []
        for indices, face_color in faces:
            avg_depth = np.mean([points[index][1] for index in indices])
            polygon = projected[indices]
            face_items.append((avg_depth, polygon, face_color))

        overlay = frame.copy()
        for _, polygon, face_color in sorted(face_items, key=lambda item: item[0]):
            cv2.fillConvexPoly(overlay, polygon, face_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

        edge_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        edge_color = self._shade_color(color, 1.35)
        for a, b in edge_pairs:
            cv2.line(frame, tuple(projected[a]), tuple(projected[b]), edge_color, 1, cv2.LINE_AA)

        if label:
            top_center = self._project_point_3d(pose, x, y, height)
            if top_center is not None:
                cv2.putText(
                    frame,
                    label,
                    (top_center[0] - 8, top_center[1] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    def _shade_color(self, color, factor):
        return tuple(max(0, min(255, int(channel * factor))) for channel in color)

    def _draw_board_outline_2d(self, frame, H):
        corners = np.array(
            [
                [0, 0],
                [self.plane_width, 0],
                [self.plane_width, self.plane_height],
                [0, self.plane_height],
            ],
            dtype=np.float32,
        )
        corners_screen = np.array([HomographyEstimator.transform_point(tuple(corner), H) for corner in corners])
        cv2.polylines(frame, [corners_screen.astype(np.int32)], True, (255, 255, 0), 2)

    def _draw_token_2d(self, frame, H, board_pos, label, color):
        screen_pos = HomographyEstimator.transform_point(board_pos, H)
        if 0 <= screen_pos[0] < frame.shape[1] and 0 <= screen_pos[1] < frame.shape[0]:
            cv2.circle(frame, screen_pos, 14, color, -1)
            cv2.putText(
                frame,
                label,
                (screen_pos[0] - 5, screen_pos[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
