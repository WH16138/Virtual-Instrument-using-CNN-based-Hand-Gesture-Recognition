import cv2
import numpy as np
import time

from ar.homography import HomographyEstimator
from ar.model_loader import ModelLoader
from ar.pyrender_renderer import PyrenderModelRenderer


class ARRenderer:
    """Render 3D-style objects whose bases are exactly on the A4 homography."""

    def __init__(self, plane_size=(210, 297)):
        self.set_plane_size(plane_size)
        self.focal_px = None
        self.homography_samples = []
        self.max_homography_samples = 18
        self.model_loader = ModelLoader()
        self.pbr_renderer = PyrenderModelRenderer()

    def set_plane_size(self, plane_size):
        self.plane_width = float(plane_size[0])
        self.plane_height = float(plane_size[1])

    def render_battlefield(self, frame, H, player_pos, enemy_pos, game_state=None, show_floor_mesh=True):
        if H is None:
            return frame

        self._update_camera_estimate(H, frame.shape)
        pose = self._pose_from_homography(H, frame.shape)

        if show_floor_mesh:
            self._draw_floor(frame, H)
            self._draw_grid(frame, H, grid_size=30)

        enemy_state = (game_state or {}).get("enemy", {})
        enemy_color = tuple(enemy_state.get("color", (30, 30, 230)))
        ground_pos = (self.plane_width * 0.5, self.plane_height * 0.5)
        ground_model_path = enemy_state.get("ground_model_path")
        ground_drawn = self.pbr_renderer.render_model(
            frame,
            ground_model_path,
            pose,
            ground_pos,
            size=min(self.plane_width, self.plane_height) * 0.62,
            height_offset=0.0,
            alpha=0.96,
        )
        if not ground_drawn:
            ground_drawn = self._draw_model_unit(
                frame,
                pos=ground_pos,
                label=None,
                color=(72, 78, 82),
                size=min(self.plane_width, self.plane_height) * 0.62,
                model_path=ground_model_path,
                pose=pose,
                height_offset=0.0,
                alpha=0.62,
                draw_label=False,
            )
        if not ground_drawn:
            self._draw_ground_platform(frame, H, ground_pos, enemy_color)

        enemy_model_path = enemy_state.get("model_path")
        enemy_float_offset = 12.0 + np.sin(time.monotonic() * 1.8) * 5.0
        enemy_drawn = self.pbr_renderer.render_model(
            frame,
            enemy_model_path,
            pose,
            enemy_pos,
            size=46,
            height_offset=enemy_float_offset,
            yaw_degrees=180.0,
            alpha=1.0,
        )
        if not enemy_drawn:
            enemy_drawn = self._draw_model_unit(
            frame,
            pos=enemy_pos,
            label="E",
            color=enemy_color,
            size=38,
            model_path=enemy_model_path,
            pose=pose,
            height_offset=enemy_float_offset,
            )
        if not enemy_drawn:
            self._draw_cube(frame, H, enemy_pos, "E", enemy_color, 34, pose=pose, height_offset=enemy_float_offset)

        return frame

    def _draw_floor(self, frame, H):
        corners = self._project_board_points(
            H,
            [
                (0, 0),
                (self.plane_width, 0),
                (self.plane_width, self.plane_height),
                (0, self.plane_height),
            ],
        )
        if corners is None:
            return

        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, corners, (34, 44, 46), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        cv2.polylines(frame, [corners], True, (0, 235, 255), 2, cv2.LINE_AA)

    def _draw_grid(self, frame, H, grid_size=30):
        HomographyEstimator.draw_grid_on_plane(
            frame,
            H,
            plane_size=(self.plane_width, self.plane_height),
            grid_size=grid_size,
            color=(70, 120, 120),
        )

    def _draw_corner_pillars(self, frame, H, pose):
        size = 18
        half = size / 2.0
        centers = [
            (half, half),
            (self.plane_width - half, half),
            (self.plane_width - half, self.plane_height - half),
            (half, self.plane_height - half),
        ]
        for center in centers:
            self._draw_box(frame, H, center, size, size, 48, (10, 10, 10), label=None, pose=pose)

    def _draw_cube(self, frame, H, pos, label, color, size, pose=None, height_offset=0.0):
        self._draw_box(
            frame,
            H,
            pos,
            size,
            size,
            size * 1.15,
            color,
            label=label,
            pose=pose,
            height_offset=height_offset,
        )

    def _draw_ground_platform(self, frame, H, center, color):
        width = self.plane_width * 0.72
        height = self.plane_height * 0.42
        x, y = center
        points = self._project_board_points(
            H,
            [
                (x - width * 0.5, y - height * 0.5),
                (x + width * 0.5, y - height * 0.5),
                (x + width * 0.5, y + height * 0.5),
                (x - width * 0.5, y + height * 0.5),
            ],
        )
        if points is None:
            return
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, points, self._shade_color(color, 0.36), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)
        cv2.polylines(frame, [points], True, self._shade_color(color, 0.95), 2, cv2.LINE_AA)

    def _draw_model_unit(
        self,
        frame,
        pos,
        label,
        color,
        size,
        model_path=None,
        pose=None,
        height_offset=0.0,
        alpha=0.78,
        draw_label=True,
    ):
        if pose is None or not model_path:
            return False
        model = self.model_loader.load(model_path)
        if model is None:
            return False

        projected = self._project_model_vertices(model.vertices, pos, size * 1.55, pose, height_offset=height_offset)
        if projected is None:
            return False

        face_items = []
        for face_index, face in enumerate(model.faces):
            polygon = projected[np.asarray(face, dtype=np.int32)]
            if len(polygon) < 3:
                continue
            depth = float(np.mean(polygon[:, 1]))
            model_color = None
            if face_index < len(model.face_colors):
                model_color = model.face_colors[face_index]
            face_items.append((depth, polygon.astype(np.int32), model_color))

        if not face_items:
            return False

        overlay = frame.copy()
        for order, (_, polygon, model_color) in enumerate(sorted(face_items, key=lambda item: item[0])):
            base_color = model_color or color
            shade = 0.72 + (order % 4) * 0.08
            cv2.fillConvexPoly(overlay, polygon, self._shade_color(base_color, shade), cv2.LINE_AA)
            cv2.polylines(frame, [polygon], True, self._shade_color(base_color, 1.25), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        if draw_label and label:
            center = tuple(np.mean(projected, axis=0).astype(np.int32))
            cv2.putText(frame, label, (center[0] - 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
        return True

    def _project_model_vertices(self, vertices, center, scale, pose, height_offset=0.0):
        cx, cy = center
        z_sign = pose.get("z_sign", 1.0)
        object_points = np.asarray(
            [
                [
                    cx + float(vertex[0]) * scale,
                    cy + float(vertex[1]) * scale,
                    z_sign * (height_offset + float(vertex[2]) * scale * 1.4),
                ]
                for vertex in vertices
            ],
            dtype=np.float64,
        )
        try:
            projected, _ = cv2.projectPoints(
                object_points,
                pose["rvec"],
                pose["tvec"],
                pose["camera_matrix"],
                pose["dist_coeffs"],
            )
        except cv2.error:
            return None
        return projected.reshape(-1, 2).astype(np.float32)

    def _draw_box(self, frame, H, center, width, depth, height, color, label=None, pose=None, height_offset=0.0):
        x, y = center
        hx = width / 2.0
        hy = depth / 2.0

        base_points_board = [
            (x - hx, y - hy),
            (x + hx, y - hy),
            (x + hx, y + hy),
            (x - hx, y + hy),
        ]
        if height_offset > 0.0 and pose is not None:
            base = self._project_raised_points(base_points_board, height_offset, pose)
        else:
            base = self._project_board_points(H, base_points_board)
        if base is None:
            return

        top = self._project_box_top(base_points_board, height, pose, height_offset=height_offset)
        if top is None:
            lift = self._height_vector(H, center, height)
            top = (base.astype(np.float32) + lift).astype(np.int32)

        self._draw_shadow(frame, base)
        self._draw_prism_faces(frame, base, top, color)
        self._draw_prism_edges(frame, base, top, color)

        if label:
            top_center = tuple(np.mean(top, axis=0).astype(np.int32))
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

    def _height_vector(self, H, center, height):
        cx, cy = center
        center_screen = np.asarray(HomographyEstimator.transform_point((cx, cy), H), dtype=np.float32)
        y_probe = max(cy - 35.0, 0.0)
        toward_top = np.asarray(HomographyEstimator.transform_point((cx, y_probe), H), dtype=np.float32)
        direction = toward_top - center_screen
        norm = np.linalg.norm(direction)
        if norm < 1e-3:
            direction = np.array([0.0, -1.0], dtype=np.float32)
        else:
            direction = direction / norm

        scale = self._local_pixel_scale(H, center)
        return direction * max(8.0, height * scale * 0.42)

    def _local_pixel_scale(self, H, center):
        cx, cy = center
        p0 = np.asarray(HomographyEstimator.transform_point((cx, cy), H), dtype=np.float32)
        px = np.asarray(HomographyEstimator.transform_point((cx + 10.0, cy), H), dtype=np.float32)
        py = np.asarray(HomographyEstimator.transform_point((cx, cy + 10.0), H), dtype=np.float32)
        scale = (np.linalg.norm(px - p0) + np.linalg.norm(py - p0)) / 20.0
        return float(max(0.2, min(3.0, scale)))

    def _camera_matrix(self, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        focal = self.focal_px or (1.05 * max(frame_width, frame_height))
        return np.asarray(
            [
                [focal, 0.0, frame_width * 0.5],
                [0.0, focal, frame_height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _pose_from_homography(self, H, frame_shape):
        image_points = self._project_board_points(
            H,
            [
                (0, 0),
                (self.plane_width, 0),
                (self.plane_width, self.plane_height),
                (0, self.plane_height),
            ],
        )
        if image_points is None:
            return None

        object_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [self.plane_width, 0.0, 0.0],
                [self.plane_width, self.plane_height, 0.0],
                [0.0, self.plane_height, 0.0],
            ],
            dtype=np.float64,
        )
        camera_matrix = self._camera_matrix(frame_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        pose_solution = self._solve_planar_pose(
            object_points,
            image_points.astype(np.float64),
            camera_matrix,
            dist_coeffs,
        )
        if pose_solution is None:
            return None
        rvec, tvec = pose_solution

        z_sign = self._choose_height_sign(H, rvec, tvec, camera_matrix, dist_coeffs)
        return {
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "z_sign": z_sign,
        }

    def _solve_planar_pose(self, object_points, image_points, camera_matrix, dist_coeffs):
        candidates = []

        try:
            result = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if result and bool(result[0]):
                for rvec, tvec in zip(result[1], result[2]):
                    candidates.append((rvec, tvec))
        except cv2.error:
            candidates = []

        if not candidates:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                return None
            if not success:
                return None
            candidates.append((rvec, tvec))

        best_pose = None
        best_error = None
        for rvec, tvec in candidates:
            try:
                projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
            except cv2.error:
                continue
            error = float(np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1).mean())
            if best_error is None or error < best_error:
                best_error = error
                best_pose = (rvec, tvec)

        return best_pose

    def _project_raised_points(self, points, height, pose):
        object_points = np.asarray(
            [[x, y, pose.get("z_sign", 1.0) * height] for x, y in points],
            dtype=np.float64,
        )
        try:
            projected, _ = cv2.projectPoints(
                object_points,
                pose["rvec"],
                pose["tvec"],
                pose["camera_matrix"],
                pose["dist_coeffs"],
            )
        except cv2.error:
            return None
        return projected.reshape(-1, 2).astype(np.int32)

    def _project_box_top(self, base_points_board, height, pose, height_offset=0.0):
        if pose is None:
            return None

        object_points = np.asarray(
            [[x, y, pose.get("z_sign", 1.0) * (height_offset + height)] for x, y in base_points_board],
            dtype=np.float64,
        )
        try:
            projected, _ = cv2.projectPoints(
                object_points,
                pose["rvec"],
                pose["tvec"],
                pose["camera_matrix"],
                pose["dist_coeffs"],
            )
        except cv2.error:
            return None
        return projected.reshape(-1, 2).astype(np.int32)

    def _choose_height_sign(self, H, rvec, tvec, camera_matrix, dist_coeffs):
        center = (self.plane_width * 0.5, self.plane_height * 0.5)
        base = np.asarray(HomographyEstimator.transform_point(center, H), dtype=np.float32)
        expected_lift = self._height_vector(H, center, 40.0)
        if np.linalg.norm(expected_lift) < 1e-3:
            return 1.0

        best_sign = 1.0
        best_score = None
        for z_sign in (-1.0, 1.0):
            object_point = np.asarray([[center[0], center[1], z_sign * 40.0]], dtype=np.float64)
            try:
                projected, _ = cv2.projectPoints(object_point, rvec, tvec, camera_matrix, dist_coeffs)
            except cv2.error:
                continue
            displacement = projected.reshape(2).astype(np.float32) - base
            score = float(np.dot(displacement, expected_lift))
            if best_score is None or score > best_score:
                best_score = score
                best_sign = z_sign
        return best_sign

    def _update_camera_estimate(self, H, frame_shape):
        H = np.asarray(H, dtype=np.float64)
        if abs(H[2, 2]) > 1e-9:
            H = H / H[2, 2]
        self._last_H = H.astype(np.float32)

        if not self._is_new_homography_sample(H):
            return
        self.homography_samples.append(H)
        if len(self.homography_samples) > self.max_homography_samples:
            self.homography_samples.pop(0)

        if len(self.homography_samples) >= 4:
            self.focal_px = self._estimate_focal_from_samples(frame_shape)

    def _is_new_homography_sample(self, H):
        if not self.homography_samples:
            return True
        corners = self._homography_corners(H)
        if corners is None:
            return False
        for sample in self.homography_samples:
            sample_corners = self._homography_corners(sample)
            if sample_corners is None:
                continue
            if np.linalg.norm(corners - sample_corners, axis=1).mean() < 18.0:
                return False
        return True

    def _homography_corners(self, H):
        points = self._project_board_points(
            H,
            [
                (0, 0),
                (self.plane_width, 0),
                (self.plane_width, self.plane_height),
                (0, self.plane_height),
            ],
        )
        return points.astype(np.float32) if points is not None else None

    def _estimate_focal_from_samples(self, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        max_dim = max(frame_width, frame_height)
        cx = frame_width * 0.5
        cy = frame_height * 0.5
        candidates = np.linspace(max_dim * 0.65, max_dim * 2.0, 48)
        best_focal = self.focal_px or (1.05 * max_dim)
        best_score = None

        for focal in candidates:
            inv_k = np.asarray(
                [
                    [1.0 / focal, 0.0, -cx / focal],
                    [0.0, 1.0 / focal, -cy / focal],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            score = 0.0
            for H in self.homography_samples:
                normalized = inv_k @ H
                r1 = normalized[:, 0]
                r2 = normalized[:, 1]
                n1 = np.linalg.norm(r1)
                n2 = np.linalg.norm(r2)
                if n1 <= 1e-9 or n2 <= 1e-9:
                    score += 1000.0
                    continue
                orthogonality_error = abs(float(np.dot(r1, r2))) / (n1 * n2)
                scale_error = abs(n1 - n2) / max((n1 + n2) * 0.5, 1e-9)
                score += orthogonality_error + scale_error

            if best_score is None or score < best_score:
                best_score = score
                best_focal = float(focal)
        return best_focal

    def _project_board_points(self, H, points):
        projected = [HomographyEstimator.transform_point(point, H) for point in points]
        if any(point is None for point in projected):
            return None
        return np.asarray(projected, dtype=np.int32)

    def _draw_shadow(self, frame, base):
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, base, (15, 15, 15), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    def _draw_prism_faces(self, frame, base, top, color):
        faces = [
            ([base[0], base[1], top[1], top[0]], self._shade_color(color, 0.62)),
            ([base[1], base[2], top[2], top[1]], self._shade_color(color, 0.78)),
            ([base[2], base[3], top[3], top[2]], self._shade_color(color, 0.50)),
            ([base[3], base[0], top[0], top[3]], self._shade_color(color, 0.70)),
            ([top[0], top[1], top[2], top[3]], self._shade_color(color, 1.12)),
        ]
        overlay = frame.copy()
        for polygon, face_color in faces:
            cv2.fillConvexPoly(overlay, np.asarray(polygon, dtype=np.int32), face_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    def _draw_prism_edges(self, frame, base, top, color):
        edge_color = self._shade_color(color, 1.35)
        for idx in range(4):
            next_idx = (idx + 1) % 4
            cv2.line(frame, tuple(base[idx]), tuple(base[next_idx]), edge_color, 1, cv2.LINE_AA)
            cv2.line(frame, tuple(top[idx]), tuple(top[next_idx]), edge_color, 1, cv2.LINE_AA)
            cv2.line(frame, tuple(base[idx]), tuple(top[idx]), edge_color, 1, cv2.LINE_AA)

    def _shade_color(self, color, factor):
        return tuple(max(0, min(255, int(channel * factor))) for channel in color)
