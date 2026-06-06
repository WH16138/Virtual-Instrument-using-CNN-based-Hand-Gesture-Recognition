import itertools

import cv2
import numpy as np

from ar.homography import HomographyEstimator


class PlaneTracker:
    """Detect and track the active planar game-board marker."""

    A4_WIDTH = 210
    A4_HEIGHT = 297
    DOOR_BOARD_SIZE = 150
    MARK_INSET = 15

    def __init__(self, detector_mode="door_marker"):
        if detector_mode not in ("door_marker", "corner_marks"):
            raise ValueError("detector_mode must be 'door_marker' or 'corner_marks'")
        self.detector_mode = detector_mode
        self.active_detector_mode = None
        self.is_registered = False
        self.last_corners = None
        self.last_homography = None
        self.smoothed_homography = None
        self.last_gray = None
        self.missed_frames = 0
        self.max_missed_frames = 18
        self.corner_smoothing = 0.35
        self.homography_ema_alpha = 0.35
        self.min_homography_confidence = 0.28
        self.redetection_interval = 6
        self.last_homography_confidence = 0.0
        self.last_marker_observed = None
        self.track_patch_radius = 8
        self.track_search_radius = 28
        self.min_track_score = 0.58
        self.last_marker_centers = None
        self.marker_lock_radius = 180.0
        self.marker_roi_radius = 108
        self.frame_index = 0
        self.last_door_image_points = None
        self.last_door_world_points = None
        self.door_detection_max_dim = 720
        self.door_canonical_size = 320
        self.door_redetection_interval = 8
        self.door_stable_redetection_interval = 18
        self.door_unstable_redetection_interval = 5
        self.current_plane_size = (
            (self.DOOR_BOARD_SIZE, self.DOOR_BOARD_SIZE)
            if detector_mode == "door_marker"
            else (self.A4_WIDTH, self.A4_HEIGHT)
        )
        self.hand_occlusion_mask = None
        self.debug_marker_candidates = []
        self.debug_marker_display_cache = []
        self.debug_enabled = False
        self.initial_detection_max_dim = 720
        self.last_reject_reason = None
        self.last_candidate_count = 0
        self.last_white_validation_score = None
        self.last_marker_quad_score = None
        self.last_selected_marker_centers = None

    def register_plane(self, frame):
        result = self.track_plane(frame)
        if not result["success"]:
            print("Board registration failed. Show the complete gate marker in the camera view.")
            return False

        self.register_tracking_result(result)
        print("Game board registered.")
        return True

    def register_tracking_result(self, result):
        if not result or not result.get("success"):
            return False

        self.is_registered = True
        self.last_corners = result["corners"]
        self.last_homography = result["H"]
        self.smoothed_homography = result["H"]
        self.current_plane_size = tuple(result.get("plane_size", self.current_plane_size))
        self.last_homography_confidence = result.get("homography_confidence", result.get("track_score", 1.0))
        if result.get("marker_centers") is not None:
            self.last_marker_centers = result["marker_centers"]
        if result.get("marker_observed") is not None:
            self.last_marker_observed = result["marker_observed"]
        if result.get("door_image_points") is not None:
            self.last_door_image_points = np.asarray(result["door_image_points"], dtype=np.float32)
        if result.get("door_world_points") is not None:
            self.last_door_world_points = np.asarray(result["door_world_points"], dtype=np.float32)
        self.active_detector_mode = result.get("detector_mode", self.detector_mode)
        return True

    def track_plane(self, frame, hand_landmarks=None, debug=False):
        self.frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.debug_enabled = bool(debug or not self.is_registered)
        self.hand_occlusion_mask = self._build_hand_occlusion_mask(hand_landmarks, gray.shape)
        self.last_reject_reason = None

        if not self.is_registered:
            self.debug_marker_candidates = []
            if self.detector_mode == "door_marker":
                door_result = self._detect_door_marker(gray)
                if door_result is not None:
                    return self._accept_marker_result(door_result, gray, "door_marker")
                return self._failure_result()

            marker_result = self._detect_corner_marks_global(
                gray,
                allow_partial=True,
                require_white_validation=True,
                allow_inferred_missing=True,
            )
            if marker_result is not None:
                method = "corner_marks" if marker_result.get("matched_points", 0) >= 4 else "corner_marks_partial"
                return self._accept_marker_result(marker_result, gray, method)
            return self._failure_result()

        if (self.active_detector_mode or self.detector_mode) == "door_marker":
            if self.frame_index % self._current_door_redetection_interval() == 0:
                door_result = self._detect_door_marker(gray, search_near_last=True)
                if door_result is not None:
                    return self._accept_marker_result(door_result, gray, "door_redetect")

            tracked = self._track_last_door_points(gray)
            if tracked is not None:
                return tracked

            door_result = self._detect_door_marker(gray, search_near_last=self.missed_frames < 3)
            if door_result is not None:
                return self._accept_marker_result(door_result, gray, "door_redetect")
        else:
            marker_result = self._detect_corner_marks(gray)
            if marker_result is not None:
                return self._accept_marker_result(marker_result, gray, "corner_marks")

            tracked = self._tracking_fallback(gray)
            if tracked is not None:
                return tracked
        self.missed_frames += 1
        if (
            self.last_corners is not None
            and self.last_homography is not None
            and self.missed_frames <= self.max_missed_frames
        ):
            active_mode = self.active_detector_mode or self.detector_mode
            marker_centers = (
                None
                if active_mode == "door_marker"
                else self._marker_points_from_homography(self.last_homography, self.current_plane_size)
            )
            return {
                "success": True,
                "H": self.last_homography,
                "matched_points": 4,
                "corners": self.last_corners,
                "marker_centers": marker_centers,
                "observed_marker_centers": None,
                "stale": True,
                "tracking_method": "hold_last",
                "track_score": 0.0,
                "homography_confidence": max(0.0, self.last_homography_confidence * 0.45),
                "marker_observed": self.last_marker_observed,
                "detector_mode": self.active_detector_mode or self.detector_mode,
                "plane_size": self.current_plane_size,
            }
        return self._failure_result()

    def _failure_result(self):
        return {
            "success": False,
            "H": None,
            "matched_points": 0,
            "corners": None,
            "marker_candidates": self._pruned_debug_marker_candidates(),
            "reject_reason": self.last_reject_reason,
            "candidate_count": self.last_candidate_count,
            "white_validation_score": self.last_white_validation_score,
            "marker_quad_score": self.last_marker_quad_score,
            "selected_marker_centers": self.last_selected_marker_centers,
        }

    def _current_door_redetection_interval(self):
        if self.missed_frames > 0 or self.last_homography_confidence < 0.62:
            return self.door_unstable_redetection_interval
        if self.last_homography_confidence >= 0.78:
            return self.door_stable_redetection_interval
        return self.door_redetection_interval

    def _detect_door_marker(self, gray, search_near_last=False):
        """Detect the single square gate marker used as the 150 mm board."""
        self.debug_marker_candidates = []
        self.last_candidate_count = 0
        self.last_marker_quad_score = None
        self.last_selected_marker_centers = None
        self.last_reject_reason = None

        source_height, source_width = gray.shape[:2]
        roi_x = 0
        roi_y = 0
        roi_gray = gray
        if search_near_last and self.last_corners is not None:
            corners = np.asarray(self.last_corners, dtype=np.float32)
            x_min = max(0, int(np.floor(np.min(corners[:, 0]) - 80)))
            y_min = max(0, int(np.floor(np.min(corners[:, 1]) - 80)))
            x_max = min(source_width, int(np.ceil(np.max(corners[:, 0]) + 80)))
            y_max = min(source_height, int(np.ceil(np.max(corners[:, 1]) + 80)))
            if x_max > x_min + 80 and y_max > y_min + 80:
                roi_x, roi_y = x_min, y_min
                roi_gray = gray[y_min:y_max, x_min:x_max]

        scale = 1.0
        work_gray = roi_gray
        max_dim = max(work_gray.shape[:2])
        if max_dim > self.door_detection_max_dim:
            scale = self.door_detection_max_dim / float(max_dim)
            work_gray = cv2.resize(
                work_gray,
                (int(round(work_gray.shape[1] * scale)), int(round(work_gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        blurred = cv2.GaussianBlur(work_gray, (5, 5), 0)
        _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:16]
        self.last_candidate_count = len(contours)

        best = None
        for contour in contours:
            ordered_work, contour_score = self._door_candidate_corners(contour, work_gray.shape)
            if ordered_work is None:
                continue

            ordered = ordered_work / max(scale, 1e-6)
            ordered[:, 0] += roi_x
            ordered[:, 1] += roi_y
            if not self._validate_door_corners(ordered, gray.shape):
                continue

            for rotation in range(4):
                oriented = np.roll(ordered, -rotation, axis=0).astype(np.float32)
                patch = self._warp_door_candidate(gray, oriented)
                validation = self._validate_door_patch(patch)
                if validation is None or validation.get("direction_index") != 2:
                    continue

                source = np.asarray(
                    [
                        [0.0, 0.0],
                        [self.DOOR_BOARD_SIZE, 0.0],
                        [self.DOOR_BOARD_SIZE, self.DOOR_BOARD_SIZE],
                        [0.0, self.DOOR_BOARD_SIZE],
                    ],
                    dtype=np.float32,
                )
                H = HomographyEstimator.compute_homography(source, oriented)
                H = self._normalize_homography(H)
                corners = self._board_corners_from_homography(H, (self.DOOR_BOARD_SIZE, self.DOOR_BOARD_SIZE))
                if not self._validate_door_corners(corners, gray.shape):
                    continue

                area_ratio = self._polygon_area(corners) / max(float(source_height * source_width), 1.0)
                area_score = float(np.clip((area_ratio - 0.015) / 0.25, 0.0, 1.0))
                confidence = float(np.clip(validation["score"] * 0.78 + contour_score * 0.12 + area_score * 0.10, 0.0, 1.0))
                if confidence < 0.48:
                    continue

                image_points, world_points = self._door_tracking_features(gray, H, corners)
                candidate = {
                    "H": H,
                    "corners": corners.astype(np.float32),
                    "marker_centers": corners.astype(np.float32),
                    "observed_marker_centers": corners.astype(np.float32),
                    "marker_observed": None,
                    "score": confidence,
                    "homography_confidence": confidence,
                    "reprojection_error": 0.0,
                    "matched_points": 4,
                    "plane_size": (self.DOOR_BOARD_SIZE, self.DOOR_BOARD_SIZE),
                    "detector_mode": "door_marker",
                    "selected_marker_centers": corners.astype(np.float32),
                    "candidate_count": len(contours),
                    "door_symbol_score": validation["symbol_score"],
                    "door_direction_score": validation["direction_score"],
                    "door_image_points": image_points,
                    "door_world_points": world_points,
                }
                self.debug_marker_candidates.append({"point": np.mean(corners, axis=0), "accepted": True, "slot": "gate"})
                if best is None or candidate["homography_confidence"] > best["homography_confidence"]:
                    best = candidate

        if best is None:
            self.last_reject_reason = "gate_marker_not_detected"
            return None

        self.last_marker_quad_score = best["homography_confidence"]
        self.last_selected_marker_centers = best["corners"].copy()
        return best

    def _door_candidate_corners(self, contour, image_shape):
        area = float(cv2.contourArea(contour))
        image_area = float(image_shape[0] * image_shape[1])
        if area < max(900.0, image_area * 0.012) or area > image_area * 0.88:
            return None, 0.0

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            return None, 0.0
        approx = cv2.approxPolyDP(contour, perimeter * 0.028, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            corners = box

        ordered = self._order_quad_points(corners)
        if ordered is None or not self._is_convex_quad(ordered):
            return None, 0.0
        edge_lengths = np.asarray(
            [np.linalg.norm(ordered[(idx + 1) % 4] - ordered[idx]) for idx in range(4)],
            dtype=np.float32,
        )
        if float(np.min(edge_lengths)) < 36.0:
            return None, 0.0
        edge_ratio = float(np.max(edge_lengths) / max(np.min(edge_lengths), 1.0))
        if edge_ratio > 2.8:
            return None, 0.0
        rectangularity = area / max(cv2.contourArea(ordered.astype(np.float32)), 1.0)
        score = float(np.clip((1.0 / edge_ratio) * 0.62 + np.clip(rectangularity, 0.0, 1.0) * 0.38, 0.0, 1.0))
        return ordered, score

    def _order_quad_points(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (4, 2):
            return None
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[int(np.argmin(sums))]
        ordered[2] = points[int(np.argmax(sums))]
        ordered[1] = points[int(np.argmin(diffs))]
        ordered[3] = points[int(np.argmax(diffs))]
        if len({tuple(np.round(point, 2)) for point in ordered}) != 4:
            center = np.mean(points, axis=0)
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            ordered = points[np.argsort(angles)].astype(np.float32)
            start = int(np.argmin(ordered.sum(axis=1)))
            ordered = np.roll(ordered, -start, axis=0)
        return ordered.astype(np.float32)

    def _warp_door_candidate(self, gray, corners):
        size = int(self.door_canonical_size)
        destination = np.asarray(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), destination)
        return cv2.warpPerspective(gray, transform, (size, size), flags=cv2.INTER_LINEAR)

    def _validate_door_patch(self, patch):
        if patch is None or patch.size == 0:
            return None
        size = int(patch.shape[0])
        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        dark = (dark > 0).astype(np.uint8)

        edge_lo = max(2, int(round(size * 0.012)))
        edge_hi = max(edge_lo + 3, int(round(size * 0.10)))
        along_lo = int(round(size * 0.10))
        along_hi = int(round(size * 0.90))
        border_ratios = [
            float(np.mean(dark[edge_lo:edge_hi, along_lo:along_hi])),
            float(np.mean(dark[size - edge_hi:size - edge_lo, along_lo:along_hi])),
            float(np.mean(dark[along_lo:along_hi, edge_lo:edge_hi])),
            float(np.mean(dark[along_lo:along_hi, size - edge_hi:size - edge_lo])),
        ]
        min_border = min(border_ratios)
        if min_border < 0.055:
            return None
        border_score = float(np.clip((min_border - 0.055) / 0.24, 0.0, 1.0))

        interior_lo = int(round(size * 0.16))
        interior_hi = int(round(size * 0.84))
        interior_dark = float(np.mean(dark[interior_lo:interior_hi, interior_lo:interior_hi]))
        if interior_dark > 0.34:
            return None
        white_score = float(np.clip((0.34 - interior_dark) / 0.30, 0.0, 1.0))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=size * 0.18,
            param1=105,
            param2=15,
            minRadius=max(10, int(round(size * 0.055))),
            maxRadius=max(18, int(round(size * 0.18))),
        )
        if circles is None:
            return None

        yy, xx = np.indices((size, size), dtype=np.float32)
        best = None
        for circle in circles[0][:8]:
            cx, cy, radius = (float(circle[0]), float(circle[1]), float(circle[2]))
            if not (size * 0.30 <= cx <= size * 0.70 and size * 0.26 <= cy <= size * 0.70):
                continue
            distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            annulus = (distance >= radius * 0.72) & (distance <= radius * 1.30)
            core = distance <= radius * 0.52
            annulus_dark = float(np.mean(dark[annulus])) if np.any(annulus) else 0.0
            core_dark = float(np.mean(dark[core])) if np.any(core) else 1.0
            if annulus_dark < 0.10 or core_dark > 0.46:
                continue
            circle_score = float(
                np.clip((annulus_dark - 0.10) / 0.40, 0.0, 1.0) * 0.74
                + np.clip((0.46 - core_dark) / 0.40, 0.0, 1.0) * 0.26
            )
            direction = self._door_direction_scores(dark, (cx, cy), radius)
            if direction is None:
                continue
            score = border_score * 0.22 + circle_score * 0.36 + direction["score"] * 0.34 + white_score * 0.08
            candidate = {
                "score": float(np.clip(score, 0.0, 1.0)),
                "symbol_score": circle_score,
                "direction_score": direction["score"],
                "direction_index": direction["index"],
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _door_direction_scores(self, dark, center, radius):
        size = int(dark.shape[0])
        center = np.asarray(center, dtype=np.float32)
        directions = np.asarray([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
        perpendiculars = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        scores = []
        strengths = []
        for direction, perpendicular in zip(directions, perpendiculars):
            if direction[0] > 0:
                border_distance = size - 1 - center[0]
            elif direction[0] < 0:
                border_distance = center[0]
            elif direction[1] > 0:
                border_distance = size - 1 - center[1]
            else:
                border_distance = center[1]
            line_start = radius + size * 0.014
            line_end = min(radius + size * 0.34, border_distance - size * 0.10)
            if line_end <= line_start + size * 0.045:
                scores.append(-1.0)
                strengths.append(0.0)
                continue
            line_mask = self._door_strip_mask(
                dark.shape,
                center,
                direction,
                perpendicular,
                line_start,
                line_end,
                size * 0.030,
            )
            strength = float(np.mean(dark[line_mask])) if np.any(line_mask) else 0.0
            strengths.append(strength)
            scores.append(strength)

        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_score = float(scores[best_index])
        second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0
        margin = best_score - second_score
        if best_score < 0.085 or margin < 0.018:
            return None
        normalized = float(
            np.clip((best_score - 0.085) / 0.34, 0.0, 1.0) * 0.72
            + np.clip(margin / 0.22, 0.0, 1.0) * 0.28
        )
        return {"index": best_index, "score": normalized}

    def _door_strip_mask(self, shape, center, direction, perpendicular, start, end, half_width):
        mask = np.zeros(shape, dtype=np.uint8)
        if end <= start:
            return mask.astype(bool)
        points = np.asarray(
            [
                center + direction * start - perpendicular * half_width,
                center + direction * start + perpendicular * half_width,
                center + direction * end + perpendicular * half_width,
                center + direction * end - perpendicular * half_width,
            ],
            dtype=np.float32,
        )
        cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 1, cv2.LINE_8)
        return mask.astype(bool)

    def _validate_door_corners(self, corners, image_shape):
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape != (4, 2):
            return False
        height, width = image_shape[:2]
        if np.any(corners[:, 0] < -2) or np.any(corners[:, 0] >= width + 2):
            return False
        if np.any(corners[:, 1] < -2) or np.any(corners[:, 1] >= height + 2):
            return False
        if self._polygon_area(corners) < 800.0 or not self._is_convex_quad(corners):
            return False
        edges = np.asarray([np.linalg.norm(corners[(idx + 1) % 4] - corners[idx]) for idx in range(4)], dtype=np.float32)
        if float(np.min(edges)) < 36.0:
            return False
        if float(np.max(edges) / max(np.min(edges), 1.0)) > 3.2:
            return False
        diagonal_a = float(np.linalg.norm(corners[2] - corners[0]))
        diagonal_b = float(np.linalg.norm(corners[3] - corners[1]))
        return max(diagonal_a, diagonal_b) / max(min(diagonal_a, diagonal_b), 1.0) <= 2.3

    def _door_tracking_features(self, gray, H, corners):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255, cv2.LINE_8)
        mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=80,
            qualityLevel=0.006,
            minDistance=6,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if features is None:
            image_points = np.empty((0, 2), dtype=np.float32)
        else:
            image_points = features.reshape(-1, 2).astype(np.float32)
        image_points = np.vstack([np.asarray(corners, dtype=np.float32), image_points])
        try:
            inverse_h = np.linalg.inv(np.asarray(H, dtype=np.float64))
        except np.linalg.LinAlgError:
            return None, None
        world_points = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2), inverse_h.astype(np.float32)
        ).reshape(-1, 2)
        size = float(self.DOOR_BOARD_SIZE)
        valid = (
            np.isfinite(world_points).all(axis=1)
            & (world_points[:, 0] >= -4.0)
            & (world_points[:, 0] <= size + 4.0)
            & (world_points[:, 1] >= -4.0)
            & (world_points[:, 1] <= size + 4.0)
        )
        return image_points[valid].astype(np.float32), world_points[valid].astype(np.float32)

    def _track_last_door_points(self, gray):
        if self.last_gray is None or self.last_door_image_points is None or self.last_door_world_points is None:
            return None
        previous = np.asarray(self.last_door_image_points, dtype=np.float32)
        world = np.asarray(self.last_door_world_points, dtype=np.float32)
        if previous.shape != world.shape or previous.ndim != 2 or previous.shape[0] < 8:
            return None

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.last_gray,
            gray,
            previous.reshape(-1, 1, 2),
            None,
            winSize=(27, 27),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.02),
        )
        if next_points is None or status is None:
            return None
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.last_gray,
            next_points,
            None,
            winSize=(27, 27),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.03),
        )
        if backward is None or backward_status is None:
            return None

        tracked = next_points.reshape(-1, 2).astype(np.float32)
        backward = backward.reshape(-1, 2).astype(np.float32)
        valid = status.reshape(-1).astype(bool) & backward_status.reshape(-1).astype(bool)
        forward_backward_error = np.linalg.norm(backward - previous, axis=1)
        valid &= np.isfinite(forward_backward_error) & (forward_backward_error <= 2.4)
        if errors is not None:
            error_values = errors.reshape(-1)
            valid &= np.isfinite(error_values) & (error_values <= 32.0)
        valid &= ~self._points_inside_hand(tracked)
        if int(np.count_nonzero(valid)) < max(7, int(previous.shape[0] * 0.32)):
            return None

        valid_world = world[valid]
        valid_image = tracked[valid]
        H, inlier_mask = cv2.findHomography(valid_world, valid_image, cv2.RANSAC, 3.2)
        if H is None or inlier_mask is None:
            return None
        inliers = inlier_mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(inliers)) < max(7, int(valid_world.shape[0] * 0.58)):
            return None

        inlier_world = valid_world[inliers]
        inlier_image = valid_image[inliers]
        H = self._normalize_homography(H)
        plane_size = (self.DOOR_BOARD_SIZE, self.DOOR_BOARD_SIZE)
        corners = self._board_corners_from_homography(H, plane_size)
        if not self._validate_door_corners(corners, gray.shape):
            return None

        projected = self._project_points(inlier_world, H)
        if projected is None:
            return None
        reprojection_error = float(np.mean(np.linalg.norm(projected - inlier_image, axis=1)))
        if reprojection_error > 4.8:
            return None

        observed_ratio = inlier_image.shape[0] / max(float(previous.shape[0]), 1.0)
        confidence = float(
            np.clip(
                (1.0 / (1.0 + reprojection_error * 0.50)) * 0.62
                + observed_ratio * 0.25
                + np.clip(1.0 - float(np.mean(forward_backward_error[valid])) / 2.4, 0.0, 1.0) * 0.13,
                0.0,
                0.91,
            )
        )
        result = {
            "H": H,
            "corners": corners.astype(np.float32),
            "marker_centers": corners.astype(np.float32),
            "observed_marker_centers": corners.astype(np.float32),
            "marker_observed": None,
            "score": confidence,
            "homography_confidence": confidence,
            "reprojection_error": reprojection_error,
            "matched_points": int(inlier_image.shape[0]),
            "plane_size": plane_size,
            "detector_mode": "door_marker",
            "door_image_points": inlier_image.copy(),
            "door_world_points": inlier_world.copy(),
            "selected_marker_centers": corners.astype(np.float32),
            "candidate_count": int(inlier_image.shape[0]),
        }
        return self._accept_marker_result(result, gray, "door_flow")

    def _project_points(self, points, H):
        points = np.asarray(points, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
        projected = (np.asarray(H, dtype=np.float32) @ homogeneous.T).T
        if not bool(np.all(np.abs(projected[:, 2]) > 1e-8)):
            return None
        return (projected[:, :2] / projected[:, 2:3]).astype(np.float32)

    def _preview_result_from_corners(self, corners, method, score):
        plane_size = self._plane_size_for_image_quad(corners)
        source = self._board_points_for_size(plane_size)
        H = HomographyEstimator.compute_homography(source, corners)
        return {
            "success": True,
            "H": H,
            "matched_points": 4,
            "corners": corners.astype(np.float32),
            "marker_centers": None,
            "stale": False,
            "tracking_method": method,
            "track_score": score,
            "homography_confidence": score,
            "marker_observed": None,
            "plane_size": plane_size,
        }

    def _preview_result_with_marker_assist(self, corners, gray):
        result = self._preview_result_from_corners(corners, "white_boundary_marker_assist", 0.58)
        match = self._detect_expected_markers_near_boundary(gray, result["H"], result["plane_size"])
        if match is None:
            return None

        observed, observed_centers = match
        matched_count = int(np.count_nonzero(observed))
        if matched_count < 2:
            return None

        marker_centers = self._marker_points_from_homography(result["H"], result["plane_size"])
        if marker_centers is None:
            return None

        if matched_count >= 3:
            mixed_centers = marker_centers.copy()
            mixed_centers[observed] = observed_centers[observed]
            refined_H = HomographyEstimator.compute_homography(
                self._marker_board_points(result["plane_size"]),
                mixed_centers,
            )
            refined_corners = self._board_corners_from_homography(refined_H, result["plane_size"])
            if self._validate_tracked_corners(refined_corners, gray.shape):
                result["H"] = refined_H
                result["corners"] = refined_corners
                marker_centers = self._marker_points_from_homography(refined_H, result["plane_size"])

        result["matched_points"] = matched_count
        result["marker_centers"] = marker_centers
        result["observed_marker_centers"] = observed_centers
        result["marker_observed"] = observed
        result["track_score"] = 0.52 + matched_count * 0.08
        result["homography_confidence"] = result["track_score"]
        return result

    def _detect_expected_markers_near_boundary(self, gray, H, plane_size):
        expected = self._marker_points_from_homography(H, plane_size)
        if expected is None:
            return None

        labels = ("tl", "tr", "br", "bl")
        observed = np.zeros(4, dtype=bool)
        observed_centers = expected.copy()
        for index, (slot_name, point) in enumerate(zip(labels, expected)):
            marker, score = self._detect_one_mark_near(
                gray,
                point,
                slot_name,
                expected_quad=expected,
            )
            if marker is None:
                self.debug_marker_candidates.append(
                    {
                        "point": point.astype(np.float32),
                        "slot": slot_name,
                        "accepted": False,
                    }
                )
                continue
            observed[index] = True
            observed_centers[index] = marker
            self.debug_marker_candidates.append(
                {
                    "point": marker.astype(np.float32),
                    "slot": slot_name,
                    "accepted": True,
                    "score": score,
                }
            )

        return observed, observed_centers

    def _match_candidates_to_expected_markers(self, H, plane_size):
        expected = self._marker_points_from_homography(H, plane_size)
        if expected is None:
            return None

        candidates = self.debug_marker_candidates or []
        if not candidates:
            return None

        labels = ("tl", "tr", "br", "bl")
        observed = np.zeros(4, dtype=bool)
        observed_centers = expected.copy()
        radius = self._marker_assist_radius(expected)

        for index, slot_name in enumerate(labels):
            best_point = None
            best_score = None
            for candidate in candidates:
                if candidate.get("slot") != slot_name:
                    continue
                point = np.asarray(candidate.get("point"), dtype=np.float32)
                if point.shape != (2,):
                    continue
                distance = float(np.linalg.norm(point - expected[index]))
                if distance > radius:
                    continue
                shape_bonus = float(candidate.get("score", 0.0)) * 12.0 if candidate.get("accepted") else 0.0
                score = distance - shape_bonus
                if best_score is None or score < best_score:
                    best_score = score
                    best_point = point

            if best_point is not None:
                observed[index] = True
                observed_centers[index] = best_point

        return observed, observed_centers

    def _marker_assist_radius(self, expected):
        edge_lengths = [
            np.linalg.norm(expected[(idx + 1) % 4] - expected[idx])
            for idx in range(4)
        ]
        median_edge = float(np.median(edge_lengths)) if edge_lengths else 0.0
        return float(np.clip(median_edge * 0.16, 24.0, 72.0))

    def _result_from_corners(self, corners, gray, method, score):
        corners = self._stabilize_corners(corners)
        plane_size = self._plane_size_for_image_quad(corners)
        source = self._board_points_for_size(plane_size)
        H = HomographyEstimator.compute_homography(source, corners)
        self.current_plane_size = plane_size
        self.last_corners = corners
        self.last_homography = H
        self.last_gray = gray
        self.last_homography_confidence = score
        self.missed_frames = 0
        return {
            "success": True,
            "H": H,
            "matched_points": 4,
            "corners": corners,
            "marker_centers": None,
            "stale": False,
            "tracking_method": method,
            "track_score": score,
            "homography_confidence": score,
            "marker_observed": None,
            "plane_size": plane_size,
        }

    def _tracking_fallback(self, gray):
        marker_result = self._track_last_markers(gray)
        if marker_result is not None:
            return marker_result

        if self.missed_frames >= 3 and self.frame_index % self.redetection_interval == 0:
            marker_result = self._detect_corner_marks_global(gray, allow_partial=True)
            if marker_result is not None:
                return self._accept_marker_result(marker_result, gray, "redetect_corner_marks")
        return None

    def _track_last_markers(self, gray):
        if self.last_marker_centers is None or self.last_gray is None:
            return None

        previous_points = np.asarray(self.last_marker_centers, dtype=np.float32).reshape(-1, 1, 2)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.last_gray,
            gray,
            previous_points,
            None,
            winSize=(31, 31),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            return None

        observed = status.reshape(-1).astype(bool)
        valid_count = int(np.count_nonzero(observed))
        if valid_count < 3:
            return None

        marker_centers = next_points.reshape(-1, 2).astype(np.float32)
        observed = observed & ~self._points_inside_hand(marker_centers)
        valid_count = int(np.count_nonzero(observed))
        if valid_count < 3:
            return None

        valid_movement = marker_centers[observed] - self.last_marker_centers[observed]
        if np.max(np.linalg.norm(valid_movement, axis=1)) > self.marker_roi_radius * 2.4:
            return None

        predicted = self._predicted_marker_points()
        marker_scores = []
        for index, slot_name in enumerate(("tl", "tr", "br", "bl")):
            if not observed[index]:
                marker_scores.append(0.0)
                continue
            verified_marker, marker_score = self._detect_one_mark_near(
                gray,
                marker_centers[index],
                slot_name,
                expected_quad=predicted,
            )
            if verified_marker is None:
                observed[index] = False
                marker_scores.append(0.0)
                continue
            marker_centers[index] = verified_marker
            marker_scores.append(marker_score)

        valid_count = int(np.count_nonzero(observed))
        if valid_count < 3:
            return None

        if valid_count < 4:
            if predicted is not None:
                marker_centers[~observed] = predicted[~observed]
            else:
                median_displacement = np.median(marker_centers[observed] - self.last_marker_centers[observed], axis=0).astype(np.float32)
                marker_centers[~observed] = self.last_marker_centers[~observed] + median_displacement

        if self._order_quad_points(marker_centers) is None:
            return None

        error_values = errors.reshape(-1)[observed] if errors is not None else np.zeros(valid_count, dtype=np.float32)
        flow_score = 1.0 / (1.0 + float(np.mean(error_values)))
        observed_scores = [score for score, is_observed in zip(marker_scores, observed) if is_observed]
        if observed_scores:
            flow_score = (flow_score * 0.45) + (float(np.mean(observed_scores)) * 0.55)
        marker_result = self._build_marker_result(marker_centers, observed, gray.shape, flow_score)
        if marker_result is None:
            return None
        return self._accept_marker_result(marker_result, gray, "optical_flow")

    def _detect_corner_marks(self, gray):
        if self.is_registered and self.last_homography is not None:
            predicted_result = self._detect_corner_marks_near_prediction(gray)
            if predicted_result is not None:
                return predicted_result
            return None
        return self._detect_corner_marks_global(gray, allow_partial=self.is_registered)

    def _detect_corner_marks_global(
        self,
        gray,
        allow_partial=False,
        require_white_validation=False,
        allow_inferred_missing=False,
    ):
        self.debug_marker_candidates = []
        self.last_candidate_count = 0
        self.last_white_validation_score = None
        self.last_marker_quad_score = None
        self.last_selected_marker_centers = None
        source_shape = gray.shape
        coord_scale = 1.0
        work_gray = gray
        if not self.is_registered:
            max_dim = max(gray.shape[:2])
            if max_dim > self.initial_detection_max_dim:
                resize_scale = self.initial_detection_max_dim / float(max_dim)
                work_gray = cv2.resize(
                    gray,
                    (int(round(gray.shape[1] * resize_scale)), int(round(gray.shape[0] * resize_scale))),
                    interpolation=cv2.INTER_AREA,
                )
                coord_scale = 1.0 / resize_scale

        mask = self._dark_mark_mask(work_gray)
        image_area = work_gray.shape[0] * work_gray.shape[1]
        components = self._connected_components(mask, min_pixels=max(5, int(image_area * 0.00001)))
        if not components:
            return None
        if len(components) > 90:
            components = sorted(components, key=len, reverse=True)[:90]

        height, width = work_gray.shape[:2]
        frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        image_area = width * height
        anchors = self._marker_anchors(width, height)
        if coord_scale != 1.0 and self.last_marker_centers is not None:
            anchors = {name: anchor / coord_scale for name, anchor in anchors.items()}
        slots = {name: {"anchor": anchor, "candidates": []} for name, anchor in anchors.items()}

        for component in components:
            if coord_scale == 1.0 and self._component_overlaps_hand(component):
                continue
            candidate = self._component_to_mark_candidate(component, image_area, width, height)
            if candidate is None:
                continue
            if allow_partial and self.last_marker_centers is not None:
                slot_names = [self._quadrant_corner_slot(candidate["blob_center"], frame_center)]
            elif self.is_registered and self.last_marker_centers is not None:
                slot_names = [self._candidate_corner_slot(candidate["blob_center"], frame_center)]
            else:
                slot_names = ["tl", "tr", "br", "bl"]

            component_debug = []
            component_accepted = []
            for slot_name in slot_names:
                if slot_name is None:
                    continue
                mark_corner = self._estimate_l_mark_corner(component, slot_name)
                if mark_corner is None:
                    continue
                display_corner = (mark_corner * coord_scale).astype(np.float32)
                debug_candidate = {
                    "point": display_corner,
                    "slot": slot_name,
                    "accepted": False,
                }
                component_debug.append(debug_candidate)
                shape_analysis = self._analyze_l_mark_shape(component, mark_corner, slot_name)
                if shape_analysis is None:
                    continue

                candidate_for_slot = dict(candidate)
                candidate_for_slot["center"] = mark_corner
                candidate_for_slot["shape_score"] = shape_analysis["score"]
                candidate_for_slot["open_vector"] = shape_analysis["open_vector"]
                candidate_for_slot["arm_angle"] = shape_analysis["arm_angle"]
                component_accepted.append(
                    {
                        "debug": debug_candidate,
                        "candidate": candidate_for_slot,
                        "slot": slot_name,
                        "score": shape_analysis["score"],
                    }
                )

            if component_accepted:
                best = max(component_accepted, key=lambda item: item["score"])
                best["debug"]["accepted"] = True
                best["debug"]["score"] = best["score"]
                best["debug"]["open_vector"] = best["candidate"].get("open_vector")
                best["debug"]["arm_angle"] = best["candidate"].get("arm_angle")
                self.debug_marker_candidates.append(best["debug"])
                slots[best["slot"]]["candidates"].append(best["candidate"])
            elif component_debug:
                self.debug_marker_candidates.append(component_debug[0])

        self.last_candidate_count = sum(len(slot["candidates"]) for slot in slots.values())
        predicted = self._predicted_marker_points() if allow_partial else None
        selected_quad = self._select_marker_quad(
            slots,
            predicted,
            allow_partial,
            frame_center,
            image_area,
            allow_inferred_missing=allow_inferred_missing,
        )
        if selected_quad is None:
            self.last_reject_reason = "marker_quad_rejected"
            return None

        marker_centers, selected, observed, score = selected_quad
        ordered = np.asarray(marker_centers, dtype=np.float32) * coord_scale
        observed = np.asarray(observed, dtype=bool)
        candidate_score = 1.0 / (1.0 + float(score))
        self.last_marker_quad_score = float(candidate_score)
        self.last_selected_marker_centers = ordered.copy()

        result = self._build_marker_result(ordered, observed, source_shape, candidate_score)
        if result is None:
            self.last_reject_reason = "homography_rejected"
            return None

        if require_white_validation:
            validation = self._validate_projected_white_board(gray, result["H"], result["plane_size"])
            result["white_validation_score"] = validation["score"]
            result["white_validation"] = validation
            self.last_white_validation_score = validation["score"]
            if not validation["ok"]:
                self.last_reject_reason = validation["reason"]
                return None

        result["marker_quad_score"] = candidate_score
        result["candidate_count"] = self.last_candidate_count
        result["selected_marker_centers"] = ordered.copy()
        result["reject_reason"] = None
        return result

    def _detect_corner_marks_near_prediction(self, gray):
        predicted = self._predicted_marker_points()
        if predicted is None or predicted.shape != (4, 2):
            return None

        marker_centers = []
        scores = []
        observed = []
        for slot_name, point in zip(("tl", "tr", "br", "bl"), predicted):
            marker, score = self._detect_one_mark_near(
                gray,
                point,
                slot_name,
                expected_quad=predicted,
            )
            if marker is None:
                marker_centers.append(point)
                scores.append(0.0)
                observed.append(False)
            else:
                marker_centers.append(marker)
                scores.append(score)
                observed.append(True)

        observed = np.asarray(observed, dtype=bool)
        if int(np.count_nonzero(observed)) < 3:
            return None

        ordered = np.asarray(marker_centers, dtype=np.float32)
        if self._order_quad_points(ordered) is None:
            return None

        frame_center = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0], dtype=np.float32)
        dummy_group = [{"area": 1.0} for _ in range(4)]
        if self._score_marker_quad(ordered, dummy_group, frame_center, gray.shape[0] * gray.shape[1]) is None:
            return None

        observed_scores = [score for score, is_observed in zip(scores, observed) if is_observed]
        candidate_score = float(np.mean(observed_scores)) if observed_scores else 0.0
        return self._build_marker_result(ordered, observed, gray.shape, candidate_score)

    def _detect_one_mark_near(self, gray, predicted, slot_name, expected_quad=None):
        if self._point_inside_hand(predicted):
            return None, 0.0

        height, width = gray.shape[:2]
        radius = self.marker_roi_radius
        cx = int(round(predicted[0]))
        cy = int(round(predicted[1]))
        x1 = max(cx - radius, 0)
        y1 = max(cy - radius, 0)
        x2 = min(cx + radius + 1, width)
        y2 = min(cy + radius + 1, height)
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None, 0.0

        roi = gray[y1:y2, x1:x2]
        mask = self._dark_mark_mask(roi)
        roi_area = roi.shape[0] * roi.shape[1]
        components = self._connected_components(mask, min_pixels=max(5, int(roi_area * 0.0004)))

        best = None
        best_score = None
        for component in components:
            global_component = component + np.array([x1, y1], dtype=np.float32)
            if self._component_overlaps_hand(global_component):
                continue
            candidate = self._component_to_mark_candidate(
                component,
                roi_area,
                roi.shape[1],
                roi.shape[0],
                apply_border_margin=False,
            )
            if candidate is None:
                continue

            mark_corner = self._estimate_l_mark_corner(component, slot_name)
            if mark_corner is None:
                continue
            shape_analysis = self._analyze_l_mark_shape(component, mark_corner, slot_name)
            if shape_analysis is None:
                continue

            candidate_center = mark_corner + np.array([x1, y1], dtype=np.float32)
            distance = np.linalg.norm(candidate_center - predicted)
            if distance > radius * 0.85:
                continue

            opening_score = self._score_marker_opening_against_quad(
                shape_analysis["open_vector"],
                candidate_center,
                expected_quad,
                slot_name,
            )
            if expected_quad is not None and opening_score is None:
                continue

            area_bonus = min(candidate["area"], 1000.0) * 0.018
            fill_penalty = abs(candidate["fill_ratio"] - 0.35) * 55.0
            shape_bonus = shape_analysis["score"] * 42.0
            opening_bonus = 0.0 if opening_score is None else opening_score * 34.0
            score = distance + fill_penalty - area_bonus - shape_bonus - opening_bonus
            if best_score is None or score < best_score:
                best_score = score
                best = candidate_center

        if best is None:
            return None, 0.0
        return best, 1.0 / (1.0 + max(float(best_score), 0.0))

    def _board_corners_from_homography(self, H, plane_size=None):
        plane_width, plane_height = self._resolve_plane_size(plane_size)
        return np.asarray(
            [
                HomographyEstimator.transform_point((0, 0), H),
                HomographyEstimator.transform_point((plane_width, 0), H),
                HomographyEstimator.transform_point((plane_width, plane_height), H),
                HomographyEstimator.transform_point((0, plane_height), H),
            ],
            dtype=np.float32,
        )

    def _validate_projected_white_board(self, gray, H, plane_size):
        corners = self._board_corners_from_homography(H, plane_size)
        if not self._validate_tracked_corners(corners, gray.shape):
            return {"ok": False, "score": 0.0, "reason": "projected_board_out_of_frame"}

        polygon = np.round(corners).astype(np.int32)
        mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)

        board_area = int(np.count_nonzero(mask))
        image_area = gray.shape[0] * gray.shape[1]
        area_ratio = board_area / max(float(image_area), 1.0)
        if area_ratio < 0.06 or area_ratio > 0.88:
            return {"ok": False, "score": 0.0, "reason": "projected_board_area_invalid"}

        kernel_size = int(np.clip(np.sqrt(board_area) * 0.035, 5, 23))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        inner_mask = cv2.erode(mask, kernel, iterations=1)
        if int(np.count_nonzero(inner_mask)) < max(120, board_area * 0.20):
            inner_mask = mask

        board_pixels = gray[inner_mask > 0]
        if board_pixels.size < 120:
            return {"ok": False, "score": 0.0, "reason": "projected_board_too_small"}

        threshold = float(np.clip(np.percentile(gray, 66), 120, 210))
        white_ratio = float(np.count_nonzero(board_pixels >= threshold)) / float(board_pixels.size)
        mean_inside = float(np.mean(board_pixels))

        dilate_size = max(kernel_size, 9)
        outer_kernel = np.ones((dilate_size, dilate_size), dtype=np.uint8)
        outer_mask = cv2.dilate(mask, outer_kernel, iterations=1)
        ring_mask = cv2.subtract(outer_mask, mask)
        ring_pixels = gray[ring_mask > 0]
        edge_contrast = mean_inside - float(np.mean(ring_pixels)) if ring_pixels.size else 0.0

        white_score = np.clip((white_ratio - 0.25) / 0.55, 0.0, 1.0)
        brightness_score = np.clip((mean_inside - 105.0) / 95.0, 0.0, 1.0)
        edge_score = np.clip((edge_contrast + 8.0) / 48.0, 0.0, 1.0)
        score = float(white_score * 0.52 + brightness_score * 0.36 + edge_score * 0.12)
        ok = score >= 0.48 and white_ratio >= 0.30 and mean_inside >= 108.0

        reason = None if ok else "white_board_validation_failed"
        return {
            "ok": ok,
            "score": score,
            "reason": reason,
            "white_ratio": white_ratio,
            "mean_inside": mean_inside,
            "edge_contrast": edge_contrast,
            "area_ratio": area_ratio,
        }

    def _predicted_marker_points(self):
        if self.last_homography is None:
            if self.last_marker_centers is None:
                return None
            last = np.asarray(self.last_marker_centers, dtype=np.float32)
            return last if last.shape == (4, 2) else None

        return self._marker_points_from_homography(self.last_homography, self.current_plane_size)

    def _marker_points_from_homography(self, H, plane_size=None):
        projected = [
            HomographyEstimator.transform_point(tuple(point), H)
            for point in self._marker_board_points(plane_size)
        ]
        if any(point is None for point in projected):
            return None
        return np.asarray(projected, dtype=np.float32)

    def _build_marker_result(self, marker_centers, observed, image_shape, detector_score):
        marker_centers = np.asarray(marker_centers, dtype=np.float32)
        observed = np.asarray(observed, dtype=bool)
        if marker_centers.shape != (4, 2) or observed.shape != (4,):
            return None
        if int(np.count_nonzero(observed)) < 3:
            return None

        plane_size = self.current_plane_size if self.is_registered else self._plane_size_for_image_quad(marker_centers)
        marker_board_points = self._marker_board_points(plane_size)
        H = HomographyEstimator.compute_homography(marker_board_points, marker_centers)
        board_corners = self._board_corners_from_homography(H, plane_size)
        if not self._validate_tracked_corners(board_corners, image_shape):
            return None

        confidence, reprojection_error = self._estimate_homography_confidence(
            H,
            marker_centers,
            observed,
            image_shape,
            detector_score,
            plane_size,
        )
        if confidence < self.min_homography_confidence:
            return None

        return {
            "H": H,
            "corners": board_corners,
            "marker_centers": marker_centers,
            "observed_marker_centers": marker_centers.copy(),
            "marker_observed": observed,
            "score": confidence,
            "homography_confidence": confidence,
            "reprojection_error": reprojection_error,
            "matched_points": int(np.count_nonzero(observed)),
            "plane_size": plane_size,
        }

    def _accept_marker_result(self, marker_result, gray, method):
        H = marker_result["H"]
        plane_size = tuple(marker_result.get("plane_size", self.current_plane_size))
        corners = self._stabilize_corners(
            marker_result["corners"],
            confidence=marker_result.get("homography_confidence", marker_result.get("score", 1.0)),
        )
        if corners is not marker_result["corners"]:
            H = HomographyEstimator.compute_homography(self._board_points_for_size(plane_size), corners)

        confidence = marker_result.get("homography_confidence", marker_result.get("score", 1.0))
        if self.is_registered:
            H = self._smooth_homography(H, confidence)
            corners = self._board_corners_from_homography(H, plane_size)
        else:
            self.smoothed_homography = H

        marker_centers = self._marker_points_from_homography(H, plane_size)
        if marker_centers is None:
            marker_centers = marker_result["marker_centers"].astype(np.float32)
        observed_marker_centers = marker_result.get("observed_marker_centers")
        if observed_marker_centers is not None:
            observed_marker_centers = np.asarray(observed_marker_centers, dtype=np.float32)

        self.current_plane_size = plane_size
        self.last_corners = corners
        self.last_homography = H
        self.last_gray = gray
        self.last_marker_centers = marker_centers
        self.last_marker_observed = marker_result.get("marker_observed")
        if marker_result.get("door_image_points") is not None:
            self.last_door_image_points = np.asarray(marker_result["door_image_points"], dtype=np.float32)
        if marker_result.get("door_world_points") is not None:
            self.last_door_world_points = np.asarray(marker_result["door_world_points"], dtype=np.float32)
        self.active_detector_mode = marker_result.get("detector_mode", self.active_detector_mode or self.detector_mode)
        self.last_homography_confidence = confidence
        self.missed_frames = 0

        return {
            "success": True,
            "H": H,
            "matched_points": marker_result.get("matched_points", 4),
            "corners": corners,
            "marker_centers": marker_centers,
            "observed_marker_centers": observed_marker_centers,
            "marker_observed": marker_result.get("marker_observed"),
            "stale": False,
            "tracking_method": method,
            "track_score": marker_result.get("score", confidence),
            "homography_confidence": confidence,
            "reprojection_error": marker_result.get("reprojection_error"),
            "reject_reason": marker_result.get("reject_reason"),
            "candidate_count": marker_result.get("candidate_count", self.last_candidate_count),
            "selected_marker_centers": marker_result.get("selected_marker_centers"),
            "white_validation_score": marker_result.get("white_validation_score"),
            "marker_quad_score": marker_result.get("marker_quad_score"),
            "detector_mode": marker_result.get("detector_mode", self.active_detector_mode or self.detector_mode),
            "door_symbol_score": marker_result.get("door_symbol_score"),
            "door_direction_score": marker_result.get("door_direction_score"),
            "door_image_points": marker_result.get("door_image_points"),
            "door_world_points": marker_result.get("door_world_points"),
            "plane_size": plane_size,
        }

    def _smooth_homography(self, H, confidence):
        H = self._normalize_homography(H)
        if self.smoothed_homography is None:
            self.smoothed_homography = H
            return H

        alpha = self.homography_ema_alpha * float(np.clip(confidence, 0.25, 1.0))
        previous = self._normalize_homography(self.smoothed_homography)
        smoothed = previous * (1.0 - alpha) + H * alpha
        self.smoothed_homography = self._normalize_homography(smoothed)
        return self.smoothed_homography

    def _normalize_homography(self, H):
        H = np.asarray(H, dtype=np.float32)
        if abs(float(H[2, 2])) > 1e-8:
            H = H / H[2, 2]
        return H.astype(np.float32)

    def _estimate_homography_confidence(
        self,
        H,
        marker_centers,
        observed,
        image_shape,
        detector_score,
        plane_size=None,
    ):
        projected = np.asarray(
            [HomographyEstimator.transform_point(tuple(point), H) for point in self._marker_board_points(plane_size)],
            dtype=np.float32,
        )
        observed_count = max(int(np.count_nonzero(observed)), 1)
        residual = np.linalg.norm(projected[observed] - marker_centers[observed], axis=1)
        reprojection_error = float(np.mean(residual)) if residual.size else 999.0
        residual_score = 1.0 / (1.0 + reprojection_error) if residual.size else 0.0
        observed_score = observed_count / 4.0

        temporal_score = 1.0
        if self.last_marker_centers is not None:
            shift = np.linalg.norm(marker_centers[observed] - self.last_marker_centers[observed], axis=1)
            temporal_score = 1.0 / (1.0 + float(np.mean(shift)) / max(self.marker_roi_radius, 1.0)) if shift.size else 0.4

        area = self._polygon_area(self._board_corners_from_homography(H, plane_size))
        image_area = max(float(image_shape[0] * image_shape[1]), 1.0)
        area_ratio = area / image_area
        area_score = np.clip((area_ratio - 0.03) / 0.20, 0.0, 1.0)

        confidence = (
            np.clip(detector_score, 0.0, 1.0) * 0.35
            + residual_score * 0.25
            + observed_score * 0.25
            + temporal_score * 0.10
            + float(area_score) * 0.05
        )
        return float(np.clip(confidence, 0.0, 1.0)), reprojection_error

    def _marker_anchors(self, width, height):
        if self.last_marker_centers is not None:
            last = np.asarray(self.last_marker_centers, dtype=np.float32)
            if last.shape == (4, 2):
                return {
                    "tl": last[0],
                    "tr": last[1],
                    "br": last[2],
                    "bl": last[3],
                }

        return {
            "tl": np.array([width * 0.18, height * 0.18], dtype=np.float32),
            "tr": np.array([width * 0.82, height * 0.18], dtype=np.float32),
            "br": np.array([width * 0.82, height * 0.82], dtype=np.float32),
            "bl": np.array([width * 0.18, height * 0.82], dtype=np.float32),
        }

    def _dark_mark_mask(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = min(140, int(np.percentile(blurred, 30)))
        global_mask = (blurred <= threshold).astype(np.uint8)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )
        mask = np.maximum(global_mask, (adaptive > 0).astype(np.uint8))
        mask = self._morph(mask, mode="dilate", iterations=2)
        mask = self._morph(mask, mode="erode", iterations=1)
        return mask

    def _pruned_debug_marker_candidates(self):
        if not self.debug_enabled:
            self.debug_marker_display_cache = []
            return []

        accepted = [candidate for candidate in self.debug_marker_candidates if candidate.get("accepted")]
        rejected = [candidate for candidate in self.debug_marker_candidates if not candidate.get("accepted")]
        pruned = self._non_max_candidates(accepted, radius=26.0, limit_per_slot=3)
        pruned.extend(self._non_max_candidates(rejected, radius=24.0, limit_per_slot=1))
        pruned = pruned[:16]
        return self._smooth_debug_marker_candidates(pruned)

    def _smooth_debug_marker_candidates(self, candidates):
        smoothed = []
        used_previous = set()

        for candidate in candidates:
            point = np.asarray(candidate.get("point"), dtype=np.float32)
            if point.shape != (2,):
                continue
            best_index = None
            best_distance = None
            for index, previous in enumerate(self.debug_marker_display_cache):
                if index in used_previous:
                    continue
                if previous.get("slot") != candidate.get("slot"):
                    continue
                previous_point = np.asarray(previous.get("point"), dtype=np.float32)
                if previous_point.shape != (2,):
                    continue
                distance = float(np.linalg.norm(point - previous_point))
                if distance > 30.0:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index

            merged = dict(candidate)
            if best_index is not None:
                previous = self.debug_marker_display_cache[best_index]
                previous_point = np.asarray(previous.get("point"), dtype=np.float32)
                merged["point"] = previous_point * 0.65 + point * 0.35
                merged["age"] = 0
                used_previous.add(best_index)
            else:
                merged["point"] = point
                merged["age"] = 0
            smoothed.append(merged)

        for index, previous in enumerate(self.debug_marker_display_cache):
            if index in used_previous:
                continue
            age = int(previous.get("age", 0)) + 1
            if age <= 3:
                faded = dict(previous)
                faded["age"] = age
                faded["stale_debug"] = True
                smoothed.append(faded)

        self.debug_marker_display_cache = smoothed[:18]
        return self.debug_marker_display_cache

    def _non_max_candidates(self, candidates, radius, limit_per_slot):
        result = []
        counts = {"tl": 0, "tr": 0, "br": 0, "bl": 0}
        ordered = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for candidate in ordered:
            slot = candidate.get("slot")
            if slot in counts and counts[slot] >= limit_per_slot:
                continue
            point = np.asarray(candidate.get("point"), dtype=np.float32)
            if point.shape != (2,):
                continue
            too_close = False
            for existing in result:
                existing_point = np.asarray(existing.get("point"), dtype=np.float32)
                if existing_point.shape == (2,) and np.linalg.norm(point - existing_point) < radius:
                    too_close = True
                    break
            if too_close:
                continue
            result.append(candidate)
            if slot in counts:
                counts[slot] += 1
        return result

    def _component_to_mark_candidate(self, pixels, image_area, frame_width, frame_height, apply_border_margin=True):
        x_min = float(np.min(pixels[:, 0]))
        x_max = float(np.max(pixels[:, 0]))
        y_min = float(np.min(pixels[:, 1]))
        y_max = float(np.max(pixels[:, 1]))
        box_width = x_max - x_min + 1.0
        box_height = y_max - y_min + 1.0
        area = float(len(pixels))

        if area < max(6.0, image_area * 0.00001):
            return None
        if area > image_area * 0.025:
            return None
        if box_width < 3.0 or box_height < 3.0:
            return None

        aspect = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
        if aspect > 14.0:
            return None

        fill_ratio = area / max(box_width * box_height, 1.0)
        if fill_ratio < 0.035:
            return None

        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
        if apply_border_margin:
            margin_x = frame_width * 0.04
            margin_y = frame_height * 0.04
            if center[0] < margin_x or center[0] > frame_width - margin_x:
                return None
            if center[1] < margin_y or center[1] > frame_height - margin_y:
                return None

        return {
            "center": center,
            "blob_center": center.copy(),
            "area": area,
            "box": (x_min, y_min, x_max, y_max),
            "fill_ratio": fill_ratio,
        }

    def _estimate_l_mark_corner(self, pixels, slot_name):
        if len(pixels) < 4:
            return None

        int_pixels = np.asarray(np.round(pixels), dtype=np.int32)
        x_min = int(np.min(int_pixels[:, 0]))
        y_min = int(np.min(int_pixels[:, 1]))
        x_max = int(np.max(int_pixels[:, 0]))
        y_max = int(np.max(int_pixels[:, 1]))
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        if width < 3 or height < 3:
            return None

        local = np.zeros((height, width), dtype=np.uint8)
        local[int_pixels[:, 1] - y_min, int_pixels[:, 0] - x_min] = 255
        corners = cv2.goodFeaturesToTrack(
            local,
            maxCorners=8,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=5,
        )

        candidates = []
        if corners is not None:
            for corner in corners.reshape(-1, 2):
                candidates.append(corner + np.array([x_min, y_min], dtype=np.float32))

        x = pixels[:, 0]
        y = pixels[:, 1]
        if slot_name == "tl":
            candidates.append(pixels[int(np.argmin(x + y))])
            key = lambda point: point[0] + point[1]
            return np.asarray(min(candidates, key=key), dtype=np.float32)
        if slot_name == "tr":
            candidates.append(pixels[int(np.argmax(x - y))])
            key = lambda point: point[0] - point[1]
            return np.asarray(max(candidates, key=key), dtype=np.float32)
        if slot_name == "br":
            candidates.append(pixels[int(np.argmax(x + y))])
            key = lambda point: point[0] + point[1]
            return np.asarray(max(candidates, key=key), dtype=np.float32)

        candidates.append(pixels[int(np.argmin(x - y))])
        key = lambda point: point[0] - point[1]
        return np.asarray(min(candidates, key=key), dtype=np.float32)

    def _analyze_l_mark_shape(self, pixels, corner, slot_name):
        if len(pixels) < 6:
            return None

        directions = {
            "tl": (1.0, 1.0),
            "tr": (-1.0, 1.0),
            "br": (-1.0, -1.0),
            "bl": (1.0, -1.0),
        }
        if slot_name not in directions:
            return None

        dx, dy = directions[slot_name]
        points = np.asarray(pixels, dtype=np.float32)
        corner = np.asarray(corner, dtype=np.float32)
        rel = points - corner
        distances = np.linalg.norm(rel, axis=1)
        valid = distances >= 2.0
        if int(np.count_nonzero(valid)) < 6:
            return None

        directions = self._dominant_ray_directions(rel[valid], distances[valid])
        if directions is None:
            return None

        dir_a, dir_b, arm_angle = directions
        if arm_angle < 32.0 or arm_angle > 150.0:
            return None

        expected_open = np.asarray([dx, dy], dtype=np.float32)
        expected_open /= max(float(np.linalg.norm(expected_open)), 1e-6)
        open_vector = dir_a + dir_b
        norm = float(np.linalg.norm(open_vector))
        if norm <= 1e-6:
            return None
        open_vector = open_vector / norm
        if float(np.dot(open_vector, expected_open)) < -0.04:
            return None

        arm_a = self._score_l_arm(rel, dir_a)
        arm_b = self._score_l_arm(rel, dir_b)
        if arm_a is None or arm_b is None:
            return None

        mask_a, coverage_a, length_a, count_a = arm_a
        mask_b, coverage_b, length_b, count_b = arm_b
        if length_a < 5.0 or length_b < 5.0:
            return None
        if coverage_a < 0.27 or coverage_b < 0.27:
            return None

        arm_mask = mask_a | mask_b
        arm_ratio = float(np.count_nonzero(arm_mask)) / max(float(len(points)), 1.0)
        if arm_ratio < 0.24:
            return None

        balance = min(count_a, count_b) / max(count_a, count_b, 1)
        if balance < 0.12:
            return None

        spill_ratio = 1.0 - arm_ratio
        if spill_ratio > 0.68:
            return None

        if self._reject_line_like_l_candidate(points, rel, dir_a, dir_b, arm_angle, arm_mask):
            return None

        corner_radius = max(5.0, min(length_a, length_b) * 0.22)
        corner_near = distances <= corner_radius
        corner_score = min(1.0, float(np.count_nonzero(corner_near)) / max(4.0, len(points) * 0.10))
        angle_score = 1.0 - min(abs(arm_angle - 90.0) / 70.0, 1.0)

        score = (
            coverage_a * 0.24
            + coverage_b * 0.24
            + arm_ratio * 0.22
            + balance * 0.14
            + corner_score * 0.06
            + angle_score * 0.10
            - spill_ratio * 0.18
        )
        if score < 0.16:
            return None
        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "open_vector": open_vector.astype(np.float32),
            "arm_angle": float(arm_angle),
        }

    def _dominant_ray_directions(self, vectors, distances):
        if len(vectors) < 6:
            return None

        angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi * 2.0)
        bins = 32
        hist = np.zeros(bins, dtype=np.float32)
        weights = np.clip(distances, 1.0, np.percentile(distances, 90))
        indices = np.floor(angles / (np.pi * 2.0) * bins).astype(np.int32) % bins
        np.add.at(hist, indices, weights.astype(np.float32))
        hist = np.convolve(np.r_[hist[-1], hist, hist[0]], np.asarray([0.25, 0.5, 0.25], dtype=np.float32), mode="valid")

        ranked = np.argsort(hist)[::-1]
        for first in ranked:
            if hist[first] <= 0.0:
                break
            angle_a = (first + 0.5) / bins * np.pi * 2.0
            dir_a = self._refine_ray_direction(vectors, distances, angle_a)
            if dir_a is None:
                continue

            for second in ranked:
                if second == first or hist[second] <= 0.0:
                    continue
                angle_b = (second + 0.5) / bins * np.pi * 2.0
                sep = self._ray_angle_degrees(angle_a, angle_b)
                if sep < 32.0 or sep > 150.0:
                    continue
                dir_b = self._refine_ray_direction(vectors, distances, angle_b)
                if dir_b is None:
                    continue
                arm_angle = self._ray_angle_degrees(
                    np.arctan2(dir_a[1], dir_a[0]),
                    np.arctan2(dir_b[1], dir_b[0]),
                )
                if arm_angle < 32.0 or arm_angle > 150.0:
                    continue
                return dir_a, dir_b, arm_angle
        return None

    def _refine_ray_direction(self, vectors, distances, angle):
        unit = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        vector_angles = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), np.pi * 2.0)
        diff = np.asarray([self._ray_angle_degrees(value, angle) for value in vector_angles], dtype=np.float32)
        selected = diff <= 34.0
        if int(np.count_nonzero(selected)) < 3:
            return None
        selected_vectors = vectors[selected]
        selected_distances = distances[selected]
        normalized = selected_vectors / np.maximum(selected_distances[:, None], 1e-6)
        refined = np.sum(normalized * selected_distances[:, None], axis=0)
        norm = float(np.linalg.norm(refined))
        if norm <= 1e-6:
            return unit
        refined = refined / norm
        if float(np.dot(refined, unit)) < 0.0:
            refined = -refined
        return refined.astype(np.float32)

    def _score_l_arm(self, rel, direction):
        projection = rel @ direction
        perpendicular = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
        positive = projection > -3.0
        if int(np.count_nonzero(positive)) < 3:
            return None

        length = float(np.percentile(projection[positive], 94))
        if length <= 3.0:
            return None

        tolerance = max(4.5, length * 0.26)
        mask = positive & (projection <= length + tolerance) & (perpendicular <= tolerance)
        count = int(np.count_nonzero(mask))
        if count < 3:
            return None

        coverage = self._axis_coverage(projection[mask], length, bins=5)
        return mask, coverage, length, count

    def _reject_line_like_l_candidate(self, points, rel, dir_a, dir_b, arm_angle, arm_mask):
        if len(points) < 10:
            return False

        direction_dot = float(np.dot(dir_a, dir_b))
        if direction_dot < -0.88:
            return True

        selected = np.asarray(rel[arm_mask], dtype=np.float32)
        if selected.shape[0] < 8:
            selected = np.asarray(rel, dtype=np.float32)
        selected = selected[np.linalg.norm(selected, axis=1) >= 2.0]
        if selected.shape[0] < 8:
            return False

        centered = selected - np.mean(selected, axis=0)
        covariance = centered.T @ centered / max(float(centered.shape[0]), 1.0)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        total_variance = float(np.sum(eigenvalues))
        if total_variance <= 1e-6:
            return False

        linearity = float(eigenvalues[-1] / total_variance)
        line_direction = eigenvectors[:, -1].astype(np.float32)
        line_norm = float(np.linalg.norm(line_direction))
        if line_norm <= 1e-6:
            return False
        line_direction /= line_norm

        scale = max(float(np.percentile(np.linalg.norm(centered, axis=1), 90)), 1.0)
        line_perpendicular = np.abs(centered[:, 0] * line_direction[1] - centered[:, 1] * line_direction[0])
        single_line_error = float(np.mean(line_perpendicular)) / scale

        ray_perpendicular_a = np.abs(selected[:, 0] * dir_a[1] - selected[:, 1] * dir_a[0])
        ray_perpendicular_b = np.abs(selected[:, 0] * dir_b[1] - selected[:, 1] * dir_b[0])
        two_ray_error = float(np.mean(np.minimum(ray_perpendicular_a, ray_perpendicular_b))) / scale

        if linearity > 0.986 and single_line_error < 0.038:
            return True
        if arm_angle > 132.0 and linearity > 0.974 and single_line_error < 0.055:
            return two_ray_error > single_line_error * 0.72
        return False

    def _ray_angle_degrees(self, angle_a, angle_b):
        diff = abs(float(angle_a - angle_b)) % (np.pi * 2.0)
        if diff > np.pi:
            diff = np.pi * 2.0 - diff
        return float(np.degrees(diff))

    def _axis_coverage(self, values, length, bins=5):
        if length <= 1e-6 or values.size == 0:
            return 0.0
        normalized = np.clip(values / length, 0.0, 0.999)
        occupied = np.zeros(bins, dtype=bool)
        occupied[(normalized * bins).astype(np.int32)] = True
        return float(np.count_nonzero(occupied)) / float(bins)

    def _quadrant_corner_slot(self, center, frame_center):
        left = center[0] < frame_center[0]
        top = center[1] < frame_center[1]
        if left and top:
            return "tl"
        if (not left) and top:
            return "tr"
        if (not left) and (not top):
            return "br"
        return "bl"

    def _candidate_corner_slot(self, center, frame_center):
        if self.last_marker_centers is not None:
            last = np.asarray(self.last_marker_centers, dtype=np.float32)
            if last.shape == (4, 2):
                labels = ("tl", "tr", "br", "bl")
                distances = np.linalg.norm(last - center, axis=1)
                best_index = int(np.argmin(distances))
                if distances[best_index] <= self.marker_lock_radius:
                    return labels[best_index]
                return None

        left = center[0] < frame_center[0]
        top = center[1] < frame_center[1]
        if left and top:
            return "tl"
        if (not left) and top:
            return "tr"
        if (not left) and (not top):
            return "br"
        return "bl"

    def _select_best_mark_candidate(self, candidates, anchor):
        return min(candidates, key=lambda candidate: self._mark_candidate_score(candidate, anchor))

    def _select_marker_quad(
        self,
        slots,
        predicted,
        allow_partial,
        frame_center,
        image_area,
        allow_inferred_missing=False,
    ):
        labels = ("tl", "tr", "br", "bl")
        anchor_is_reliable = self.is_registered or self.last_marker_centers is not None or predicted is not None
        option_limit = 4 if anchor_is_reliable else 6
        option_groups = []
        for index, slot_name in enumerate(labels):
            slot = slots[slot_name]
            options = sorted(
                slot["candidates"],
                key=lambda candidate: self._candidate_option_score(
                    candidate,
                    slot["anchor"],
                    anchor_is_reliable,
                ),
            )[:option_limit]
            if not options and predicted is not None:
                options = [
                    {
                        "center": predicted[index],
                        "area": 1.0,
                        "fill_ratio": 0.35,
                        "predicted": True,
                    }
                ]
            if not options and allow_partial and allow_inferred_missing:
                options = [
                    {
                        "center": np.array([np.nan, np.nan], dtype=np.float32),
                        "area": 1.0,
                        "fill_ratio": 0.35,
                        "predicted": True,
                        "inferred": True,
                    }
                ]
            if not options:
                return None
            option_groups.append(options)

        best = None
        best_score = None
        for combination in itertools.product(*option_groups):
            combination = [dict(item) for item in combination]
            inferred_count = sum(1 for item in combination if item.get("inferred", False))
            if inferred_count > 1:
                continue
            if inferred_count == 1:
                inferred_points = [
                    None if item.get("inferred", False) else np.asarray(item["center"], dtype=np.float32)
                    for item in combination
                ]
                inferred_center = self._infer_missing_marker_center(inferred_points)
                if inferred_center is None:
                    continue
                for item in combination:
                    if item.get("inferred", False):
                        item["center"] = inferred_center
                        break

            observed = np.asarray([not item.get("predicted", False) for item in combination], dtype=bool)
            if int(np.count_nonzero(observed)) < (3 if allow_partial else 4):
                continue

            points = np.asarray([item["center"] for item in combination], dtype=np.float32)
            if not np.isfinite(points).all():
                continue
            if self._has_duplicate_points(points, min_distance=22.0):
                continue
            if self._order_quad_points(points) is None:
                continue

            quad_score = self._score_marker_quad(points, combination, frame_center, image_area)
            if quad_score is None:
                continue

            anchor_score = 0.0
            shape_score = 0.0
            for item, slot_name in zip(combination, labels):
                if item.get("predicted", False):
                    continue
                if anchor_is_reliable:
                    anchor_score += self._mark_candidate_score(item, slots[slot_name]["anchor"]) / 180.0
                shape_score += 1.0 - float(item.get("shape_score", 0.0))
            anchor_weight = 0.08 if anchor_is_reliable else 0.0
            total_score = quad_score + anchor_score * anchor_weight + shape_score * 0.20
            if best_score is None or total_score < best_score:
                best_score = total_score
                best = (points, list(combination), observed, total_score)

        return best

    def _infer_missing_marker_center(self, points):
        if len(points) != 4:
            return None

        missing = [index for index, point in enumerate(points) if point is None]
        if len(missing) != 1:
            return None

        index = missing[0]
        try:
            tl, tr, br, bl = points
            if index == 0:
                inferred = tr + bl - br
            elif index == 1:
                inferred = tl + br - bl
            elif index == 2:
                inferred = tr + bl - tl
            else:
                inferred = tl + br - tr
        except TypeError:
            return None

        inferred = np.asarray(inferred, dtype=np.float32)
        if inferred.shape != (2,) or not np.isfinite(inferred).all():
            return None
        return inferred

    def _mark_candidate_score(self, candidate, anchor):
        distance = np.linalg.norm(candidate["center"] - anchor)
        area_bonus = min(candidate["area"], 1200.0) * 0.02
        fill_penalty = abs(candidate["fill_ratio"] - 0.35) * 80.0
        shape_bonus = candidate.get("shape_score", 0.0) * 54.0
        return distance + fill_penalty - area_bonus - shape_bonus

    def _candidate_option_score(self, candidate, anchor, anchor_is_reliable):
        if anchor_is_reliable:
            return self._mark_candidate_score(candidate, anchor)

        shape_score = float(candidate.get("shape_score", 0.0))
        fill_penalty = abs(float(candidate.get("fill_ratio", 0.35)) - 0.35)
        area_bonus = min(float(candidate.get("area", 0.0)), 1200.0) / 1200.0
        return (1.0 - shape_score) * 1.8 + fill_penalty * 0.55 - area_bonus * 0.08

    def _has_duplicate_points(self, points, min_distance):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < min_distance:
                    return True
        return False

    def _order_quad_points(self, points):
        if points.shape != (4, 2):
            return None

        sums = points[:, 0] + points[:, 1]
        diffs = points[:, 0] - points[:, 1]
        ordered = np.asarray(
            [
                points[int(np.argmin(sums))],
                points[int(np.argmax(diffs))],
                points[int(np.argmax(sums))],
                points[int(np.argmin(diffs))],
            ],
            dtype=np.float32,
        )

        if len({tuple(point) for point in ordered}) != 4:
            return None
        if self._polygon_area(ordered) < 500.0 or not self._is_convex_quad(ordered):
            return None
        return ordered

    def _score_marker_quad(self, ordered, group, frame_center, image_area):
        area = self._polygon_area(ordered)
        area_ratio = area / max(float(image_area), 1.0)
        if area_ratio < 0.08 or area_ratio > 0.82:
            return None

        edge_lengths = [
            np.linalg.norm(ordered[(idx + 1) % 4] - ordered[idx])
            for idx in range(4)
        ]
        if min(edge_lengths) < 45.0:
            return None

        top_width = edge_lengths[0]
        right_height = edge_lengths[1]
        bottom_width = edge_lengths[2]
        left_height = edge_lengths[3]
        quad_width = max((top_width + bottom_width) * 0.5, 1.0)
        quad_height = max((left_height + right_height) * 0.5, 1.0)
        observed_ratio = max(quad_width, quad_height) / max(min(quad_width, quad_height), 1.0)
        expected_ratio = (self.A4_HEIGHT - self.MARK_INSET * 2) / (self.A4_WIDTH - self.MARK_INSET * 2)
        ratio_error = abs(observed_ratio - expected_ratio) / expected_ratio
        if ratio_error > 0.38:
            return None

        if self.last_marker_centers is not None:
            marker_shift = np.linalg.norm(ordered - self.last_marker_centers, axis=1).mean()
            max_shift = max(self.marker_lock_radius, np.sqrt(float(image_area)) * 0.22)
            if marker_shift > max_shift:
                return None
        else:
            center = self._quad_center(ordered)
            offsets = ordered - center
            if not (
                offsets[0, 0] < 0 and offsets[0, 1] < 0
                and offsets[1, 0] > 0 and offsets[1, 1] < 0
                and offsets[2, 0] > 0 and offsets[2, 1] > 0
                and offsets[3, 0] < 0 and offsets[3, 1] > 0
            ):
                return None

        center = self._quad_center(ordered)
        opening_scores = []
        for index, (item, point) in enumerate(zip(group, ordered)):
            open_vector = item.get("open_vector")
            if open_vector is None:
                continue
            open_vector = np.asarray(open_vector, dtype=np.float32)
            score = self._opening_wedge_score(
                open_vector,
                ordered[(index - 1) % 4] - point,
                ordered[(index + 1) % 4] - point,
            )
            if score is None:
                continue
            opening_scores.append(score)
        opening_penalty = 0.0
        if opening_scores:
            mean_opening_score = float(np.mean(opening_scores))
            if mean_opening_score < 0.14:
                return None
            opening_penalty = 1.0 - mean_opening_score

        center_score = np.linalg.norm(center - frame_center) / max(np.sqrt(image_area), 1.0)
        center_weight = 0.12 if self.last_marker_centers is None else 0.22
        area_score = 1.0 - min(area_ratio / 0.35, 1.0)
        observed_items = [item for item in group if not item.get("predicted", False)]
        size_values = np.asarray([item["area"] for item in observed_items], dtype=np.float32)
        size_balance = float(np.std(size_values) / max(np.mean(size_values), 1.0))
        shape_values = [item.get("shape_score") for item in observed_items if item.get("shape_score") is not None]
        shape_penalty = 0.0
        if shape_values:
            shape_penalty = 1.0 - float(np.mean(shape_values))
        return (
            ratio_error * 2.0
            + center_score * center_weight
            + area_score * 0.5
            + size_balance * 0.25
            + shape_penalty * 0.55
            + opening_penalty * 0.65
        )

    def _quad_center(self, ordered):
        diagonal_center = self._intersect_segments(ordered[0], ordered[2], ordered[1], ordered[3])
        if diagonal_center is not None:
            return diagonal_center.astype(np.float32)
        return ordered.mean(axis=0).astype(np.float32)

    def _score_marker_opening_against_quad(self, open_vector, point, expected_quad, slot_name):
        if expected_quad is None:
            return None

        labels = ("tl", "tr", "br", "bl")
        if slot_name not in labels:
            return None
        index = labels.index(slot_name)

        expected_quad = np.asarray(expected_quad, dtype=np.float32)
        point = np.asarray(point, dtype=np.float32)
        if expected_quad.shape != (4, 2) or point.shape != (2,):
            return None
        if not np.isfinite(expected_quad).all() or not np.isfinite(point).all():
            return None

        edge_a = expected_quad[(index - 1) % 4] - point
        edge_b = expected_quad[(index + 1) % 4] - point
        score = self._opening_wedge_score(open_vector, edge_a, edge_b)
        if score is None or score < 0.16:
            return None
        return score

    def _opening_wedge_score(self, open_vector, edge_a, edge_b):
        open_norm = float(np.linalg.norm(open_vector))
        norm_a = float(np.linalg.norm(edge_a))
        norm_b = float(np.linalg.norm(edge_b))
        if open_norm <= 1e-6 or norm_a <= 1e-6 or norm_b <= 1e-6:
            return None

        open_unit = np.asarray(open_vector, dtype=np.float32) / open_norm
        edge_a_unit = np.asarray(edge_a, dtype=np.float32) / norm_a
        edge_b_unit = np.asarray(edge_b, dtype=np.float32) / norm_b

        dot_a = float(np.dot(open_unit, edge_a_unit))
        dot_b = float(np.dot(open_unit, edge_b_unit))
        min_dot = min(dot_a, dot_b)
        mean_dot = (dot_a + dot_b) * 0.5

        if min_dot < -0.06:
            return None
        return float(np.clip((min_dot + 0.06) * 0.95 + mean_dot * 0.35, 0.0, 1.0))

    def _intersect_segments(self, a, b, c, d):
        line_ab = self._line_from_points(a, b)
        line_cd = self._line_from_points(c, d)
        if line_ab is None or line_cd is None:
            return None
        return self._intersect_lines(line_ab, line_cd)

    def _line_from_points(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if np.linalg.norm(b - a) <= 1e-6:
            return None
        p1 = np.array([a[0], a[1], 1.0], dtype=np.float64)
        p2 = np.array([b[0], b[1], 1.0], dtype=np.float64)
        line = np.cross(p1, p2)
        norm = np.linalg.norm(line[:2])
        if norm <= 1e-9:
            return None
        return line / norm

    def _marker_board_points(self, plane_size=None):
        plane_width, plane_height = self._resolve_plane_size(plane_size)
        inset = self.MARK_INSET
        return np.asarray(
            [
                [inset, inset],
                [plane_width - inset, inset],
                [plane_width - inset, plane_height - inset],
                [inset, plane_height - inset],
            ],
            dtype=np.float32,
        )

    def _stabilize_corners(self, corners, confidence=1.0):
        corners = corners.astype(np.float32)
        if self.last_corners is None:
            return corners

        displacement = np.linalg.norm(corners - self.last_corners, axis=1).mean()
        if displacement > 90.0 and confidence >= 0.72:
            return corners
        if displacement > 140.0:
            return corners

        confidence = float(np.clip(confidence, 0.0, 1.0))
        alpha = np.clip(self.corner_smoothing * (0.45 + confidence), 0.12, 0.70)
        return (self.last_corners * (1.0 - alpha) + corners * alpha).astype(np.float32)

    def _track_last_corners(self, gray):
        if self.last_gray is None or self.last_corners is None:
            return None, 0.0

        tracked = []
        valid_scores = []
        displacements = []

        for corner in self.last_corners:
            next_corner, score = self._track_one_corner(self.last_gray, gray, corner)
            if next_corner is None or score < self.min_track_score:
                tracked.append(None)
                continue

            tracked.append(next_corner)
            valid_scores.append(score)
            displacements.append(next_corner - corner)

        valid_count = len(valid_scores)
        if valid_count < 2:
            return None, 0.0

        if valid_count < 4:
            median_displacement = np.median(np.asarray(displacements, dtype=np.float32), axis=0)
            for idx, point in enumerate(tracked):
                if point is None:
                    tracked[idx] = self.last_corners[idx] + median_displacement

        corners = np.asarray(tracked, dtype=np.float32)
        if not self._validate_tracked_corners(corners, gray.shape):
            return None, 0.0

        smoothed = (self.last_corners * 0.25 + corners * 0.75).astype(np.float32)
        return smoothed, float(np.mean(valid_scores))

    def _track_one_corner(self, previous_gray, current_gray, corner):
        patch = self._extract_patch(previous_gray, corner, self.track_patch_radius)
        if patch is None:
            return None, 0.0

        search_radius = self.track_search_radius
        x = int(round(corner[0]))
        y = int(round(corner[1]))
        best_score = -1.0
        best_point = None

        y_min = max(y - search_radius, self.track_patch_radius)
        y_max = min(y + search_radius, current_gray.shape[0] - self.track_patch_radius - 1)
        x_min = max(x - search_radius, self.track_patch_radius)
        x_max = min(x + search_radius, current_gray.shape[1] - self.track_patch_radius - 1)

        if x_min > x_max or y_min > y_max:
            return None, 0.0

        step = 3
        for cy in range(y_min, y_max + 1, step):
            for cx in range(x_min, x_max + 1, step):
                candidate = self._extract_patch(current_gray, (cx, cy), self.track_patch_radius)
                if candidate is None:
                    continue
                score = self._normalized_cross_correlation(patch, candidate)
                if score > best_score:
                    best_score = score
                    best_point = np.array([cx, cy], dtype=np.float32)

        if best_point is None:
            return None, 0.0

        refined_point, refined_score = self._refine_corner_match(previous_gray, current_gray, patch, best_point)
        if refined_point is not None and refined_score >= best_score:
            return refined_point, refined_score
        return best_point, best_score

    def _refine_corner_match(self, previous_gray, current_gray, patch, coarse_point):
        del previous_gray
        x = int(round(coarse_point[0]))
        y = int(round(coarse_point[1]))
        best_score = -1.0
        best_point = None

        for cy in range(y - 2, y + 3):
            for cx in range(x - 2, x + 3):
                candidate = self._extract_patch(current_gray, (cx, cy), self.track_patch_radius)
                if candidate is None:
                    continue
                score = self._normalized_cross_correlation(patch, candidate)
                if score > best_score:
                    best_score = score
                    best_point = np.array([cx, cy], dtype=np.float32)

        return best_point, best_score

    def _extract_patch(self, gray, center, radius):
        x = int(round(center[0]))
        y = int(round(center[1]))
        if x - radius < 0 or y - radius < 0:
            return None
        if x + radius >= gray.shape[1] or y + radius >= gray.shape[0]:
            return None
        return gray[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(np.float32)

    def _normalized_cross_correlation(self, patch_a, patch_b):
        a = patch_a - float(np.mean(patch_a))
        b = patch_b - float(np.mean(patch_b))
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator <= 1e-6:
            return -1.0
        return float(np.sum(a * b) / denominator)

    def _validate_tracked_corners(self, corners, image_shape):
        height, width = image_shape[:2]
        if np.any(corners[:, 0] < 0) or np.any(corners[:, 0] >= width):
            return False
        if np.any(corners[:, 1] < 0) or np.any(corners[:, 1] >= height):
            return False
        if self._polygon_area(corners) < 500.0:
            return False
        if not self._is_convex_quad(corners):
            return False

        edge_lengths = [
            np.linalg.norm(corners[(idx + 1) % 4] - corners[idx])
            for idx in range(4)
        ]
        if min(edge_lengths) < 30.0:
            return False

        top_width = edge_lengths[0]
        right_height = edge_lengths[1]
        bottom_width = edge_lengths[2]
        left_height = edge_lengths[3]
        quad_width = max((top_width + bottom_width) * 0.5, 1.0)
        quad_height = max((left_height + right_height) * 0.5, 1.0)
        observed_ratio = max(quad_width, quad_height) / max(min(quad_width, quad_height), 1.0)
        expected_ratio = self.A4_HEIGHT / self.A4_WIDTH
        ratio_error = abs(observed_ratio - expected_ratio) / expected_ratio
        return ratio_error <= 0.65

    def _detect_centered_a4_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._white_paper_mask(gray)
        mask = self._erode_dilate(mask)
        components = self._connected_components(mask)

        if not components:
            return None

        frame_height, frame_width = gray.shape[:2]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        best = None
        best_score = None

        for component in components:
            candidate = self._component_to_candidate(component, frame_width, frame_height)
            if candidate is None:
                continue

            center_distance = np.linalg.norm(candidate["center"] - frame_center)
            center_score = center_distance / max(frame_width, frame_height)
            area_score = 1.0 - min(candidate["area_ratio"] / 0.40, 1.0)
            ratio_score = candidate["ratio_error"]
            score = center_score * 1.8 + ratio_score * 2.2 + area_score * 0.7

            if best_score is None or score < best_score:
                best = candidate
                best_score = score

        if best is None:
            return None

        return best["corners"]

    def _white_paper_mask(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = max(150, int(np.percentile(blurred, 72)))
        return (blurred >= threshold).astype(np.uint8)

    def _erode_dilate(self, mask):
        eroded = self._morph(mask, mode="erode", iterations=1)
        return self._morph(eroded, mode="dilate", iterations=2)

    def _morph(self, mask, mode, iterations):
        kernel = np.ones((3, 3), dtype=np.uint8)
        if mode == "erode":
            return cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

    def _connected_components(self, mask, min_pixels=None):
        height, width = mask.shape
        if min_pixels is None:
            min_pixels = max(300, int(width * height * 0.015))

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4)
        components = []
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if w <= 0 or h <= 0:
                continue

            local = labels[y : y + h, x : x + w] == label
            ys, xs = np.where(local)
            if len(xs) < min_pixels:
                continue
            pixels = np.column_stack([xs + x, ys + y]).astype(np.float32)
            components.append(pixels)

        return components

    def _build_hand_occlusion_mask(self, hand_landmarks, image_shape):
        if not hand_landmarks:
            return None

        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for landmarks in hand_landmarks:
            points = np.asarray(
                [
                    [
                        int(np.clip(point[0] * width, 0, width - 1)),
                        int(np.clip(point[1] * height, 0, height - 1)),
                    ]
                    for point in landmarks
                ],
                dtype=np.int32,
            )
            if points.shape[0] < 3:
                continue
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

        if np.count_nonzero(mask) == 0:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
        return cv2.dilate(mask, kernel, iterations=1)

    def _point_inside_hand(self, point):
        if self.hand_occlusion_mask is None:
            return False
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or y < 0 or y >= self.hand_occlusion_mask.shape[0] or x >= self.hand_occlusion_mask.shape[1]:
            return False
        return bool(self.hand_occlusion_mask[y, x] > 0)

    def _points_inside_hand(self, points):
        points = np.asarray(points, dtype=np.float32)
        return np.asarray([self._point_inside_hand(point) for point in points], dtype=bool)

    def _component_overlaps_hand(self, pixels):
        if self.hand_occlusion_mask is None or len(pixels) == 0:
            return False
        coords = np.asarray(np.round(pixels), dtype=np.int32)
        height, width = self.hand_occlusion_mask.shape[:2]
        inside = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < width)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < height)
        )
        if not np.any(inside):
            return False
        coords = coords[inside]
        overlap = self.hand_occlusion_mask[coords[:, 1], coords[:, 0]] > 0
        return float(np.count_nonzero(overlap)) / max(float(len(coords)), 1.0) >= 0.20

    def _component_to_candidate(self, pixels, frame_width, frame_height):
        x_min = float(np.min(pixels[:, 0]))
        x_max = float(np.max(pixels[:, 0]))
        y_min = float(np.min(pixels[:, 1]))
        y_max = float(np.max(pixels[:, 1]))

        box_width = x_max - x_min
        box_height = y_max - y_min
        if box_width < 40 or box_height < 40:
            return None

        box_area = box_width * box_height
        image_area = frame_width * frame_height
        area_ratio = box_area / image_area
        if area_ratio < 0.025 or area_ratio > 0.85:
            return None

        fill_ratio = len(pixels) / max(box_area, 1.0)
        if fill_ratio < 0.35:
            return None

        corners = self._estimate_quad_corners(pixels)
        if corners is None:
            return None

        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        quad_width = max((top_width + bottom_width) * 0.5, 1.0)
        quad_height = max((left_height + right_height) * 0.5, 1.0)

        observed_ratio = max(quad_width, quad_height) / max(min(quad_width, quad_height), 1.0)
        expected_ratio = self.A4_HEIGHT / self.A4_WIDTH
        ratio_error = abs(observed_ratio - expected_ratio) / expected_ratio
        if ratio_error > 0.45:
            return None

        return {
            "corners": corners,
            "center": np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32),
            "area_ratio": area_ratio,
            "ratio_error": ratio_error,
        }

    def _estimate_quad_corners(self, pixels):
        if len(pixels) < 4:
            return None

        boundary = self._component_boundary_pixels(pixels)
        if len(boundary) < 16:
            return None

        x = pixels[:, 0]
        y = pixels[:, 1]
        sum_xy = x + y
        diff_xy = x - y

        corners = np.array(
            [
                pixels[int(np.argmin(sum_xy))],
                pixels[int(np.argmax(diff_xy))],
                pixels[int(np.argmax(sum_xy))],
                pixels[int(np.argmin(diff_xy))],
            ],
            dtype=np.float32,
        )

        refined = self._refine_quad_from_boundary(boundary, corners)
        if refined is not None:
            corners = refined

        if self._polygon_area(corners) < 500.0:
            return None
        if not self._is_convex_quad(corners):
            return None
        return corners

    def _component_boundary_pixels(self, pixels):
        int_pixels = np.asarray(np.round(pixels), dtype=np.int32)
        x_min = int(np.min(int_pixels[:, 0]))
        y_min = int(np.min(int_pixels[:, 1]))
        x_max = int(np.max(int_pixels[:, 0]))
        y_max = int(np.max(int_pixels[:, 1]))

        local_width = x_max - x_min + 3
        local_height = y_max - y_min + 3
        local = np.zeros((local_height, local_width), dtype=np.uint8)

        lx = int_pixels[:, 0] - x_min + 1
        ly = int_pixels[:, 1] - y_min + 1
        local[ly, lx] = 1

        center = local[1:-1, 1:-1]
        up = local[0:-2, 1:-1]
        down = local[2:, 1:-1]
        left = local[1:-1, 0:-2]
        right = local[1:-1, 2:]
        boundary_mask = (center > 0) & ((up == 0) | (down == 0) | (left == 0) | (right == 0))

        by, bx = np.where(boundary_mask)
        boundary = np.column_stack([bx + x_min, by + y_min]).astype(np.float32)
        return boundary

    def _refine_quad_from_boundary(self, boundary, corners):
        refined = corners.astype(np.float32)
        for _ in range(2):
            lines = []
            for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
                edge_points = self._points_near_edge(boundary, refined[start], refined[end])
                if len(edge_points) < 10:
                    return None
                line = self._fit_line_tls(edge_points)
                if line is None:
                    return None
                lines.append(line)

            intersections = [
                self._intersect_lines(lines[3], lines[0]),
                self._intersect_lines(lines[0], lines[1]),
                self._intersect_lines(lines[1], lines[2]),
                self._intersect_lines(lines[2], lines[3]),
            ]
            if any(point is None for point in intersections):
                return None

            refined = np.asarray(intersections, dtype=np.float32)
            if self._polygon_area(refined) < 500.0 or not self._is_convex_quad(refined):
                return None

        return refined

    def _points_near_edge(self, points, p1, p2):
        edge = p2 - p1
        edge_length = np.linalg.norm(edge)
        if edge_length <= 1e-6:
            return np.empty((0, 2), dtype=np.float32)

        rel = points - p1
        direction = edge / edge_length
        projection = rel @ direction
        normal_distance = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])

        diagonal = np.linalg.norm(np.ptp(points, axis=0))
        threshold = max(4.0, diagonal * 0.025)
        along_mask = (projection >= -edge_length * 0.18) & (projection <= edge_length * 1.18)
        near_mask = normal_distance <= threshold
        selected = points[along_mask & near_mask]

        if len(selected) < 10:
            threshold = max(8.0, diagonal * 0.045)
            near_mask = normal_distance <= threshold
            selected = points[along_mask & near_mask]
        return selected.astype(np.float32)

    def _fit_line_tls(self, points):
        if len(points) < 2:
            return None

        centroid = points.mean(axis=0)
        centered = points - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        direction = vh[0]
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm <= 1e-9:
            return None
        normal = normal / norm
        c = -float(normal @ centroid)
        return np.array([normal[0], normal[1], c], dtype=np.float64)

    def _intersect_lines(self, line_a, line_b):
        point = np.cross(line_a, line_b)
        if abs(point[2]) <= 1e-9:
            return None
        return np.array([point[0] / point[2], point[1] / point[2]], dtype=np.float32)

    def _polygon_area(self, points):
        x = points[:, 0]
        y = points[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)

    def _is_convex_quad(self, points):
        signs = []
        for idx in range(4):
            a = points[idx]
            b = points[(idx + 1) % 4]
            c = points[(idx + 2) % 4]
            ab = b - a
            bc = c - b
            cross = ab[0] * bc[1] - ab[1] * bc[0]
            signs.append(cross)

        signs = np.asarray(signs)
        return bool(np.all(signs > 0) or np.all(signs < 0))

    def _board_points_for_corners(self, corners):
        return self._board_points_for_size(self._plane_size_for_image_quad(corners))

    def _board_points_for_size(self, plane_size):
        plane_width, plane_height = self._resolve_plane_size(plane_size)
        return np.array(
            [
                [0, 0],
                [plane_width, 0],
                [plane_width, plane_height],
                [0, plane_height],
            ],
            dtype=np.float32,
        )

    def _plane_size_for_image_quad(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (4, 2):
            return self.current_plane_size

        top = np.linalg.norm(points[1] - points[0])
        right = np.linalg.norm(points[2] - points[1])
        bottom = np.linalg.norm(points[2] - points[3])
        left = np.linalg.norm(points[3] - points[0])
        horizontal = (top + bottom) * 0.5
        vertical = (left + right) * 0.5

        if horizontal > vertical:
            return (self.A4_HEIGHT, self.A4_WIDTH)
        return (self.A4_WIDTH, self.A4_HEIGHT)

    def _resolve_plane_size(self, plane_size=None):
        if plane_size is None:
            plane_size = self.current_plane_size
        plane_width, plane_height = plane_size
        return float(plane_width), float(plane_height)
