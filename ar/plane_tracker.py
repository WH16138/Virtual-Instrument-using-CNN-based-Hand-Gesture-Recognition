import cv2
import numpy as np

from ar.homography import HomographyEstimator


class PlaneTracker:
    """Detect a centered white A4 sheet and use it as the game board."""

    A4_WIDTH = 210
    A4_HEIGHT = 297
    MARK_INSET = 15

    def __init__(self):
        self.is_registered = False
        self.last_corners = None
        self.last_homography = None
        self.last_gray = None
        self.missed_frames = 0
        self.max_missed_frames = 10
        self.corner_smoothing = 0.35
        self.track_patch_radius = 8
        self.track_search_radius = 28
        self.min_track_score = 0.58
        self.last_marker_centers = None
        self.marker_lock_radius = 70.0
        self.marker_roi_radius = 72

    def register_plane(self, frame):
        result = self.track_plane(frame)
        if not result["success"]:
            print("A4 board registration failed. Place a white A4 sheet near the center of the camera view.")
            return False

        self.register_tracking_result(result)
        print("A4 board registered.")
        return True

    def register_tracking_result(self, result):
        if not result or not result.get("success"):
            return False

        self.is_registered = True
        self.last_corners = result["corners"]
        self.last_homography = result["H"]
        if result.get("marker_centers") is not None:
            self.last_marker_centers = result["marker_centers"]
        return True

    def track_plane(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        marker_result = self._detect_corner_marks(gray)
        if marker_result is not None:
            H = marker_result["H"]
            corners = marker_result["corners"]
            if self.is_registered:
                self.last_corners = corners
                self.last_homography = H
                self.last_gray = gray
                self.last_marker_centers = marker_result["marker_centers"]
                self.missed_frames = 0
            return {
                "success": True,
                "H": H,
                "matched_points": 4,
                "corners": corners,
                "marker_centers": marker_result["marker_centers"],
                "stale": False,
                "tracking_method": "corner_marks",
                "track_score": marker_result["score"],
            }

        if not self.is_registered:
            corners = self._detect_centered_a4_corners(frame)
            if corners is not None:
                return self._preview_result_from_corners(corners, "white_boundary", 1.0)
            return {"success": False, "H": None, "matched_points": 0, "corners": None}

        tracked = self._tracking_fallback(gray)
        if tracked is not None:
            return tracked

        self.missed_frames += 1
        if (
            self.last_corners is not None
            and self.last_homography is not None
            and self.missed_frames <= self.max_missed_frames
        ):
            return {
                "success": True,
                "H": self.last_homography,
                "matched_points": 4,
                "corners": self.last_corners,
                "marker_centers": None,
                "stale": True,
                "tracking_method": "hold_last",
                "track_score": 0.0,
            }
        return {"success": False, "H": None, "matched_points": 0, "corners": None}

    def _preview_result_from_corners(self, corners, method, score):
        source = self._board_points_for_corners(corners)
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
        }

    def _result_from_corners(self, corners, gray, method, score):
        corners = self._stabilize_corners(corners)
        source = self._board_points_for_corners(corners)
        H = HomographyEstimator.compute_homography(source, corners)
        self.last_corners = corners
        self.last_homography = H
        self.last_gray = gray
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
        }

    def _tracking_fallback(self, gray):
        marker_result = self._track_last_markers(gray)
        if marker_result is not None:
            return marker_result
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

        status = status.reshape(-1)
        if np.count_nonzero(status) < 4:
            return None

        marker_centers = next_points.reshape(-1, 2).astype(np.float32)
        movement = np.linalg.norm(marker_centers - self.last_marker_centers, axis=1)
        if np.max(movement) > self.marker_roi_radius * 1.5:
            return None

        if self._order_quad_points(marker_centers) is None:
            return None

        height, width = gray.shape[:2]
        frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        dummy_group = [{"area": 1.0} for _ in range(4)]
        if self._score_marker_quad(marker_centers, dummy_group, frame_center, width * height) is None:
            return None

        H = HomographyEstimator.compute_homography(self._marker_board_points(), marker_centers)
        board_corners = np.asarray(
            [
                HomographyEstimator.transform_point((0, 0), H),
                HomographyEstimator.transform_point((self.A4_WIDTH, 0), H),
                HomographyEstimator.transform_point((self.A4_WIDTH, self.A4_HEIGHT), H),
                HomographyEstimator.transform_point((0, self.A4_HEIGHT), H),
            ],
            dtype=np.float32,
        )
        if not self._validate_tracked_corners(board_corners, gray.shape):
            return None

        self.last_marker_centers = marker_centers
        self.last_corners = board_corners
        self.last_homography = H
        self.last_gray = gray
        self.missed_frames = 0
        return {
            "success": True,
            "H": H,
            "matched_points": 4,
            "corners": board_corners,
            "marker_centers": marker_centers,
            "stale": False,
            "tracking_method": "optical_flow",
            "track_score": 1.0 / (1.0 + float(np.mean(errors[status == 1]))),
        }

    def _detect_corner_marks(self, gray):
        if self.is_registered and self.last_homography is not None:
            return self._detect_corner_marks_near_prediction(gray)

        mask = self._dark_mark_mask(gray)
        image_area = gray.shape[0] * gray.shape[1]
        components = self._connected_components(mask, min_pixels=max(8, int(image_area * 0.000015)))
        if len(components) < 4:
            return None

        height, width = gray.shape[:2]
        frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        image_area = width * height
        anchors = self._marker_anchors(width, height)
        slots = {name: {"anchor": anchor, "candidates": []} for name, anchor in anchors.items()}

        for component in components:
            candidate = self._component_to_mark_candidate(component, image_area, width, height)
            if candidate is None:
                continue
            slot_name = self._candidate_corner_slot(candidate["blob_center"], frame_center)
            if slot_name is not None:
                mark_corner = self._estimate_l_mark_corner(component, slot_name)
                if mark_corner is None:
                    continue
                candidate["center"] = mark_corner
                slots[slot_name]["candidates"].append(candidate)

        selected = []
        for slot_name in ("tl", "tr", "br", "bl"):
            slot = slots[slot_name]
            if not slot["candidates"]:
                return None
            selected.append(self._select_best_mark_candidate(slot["candidates"], slot["anchor"]))

        ordered = np.asarray([item["center"] for item in selected], dtype=np.float32)
        if self._order_quad_points(ordered) is None:
            return None
        score = self._score_marker_quad(ordered, selected, frame_center, image_area)
        if score is None:
            return None

        marker_board_points = self._marker_board_points()
        H = HomographyEstimator.compute_homography(marker_board_points, ordered)
        board_corners = self._board_corners_from_homography(H)

        if not self._validate_tracked_corners(board_corners, gray.shape):
            return None

        return {
            "H": H,
            "corners": board_corners,
            "marker_centers": ordered,
            "score": 1.0 / (1.0 + float(score)),
        }

    def _detect_corner_marks_near_prediction(self, gray):
        predicted = np.asarray(
            [
                HomographyEstimator.transform_point(tuple(point), self.last_homography)
                for point in self._marker_board_points()
            ],
            dtype=np.float32,
        )

        if predicted.shape != (4, 2):
            return None

        marker_centers = []
        scores = []
        image_area = gray.shape[0] * gray.shape[1]
        for slot_name, point in zip(("tl", "tr", "br", "bl"), predicted):
            marker, score = self._detect_one_mark_near(gray, point, slot_name)
            if marker is None:
                return None
            marker_centers.append(marker)
            scores.append(score)

        ordered = np.asarray(marker_centers, dtype=np.float32)
        if self._order_quad_points(ordered) is None:
            return None

        frame_center = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0], dtype=np.float32)
        dummy_group = [{"area": 1.0} for _ in range(4)]
        if self._score_marker_quad(ordered, dummy_group, frame_center, image_area) is None:
            return None

        H = HomographyEstimator.compute_homography(self._marker_board_points(), ordered)
        board_corners = self._board_corners_from_homography(H)
        if not self._validate_tracked_corners(board_corners, gray.shape):
            return None

        return {
            "H": H,
            "corners": board_corners,
            "marker_centers": ordered,
            "score": float(np.mean(scores)),
        }

    def _detect_one_mark_near(self, gray, predicted, slot_name):
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

            candidate_center = mark_corner + np.array([x1, y1], dtype=np.float32)
            distance = np.linalg.norm(candidate_center - predicted)
            if distance > radius * 0.85:
                continue

            area_bonus = min(candidate["area"], 1000.0) * 0.018
            fill_penalty = abs(candidate["fill_ratio"] - 0.35) * 55.0
            score = distance + fill_penalty - area_bonus
            if best_score is None or score < best_score:
                best_score = score
                best = candidate_center

        if best is None:
            return None, 0.0
        return best, 1.0 / (1.0 + max(float(best_score), 0.0))

    def _board_corners_from_homography(self, H):
        return np.asarray(
            [
                HomographyEstimator.transform_point((0, 0), H),
                HomographyEstimator.transform_point((self.A4_WIDTH, 0), H),
                HomographyEstimator.transform_point((self.A4_WIDTH, self.A4_HEIGHT), H),
                HomographyEstimator.transform_point((0, self.A4_HEIGHT), H),
            ],
            dtype=np.float32,
        )

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
        threshold = min(120, int(np.percentile(blurred, 24)))
        mask = (blurred <= threshold).astype(np.uint8)
        mask = self._morph(mask, mode="dilate", iterations=2)
        mask = self._morph(mask, mode="erode", iterations=1)
        return mask

    def _component_to_mark_candidate(self, pixels, image_area, frame_width, frame_height, apply_border_margin=True):
        x_min = float(np.min(pixels[:, 0]))
        x_max = float(np.max(pixels[:, 0]))
        y_min = float(np.min(pixels[:, 1]))
        y_max = float(np.max(pixels[:, 1]))
        box_width = x_max - x_min + 1.0
        box_height = y_max - y_min + 1.0
        area = float(len(pixels))

        if area < max(12.0, image_area * 0.00002):
            return None
        if area > image_area * 0.025:
            return None
        if box_width < 3.0 or box_height < 3.0:
            return None

        aspect = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
        if aspect > 10.0:
            return None

        fill_ratio = area / max(box_width * box_height, 1.0)
        if fill_ratio < 0.04:
            return None

        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
        if apply_border_margin:
            margin_x = frame_width * 0.08
            margin_y = frame_height * 0.08
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
        def score(candidate):
            distance = np.linalg.norm(candidate["center"] - anchor)
            area_bonus = min(candidate["area"], 1200.0) * 0.02
            fill_penalty = abs(candidate["fill_ratio"] - 0.35) * 80.0
            return distance + fill_penalty - area_bonus

        return min(candidates, key=score)

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
            if marker_shift > self.marker_lock_radius:
                return None
        else:
            center = ordered.mean(axis=0)
            offsets = ordered - center
            if not (
                offsets[0, 0] < 0 and offsets[0, 1] < 0
                and offsets[1, 0] > 0 and offsets[1, 1] < 0
                and offsets[2, 0] > 0 and offsets[2, 1] > 0
                and offsets[3, 0] < 0 and offsets[3, 1] > 0
            ):
                return None

        center = ordered.mean(axis=0)
        center_score = np.linalg.norm(center - frame_center) / max(np.sqrt(image_area), 1.0)
        area_score = 1.0 - min(area_ratio / 0.35, 1.0)
        size_values = np.asarray([item["area"] for item in group], dtype=np.float32)
        size_balance = float(np.std(size_values) / max(np.mean(size_values), 1.0))
        return ratio_error * 2.0 + center_score * 1.2 + area_score * 0.5 + size_balance * 0.25

    def _marker_board_points(self):
        inset = self.MARK_INSET
        return np.asarray(
            [
                [inset, inset],
                [self.A4_WIDTH - inset, inset],
                [self.A4_WIDTH - inset, self.A4_HEIGHT - inset],
                [inset, self.A4_HEIGHT - inset],
            ],
            dtype=np.float32,
        )

    def _stabilize_corners(self, corners):
        if self.last_corners is None:
            return corners.astype(np.float32)

        displacement = np.linalg.norm(corners - self.last_corners, axis=1).mean()
        if displacement > 90.0:
            return corners.astype(np.float32)

        alpha = self.corner_smoothing
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
        return np.array(
            [
                [0, 0],
                [self.A4_WIDTH, 0],
                [self.A4_WIDTH, self.A4_HEIGHT],
                [0, self.A4_HEIGHT],
            ],
            dtype=np.float32,
        )
