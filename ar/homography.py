import cv2
import numpy as np


class HomographyEstimator:
    """Direct homography utilities implemented with normalized DLT."""

    @staticmethod
    def compute_homography(src_points, dst_points):
        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("compute_homography expects exactly four 2D source and destination points")

        src_norm, src_t = HomographyEstimator._normalize_points(src)
        dst_norm, dst_t = HomographyEstimator._normalize_points(dst)

        rows = []
        for (x, y), (u, v) in zip(src_norm, dst_norm):
            rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        h_norm = vh[-1].reshape(3, 3)
        h = np.linalg.inv(dst_t) @ h_norm @ src_t

        if abs(h[2, 2]) > 1e-12:
            h = h / h[2, 2]
        return h.astype(np.float32)

    @staticmethod
    def _normalize_points(points):
        centroid = points.mean(axis=0)
        centered = points - centroid
        mean_distance = np.mean(np.linalg.norm(centered, axis=1))
        scale = np.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
        transform = np.array(
            [
                [scale, 0.0, -scale * centroid[0]],
                [0.0, scale, -scale * centroid[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
        normalized = (transform @ homogeneous.T).T[:, :2]
        return normalized, transform

    @staticmethod
    def transform_point(point, H):
        if H is None:
            return point

        p = np.array([point[0], point[1], 1.0], dtype=np.float64)
        p_transformed = np.asarray(H, dtype=np.float64) @ p
        if abs(p_transformed[2]) <= 1e-12:
            return (int(point[0]), int(point[1]))
        x = int(round(p_transformed[0] / p_transformed[2]))
        y = int(round(p_transformed[1] / p_transformed[2]))
        return (x, y)

    @staticmethod
    def transform_points(points, H):
        return [HomographyEstimator.transform_point(point, H) for point in points]

    @staticmethod
    def draw_grid_on_plane(frame, H, plane_size=(210, 297), grid_size=30, color=(50, 50, 50)):
        if H is None:
            return frame

        width, height = plane_size

        for x in range(0, width + 1, grid_size):
            p1 = HomographyEstimator.transform_point((x, 0), H)
            p2 = HomographyEstimator.transform_point((x, height), H)
            cv2.line(frame, p1, p2, color, 1)

        for y in range(0, height + 1, grid_size):
            p1 = HomographyEstimator.transform_point((0, y), H)
            p2 = HomographyEstimator.transform_point((width, y), H)
            cv2.line(frame, p1, p2, color, 1)

        return frame
