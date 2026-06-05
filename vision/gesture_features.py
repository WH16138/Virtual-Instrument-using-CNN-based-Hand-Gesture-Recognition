import numpy as np


FEATURE_SIZE = 63
GESTURE_CLASSES = ["Fist", "Open_Palm", "V_Sign", "OK_Sign"]


def landmarks_to_feature_vector(landmarks):
    """Convert 21 hand landmarks into a normalized 63-value feature vector."""
    if landmarks is None:
        return None

    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        return None

    wrist = points[0].copy()
    normalized = points - wrist

    scale = np.max(np.linalg.norm(normalized[:, :2], axis=1))
    if scale <= 1e-6:
        return None

    normalized = normalized / scale
    return normalized.reshape(FEATURE_SIZE).astype(np.float32)


def feature_vector_to_landmarks(feature_vector):
    """Convert a normalized 63-value vector back to 21x3 points for preview drawing."""
    values = np.asarray(feature_vector, dtype=np.float32)
    if values.shape != (FEATURE_SIZE,):
        return None
    return values.reshape(21, 3)
