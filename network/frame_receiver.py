import threading
import time

import numpy as np


class FrameReceiver:
    """Hold only the latest frame received from the smartphone client."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self._last_update = None
        self._client_connected = False

    def update_frame(self, frame: np.ndarray):
        """Replace the latest frame with the newly received one."""
        if frame is None:
            return
        with self._lock:
            self._latest_frame = frame
            self._last_update = time.monotonic()

    def get_latest_frame(self):
        """Return the newest available frame or None if no frame has arrived."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def set_client_connected(self, connected: bool):
        with self._lock:
            self._client_connected = connected

    def is_client_connected(self):
        with self._lock:
            return self._client_connected

    def get_last_update_age(self):
        """Return seconds since the latest frame arrived, or None if no frame arrived."""
        with self._lock:
            if self._last_update is None:
                return None
            return time.monotonic() - self._last_update

    def is_frame_fresh(self, max_age_seconds: float):
        """Return True when a frame exists and was updated recently."""
        age = self.get_last_update_age()
        return age is not None and age <= max_age_seconds
