import time
from collections import deque

class TrackingMetrics:
    """
    TrackingMetrics computes real-time performance and tracking metrics,
    including rolling average processing FPS, current person count, total unique people count,
    and internal MOT tracking history logs.
    """
    def __init__(self, fps_window_size: int = 30):
        self.fps_window_size = fps_window_size
        self.frame_times = deque(maxlen=fps_window_size)
        self.history = []
        self.frame_count = 0

    def update(self, process_time_sec: float, current_tracks: list, total_unique_count: int):
        """
        Updates metrics for the current frame.

        Args:
            process_time_sec: Frame processing duration in seconds
            current_tracks: List of tracked person objects in current frame
            total_unique_count: Total unique Track IDs counted by tracker
        """
        self.frame_count += 1
        current_time = time.time()

        if process_time_sec > 0:
            self.frame_times.append(process_time_sec)

        # Log frame tracking details internally for future MOT evaluation metrics
        for t in current_tracks:
            self.history.append({
                'frame_num': self.frame_count,
                'timestamp': current_time,
                'track_id': t['track_id'],
                'bbox': t['bbox'],
                'confidence': t['conf']
            })

    def get_fps(self) -> float:
        """Calculates rolling average processing FPS."""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return (1.0 / avg_time) if avg_time > 0 else 0.0

    def reset(self):
        """Resets all metrics and history."""
        self.frame_times.clear()
        self.history.clear()
        self.frame_count = 0
