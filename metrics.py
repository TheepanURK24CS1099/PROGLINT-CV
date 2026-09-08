from collections import deque


class TrackingMetrics:
    """Computes rolling average FPS for real-time video processing."""
    def __init__(self, window_size: int = 30):
        self.frame_times = deque(maxlen=window_size)

    # Step 1: Record time taken for current frame
    def update(self, process_time_sec: float):
        if process_time_sec > 0:
            self.frame_times.append(process_time_sec)

    # Step 2: Compute average FPS
    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return (1.0 / avg_time) if avg_time > 0 else 0.0

    def reset(self):
        self.frame_times.clear()
