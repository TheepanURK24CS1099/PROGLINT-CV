import cv2


class PersonCounter:
    """
    Manages Person IN/OUT and INSIDE counting using bottom-center (feet) ground anchor
    and state transition tracking strictly per ByteTrack track_id.
    """
    def __init__(self, line_position: float = 0.5, max_disappeared_frames: int = 30):
        # Stores counting line position ratio relative to frame height (0.0 to 1.0)
        self.line_position = line_position
        self.max_disappeared_frames = max_disappeared_frames

        # Cumulative IN and OUT line-crossing counts
        self.in_count = 0
        self.out_count = 0

        # Set of track IDs currently inside the monitored area
        self.inside_ids = set()

        # Set of all unique ByteTrack track IDs observed across the session
        self.unique_track_ids = set()

        # Each track's previous side of the counting line ('ABOVE' or 'BELOW')
        self.track_side = {}

        # Records the last counted direction ('IN' or 'OUT') per track_id to prevent double-counting
        self.counted_events = {}

        # Records the last frame index a ByteTrack track ID was detected
        self.last_seen_frame = {}

        # Tracks current video frame index
        self.current_frame = 0

    def set_line_position(self, line_position: float):
        # Dynamically updates horizontal line position ratio
        self.line_position = max(0.05, min(0.95, line_position))

    def _get_anchor_point(self, bbox: list) -> tuple:
        """Calculates (x_center, y_max) bottom-center anchor point (feet level)."""
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2.0
        y_max = float(y2)
        return int(x_center), int(y_max)

    def update(self, tracked_persons: list, frame_shape: tuple):
        """
        Updates line crossing state for all tracked persons using bottom-center anchor points.
        """
        self.current_frame += 1
        height, width = frame_shape[:2]

        # Calculates single horizontal counting line Y-coordinate
        line_y = int(height * self.line_position)

        for person in tracked_persons:
            track_id = int(person['track_id'])
            bbox = person['bbox']

            self.unique_track_ids.add(track_id)
            _, y_max = self._get_anchor_point(bbox)

            # Determine current side relative to single horizontal counting line
            current_side = 'ABOVE' if y_max < line_y else 'BELOW'

            frames_since_seen = self.current_frame - self.last_seen_frame.get(track_id, self.current_frame)
            track_was_lost = frames_since_seen > self.max_disappeared_frames

            if track_id in self.track_side and not track_was_lost:
                previous_side = self.track_side[track_id]

                # Count IN event only when transitioning from ABOVE to BELOW the line
                if previous_side == 'ABOVE' and current_side == 'BELOW':
                    if self.counted_events.get(track_id) != 'IN':
                        self.in_count += 1
                        self.inside_ids.add(track_id)
                        self.counted_events[track_id] = 'IN'
                    self.track_side[track_id] = 'BELOW'

                # Count OUT event only when transitioning from BELOW to ABOVE the line
                elif previous_side == 'BELOW' and current_side == 'ABOVE':
                    if self.counted_events.get(track_id) != 'OUT':
                        self.out_count += 1
                        self.inside_ids.discard(track_id)
                        self.counted_events[track_id] = 'OUT'
                    self.track_side[track_id] = 'ABOVE'
            else:
                # Initialize side state for newly detected track
                self.track_side[track_id] = current_side

            self.last_seen_frame[track_id] = self.current_frame

        return tracked_persons

    def get_counts(self) -> dict:
        return {
            'in': self.in_count,
            'out': self.out_count,
            'inside': len(self.inside_ids),
            'total_unique': len(self.unique_track_ids)
        }

    def draw_counter_overlay(self, frame):
        """
        Draws a single, crisp horizontal counting line across the frame (Cyan, thickness 2px),
        direction indicators, and statistics overlay badge.
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        line_y = int(height * self.line_position)
        line_color = (255, 255, 0)  # Crisp Cyan single horizontal line in BGR

        # Single crisp horizontal line (2px thickness, no extra bands/buffers)
        cv2.line(annotated, (0, line_y), (width, line_y), line_color, 2, cv2.LINE_AA)

        # Counting line text and direction indicators
        cv2.putText(annotated, f"COUNTING LINE (Y={line_y})", (15, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, "OUT ^", (width - 80, line_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "IN v", (width - 80, line_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Live IN, OUT, and INSIDE statistics overlay box
        counts = self.get_counts()
        badge_text = f"IN: {counts['in']}  |  OUT: {counts['out']}  |  INSIDE: {counts['inside']}"
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), (30, 30, 30), -1)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), line_color, 2)
        cv2.putText(annotated, badge_text, (20, 10 + text_h + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def reset(self):
        # Resets all counting metrics and tracking dictionaries for a new session
        self.in_count = 0
        self.out_count = 0
        self.current_frame = 0
        self.inside_ids.clear()
        self.unique_track_ids.clear()
        self.track_side.clear()
        self.counted_events.clear()
        self.last_seen_frame.clear()


