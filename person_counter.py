import cv2


class PersonCounter:
    """
    Manages Person IN/OUT and INSIDE counting using center-point line crossing.
    Tracks state strictly by ByteTrack track_id with duplicate count prevention.
    """
    def __init__(self, line_position: float = 0.5, max_disappeared_frames: int = 30):
        # This code stores the counting line position ratio relative to frame height (0.0 to 1.0)
        self.line_position = line_position
        self.max_disappeared_frames = max_disappeared_frames

        # This code maintains cumulative IN and OUT line-crossing event counts
        self.in_count = 0
        self.out_count = 0

        # This code stores the set of track IDs currently inside the monitored area
        self.inside_ids = set()

        # This code stores the set of all unique ByteTrack track IDs observed across the session
        self.unique_track_ids = set()

        # This code stores each track's previous side of the counting line ('ABOVE' or 'BELOW')
        self.track_side = {}

        # This code records the last frame index a ByteTrack track ID was detected
        self.last_seen_frame = {}

        # This code tracks the current video frame index
        self.current_frame = 0

    def set_line_position(self, line_position: float):
        # This code dynamically updates the horizontal line position ratio
        self.line_position = max(0.05, min(0.95, line_position))

    def update(self, tracked_persons: list, frame_shape: tuple):
        """
        Updates line crossing state for all tracked persons in the current frame.
        """
        self.current_frame += 1
        height, width = frame_shape[:2]

        # This code calculates the horizontal counting line Y-coordinate from relative line position
        line_y = int(height * self.line_position)

        for person in tracked_persons:
            track_id = int(person['track_id'])
            x1, y1, x2, y2 = person['bbox']

            # This code tracks all unique ByteTrack track IDs seen throughout the video session
            self.unique_track_ids.add(track_id)

            # This code calculates the center Y-coordinate of the person bounding box
            center_y = (y1 + y2) / 2.0

            # This code determines whether the person's center point is ABOVE or BELOW the line
            current_side = 'ABOVE' if center_y < line_y else 'BELOW'

            # This code calculates frames since track was last seen to handle track loss
            frames_since_seen = self.current_frame - self.last_seen_frame.get(track_id, self.current_frame)
            track_was_lost = frames_since_seen > self.max_disappeared_frames

            # This code stores each track's previous side of the counting line so the same person is not counted on every frame
            if track_id in self.track_side and not track_was_lost:
                previous_side = self.track_side[track_id]

                # This code counts one IN event only when a tracked person changes from ABOVE the line to BELOW the line
                if previous_side == 'ABOVE' and current_side == 'BELOW':
                    self.in_count += 1
                    # This code stores the track ID as currently inside so the occupancy count can be calculated without duplicate entries
                    self.inside_ids.add(track_id)
                    self.track_side[track_id] = 'BELOW'

                # This code counts one OUT event only when a tracked person changes from BELOW the line to ABOVE the line
                elif previous_side == 'BELOW' and current_side == 'ABOVE':
                    self.out_count += 1
                    # This code removes the track ID from the currently inside set when exiting
                    self.inside_ids.discard(track_id)
                    self.track_side[track_id] = 'ABOVE'
            else:
                # This code initializes the side state for newly observed tracks without triggering a false count at video start or after long track loss
                self.track_side[track_id] = current_side

            # This code records the current frame index for track loss management
            self.last_seen_frame[track_id] = self.current_frame

        return tracked_persons

    def get_counts(self) -> dict:
        # This code calculates currently inside count using len(inside_ids) to handle initial occupancy safely
        return {
            'in': self.in_count,
            'out': self.out_count,
            'inside': len(self.inside_ids),
            'total_unique': len(self.unique_track_ids)
        }

    def draw_counter_overlay(self, frame):
        """
        Draws horizontal counting line, direction arrows, and statistics badge on frame.
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # This code draws the horizontal counting line across the video frame
        line_y = int(height * self.line_position)
        line_color = (0, 165, 255)  # Orange/Amber line
        cv2.line(annotated, (0, line_y), (width, line_y), line_color, 3)

        # Draw counting line text and direction indicators
        cv2.putText(annotated, f"COUNTING LINE (Y={line_y})", (15, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)
        cv2.putText(annotated, "OUT ^", (width - 80, line_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "IN v", (width - 80, line_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # This code draws the live IN, OUT, and INSIDE statistics overlay box on the video frame
        counts = self.get_counts()
        badge_text = f"IN: {counts['in']}  |  OUT: {counts['out']}  |  INSIDE: {counts['inside']}"
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), (30, 30, 30), -1)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), line_color, 2)
        cv2.putText(annotated, badge_text, (20, 10 + text_h + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def reset(self):
        # This code resets all counting metrics and tracking dictionaries for a new session
        self.in_count = 0
        self.out_count = 0
        self.current_frame = 0
        self.inside_ids.clear()
        self.unique_track_ids.clear()
        self.track_side.clear()
        self.last_seen_frame.clear()
