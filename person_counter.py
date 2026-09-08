import cv2


class PersonCounter:
    """
    Manages Person IN/OUT Counting by detecting line-crossing transitions of tracked persons.
    Keyed strictly by ByteTrack track_id with inside_ids tracking and video start state handling.
    """
    def __init__(self, line_position: float = 0.5, max_disappeared_frames: int = 30):
        # This code is used for IN/OUT person counting to store the counting line position ratio relative to frame height (0.0 to 1.0)
        self.line_position = line_position
        
        # This code is used for IN/OUT person counting to maintain cumulative IN and OUT line-crossing event counts
        self.in_count = 0
        self.out_count = 0

        # This code is used for IN/OUT person counting to track currently inside ByteTrack track_ids using a set
        self.inside_ids = set()

        # This code is used for IN/OUT person counting to store previous center position (cx, cy) keyed strictly by ByteTrack track_id
        self.previous_positions = {}

        # This code is used for IN/OUT person counting to maintain side state ('ABOVE' or 'BELOW') keyed strictly by ByteTrack track_id
        self.track_side = {}

        # This code is used for IN/OUT person counting to record the last frame index a ByteTrack track_id was observed
        self.last_seen_frame = {}

        # This code is used for IN/OUT person counting to track frame index and maximum track loss frames
        self.current_frame = 0
        self.max_disappeared_frames = max_disappeared_frames

    def set_line_position(self, line_position: float):
        # This code is used for IN/OUT person counting to dynamically update counting line position ratio
        self.line_position = max(0.05, min(0.95, line_position))

    def update(self, tracked_persons: list, frame_shape: tuple):
        """
        Updates tracking positions and detects line crossings for person IN/OUT counting using ByteTrack track_id.
        """
        self.current_frame += 1
        height, width = frame_shape[:2]

        # This code is used for IN/OUT person counting to calculate horizontal counting line Y-coordinate
        # ABOVE region: cy < line_y (Entry / Outside region)
        # BELOW region: cy >= line_y (Exit / Inside region)
        line_y = int(height * self.line_position)

        for person in tracked_persons:
            # IMPORTANT: Use raw ByteTrack track_id for crossing state to ensure separate independent track state
            track_id = int(person['track_id'])
            bbox = person['bbox']

            # This code is used for IN/OUT person counting to calculate center point (cx, cy) of bounding box
            x1, y1, x2, y2 = bbox
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Determine current side relative to counting line:
            # 'ABOVE' (cy < line_y) -> Entry / Outside region
            # 'BELOW' (cy >= line_y) -> Inside region
            current_side = 'ABOVE' if cy < line_y else 'BELOW'

            # This code is used for IN/OUT person counting to handle video start state and track disappearance.
            # If a track is seen for the first time or was lost for > max_disappeared_frames, initialize side state without counting.
            frames_since_last_seen = self.current_frame - self.last_seen_frame.get(track_id, self.current_frame)
            track_was_lost = frames_since_last_seen > self.max_disappeared_frames

            if track_id in self.track_side and not track_was_lost:
                previous_side = self.track_side[track_id]

                # This code is used for IN/OUT person counting to detect top-to-bottom line crossing (ABOVE -> BELOW => IN count +1)
                if previous_side == 'ABOVE' and current_side == 'BELOW':
                    self.in_count += 1
                    # This code is used for IN/OUT person counting to store the track ID inside the current-inside set after a valid IN crossing
                    self.inside_ids.add(track_id)
                    self.track_side[track_id] = 'BELOW'

                # This code is used for IN/OUT person counting to detect bottom-to-top line crossing (BELOW -> ABOVE => OUT count +1)
                elif previous_side == 'BELOW' and current_side == 'ABOVE':
                    self.out_count += 1
                    # This code is used for IN/OUT person counting to safely remove track ID from current-inside set after a valid OUT crossing
                    self.inside_ids.discard(track_id)
                    self.track_side[track_id] = 'ABOVE'
            else:
                # Video start state or track re-appearance: initialize side state without incrementing counters
                self.track_side[track_id] = current_side

            # This code is used for IN/OUT person counting to store center position and last seen frame for ByteTrack track_id
            self.previous_positions[track_id] = (cx, cy)
            self.last_seen_frame[track_id] = self.current_frame

        return tracked_persons

    def get_counts(self) -> dict:
        # This code is used for IN/OUT person counting to return currently inside count using len(inside_ids)
        currently_inside = len(self.inside_ids)
        return {
            'in': self.in_count,
            'out': self.out_count,
            'inside': currently_inside
        }

    def draw_counter_overlay(self, frame):
        """
        Draws the counting line and IN / OUT / INSIDE stats overlay badge on the frame.
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # This code is used for IN/OUT person counting to draw the horizontal line across video frame
        line_y = int(height * self.line_position)
        line_color = (0, 165, 255)  # Amber line in BGR
        cv2.line(annotated, (0, line_y), (width, line_y), line_color, 3)

        # Draw "COUNTING LINE" text label on the line
        label_text = f"--- COUNTING LINE (Y={line_y}) ---"
        cv2.putText(annotated, label_text, (15, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)

        # Draw direction arrows near line for clear IN/OUT visual indication
        cv2.putText(annotated, "OUT ^", (width - 80, line_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "IN v", (width - 80, line_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # This code is used for IN/OUT person counting to draw IN, OUT, and INSIDE overlay box on video frame
        counts = self.get_counts()
        badge_text = f"IN: {counts['in']}  |  OUT: {counts['out']}  |  INSIDE: {counts['inside']}"
        
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), (30, 30, 30), -1)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), line_color, 2)
        
        cv2.putText(annotated, badge_text, (20, 10 + text_h + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def reset(self):
        # This code is used for IN/OUT person counting to reset all counters and state tracking dictionaries
        self.in_count = 0
        self.out_count = 0
        self.current_frame = 0
        self.inside_ids.clear()
        self.previous_positions.clear()
        self.track_side.clear()
        self.last_seen_frame.clear()
