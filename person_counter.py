import cv2


class PersonCounter:
    """
    Manages Person IN/OUT Counting by detecting when tracked persons cross a configurable counting line.
    Prevents duplicate counts using per-track position history, discrete side-state tracking,
    and ByteTrack track_id state keying.
    """
    def __init__(self, line_position: float = 0.5, max_disappeared_frames: int = 30):
        # This code is used to store the counting line position ratio relative to frame height (0.0 to 1.0)
        self.line_position = line_position
        
        # This code is used to maintain cumulative IN and OUT line-crossing event counts
        self.in_count = 0
        self.out_count = 0

        # This code is used to store the previous center position (cx, cy) keyed strictly by ByteTrack track_id
        self.previous_positions = {}

        # This code is used to maintain the side state ('ABOVE' or 'BELOW') keyed strictly by ByteTrack track_id
        # to guarantee that a single line crossing event produces exactly one counting event.
        self.track_side = {}

        # This code is used to record the last frame index a ByteTrack track_id was observed
        # to handle temporary track loss and prevent fake crossing counts upon track re-entry.
        self.last_seen_frame = {}

        # This code is used to track current frame index and maximum allowed track disappearance frames
        self.current_frame = 0
        self.max_disappeared_frames = max_disappeared_frames

    def set_line_position(self, line_position: float):
        # This code is used to dynamically update the counting line position ratio when adjusted in the UI
        self.line_position = max(0.05, min(0.95, line_position))

    def update(self, tracked_persons: list, frame_shape: tuple):
        """
        Updates tracking positions and detects line crossings for person IN/OUT counting using ByteTrack track_id.
        """
        self.current_frame += 1
        height, width = frame_shape[:2]

        # This code is used to calculate the Y-coordinate of the horizontal counting line for person IN/OUT counting
        line_y = int(height * self.line_position)

        for person in tracked_persons:
            # IMPORTANT: Use raw ByteTrack track_id for crossing state to ensure separate independent track state
            track_id = int(person['track_id'])
            bbox = person['bbox']

            # This code is used to calculate the center point (cx, cy) of the person's bounding box for IN/OUT counting
            x1, y1, x2, y2 = bbox
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Determine current side relative to counting line:
            # 'ABOVE' (cy < line_y) -> Entry / Outside region
            # 'BELOW' (cy >= line_y) -> Inside region
            current_side = 'ABOVE' if cy < line_y else 'BELOW'

            # This code is used to check whether the track_id was previously lost for too many frames.
            # If a track was missing for > max_disappeared_frames, re-initialize state without counting a fake crossing event.
            frames_since_last_seen = self.current_frame - self.last_seen_frame.get(track_id, self.current_frame)
            track_was_lost = frames_since_last_seen > self.max_disappeared_frames

            # This code is used to detect line crossing transitions (ABOVE <-> BELOW) for person IN/OUT counting
            if track_id in self.track_side and not track_was_lost:
                previous_side = self.track_side[track_id]

                # Detect top-to-bottom line crossing transition (Moving DOWN: ABOVE -> BELOW => IN count +1)
                if previous_side == 'ABOVE' and current_side == 'BELOW':
                    self.in_count += 1
                    self.track_side[track_id] = 'BELOW'

                # Detect bottom-to-top line crossing transition (Moving UP: BELOW -> ABOVE => OUT count +1)
                elif previous_side == 'BELOW' and current_side == 'ABOVE':
                    self.out_count += 1
                    self.track_side[track_id] = 'ABOVE'
            else:
                # First observation or re-appearance after track loss: initialize side state without incrementing counters
                self.track_side[track_id] = current_side

            # This code is used to update previous center position and last seen frame for this ByteTrack track_id
            self.previous_positions[track_id] = (cx, cy)
            self.last_seen_frame[track_id] = self.current_frame

        return tracked_persons

    def get_counts(self) -> dict:
        # This code is used to calculate currently inside people count (INSIDE = IN - OUT) for person IN/OUT counting
        currently_inside = max(0, self.in_count - self.out_count)
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

        # This code is used to calculate line Y position for drawing the counting line on the video frame
        line_y = int(height * self.line_position)

        # Draw horizontal counting line across the frame
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

        # This code is used to draw IN, OUT, and INSIDE overlay box on video frame for person IN/OUT counting
        counts = self.get_counts()
        badge_text = f"IN: {counts['in']}  |  OUT: {counts['out']}  |  INSIDE: {counts['inside']}"
        
        # Overlay background rectangle in top left corner
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), (30, 30, 30), -1)
        cv2.rectangle(annotated, (10, 10), (text_w + 30, text_h + 25), line_color, 2)
        
        cv2.putText(annotated, badge_text, (20, 10 + text_h + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def reset(self):
        # This code is used to reset all counters and track state histories for person IN/OUT counting
        self.in_count = 0
        self.out_count = 0
        self.current_frame = 0
        self.previous_positions.clear()
        self.track_side.clear()
        self.last_seen_frame.clear()
