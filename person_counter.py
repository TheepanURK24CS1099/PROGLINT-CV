import cv2


class PersonCounter:
    """
    Manages Person IN/OUT Counting by detecting when tracked persons cross a configurable counting line.
    Prevents duplicate counts using per-track position history and discrete side-state tracking.
    """
    def __init__(self, line_position: float = 0.5):
        # This code is used to store the counting line position ratio relative to frame height (0.0 to 1.0)
        self.line_position = line_position
        
        # This code is used to maintain cumulative IN and OUT crossing counts for person IN/OUT counting
        self.in_count = 0
        self.out_count = 0

        # This code is used to store the previous center position (cx, cy) of each tracked person
        # so that we can detect whether the person actually crossed the counting line.
        self.previous_positions = {}

        # This code is used to maintain the side state ('ABOVE' or 'BELOW') for each tracked person
        # to guarantee that a single line crossing event triggers exactly one count.
        self.track_side = {}

    def set_line_position(self, line_position: float):
        # This code is used to dynamically update the counting line position ratio when adjusted in the UI
        self.line_position = max(0.05, min(0.95, line_position))

    def update(self, tracked_persons: list, frame_shape: tuple):
        """
        Updates tracking positions and detects line crossings for person IN/OUT counting.
        """
        height, width = frame_shape[:2]

        # This code is used to calculate the Y-coordinate of the horizontal counting line for person IN/OUT counting
        line_y = int(height * self.line_position)

        for person in tracked_persons:
            # Using person_id (e.g., 'P001') if available, or track_id for tracking state consistency
            track_id = person.get('person_id', person['track_id'])
            bbox = person['bbox']

            # This code is used to calculate the center point of the person's bounding box for IN/OUT counting
            x1, y1, x2, y2 = bbox
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Determine current side relative to the counting line: 'ABOVE' (cy < line_y) or 'BELOW' (cy >= line_y)
            current_side = 'ABOVE' if cy < line_y else 'BELOW'

            # This code is used to check whether the tracked person's center crossed the counting line
            # and determine the IN or OUT direction without duplicate counting.
            if track_id in self.track_side:
                previous_side = self.track_side[track_id]

                # Detect top-to-bottom crossing across counting line (Moving DOWN -> IN direction)
                if previous_side == 'ABOVE' and current_side == 'BELOW':
                    self.in_count += 1
                    self.track_side[track_id] = 'BELOW'

                # Detect bottom-to-top crossing across counting line (Moving UP -> OUT direction)
                elif previous_side == 'BELOW' and current_side == 'ABOVE':
                    self.out_count += 1
                    self.track_side[track_id] = 'ABOVE'
            else:
                # First time person is detected: initialize side state without triggering false count
                self.track_side[track_id] = current_side

            # This code is used to store updated center position for that track_id for person IN/OUT counting
            self.previous_positions[track_id] = (cx, cy)

        return tracked_persons

    def get_counts(self) -> dict:
        # This code is used to calculate currently inside people count (IN - OUT) for person IN/OUT counting
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
        # This code is used to reset all counters and position histories for person IN/OUT counting
        self.in_count = 0
        self.out_count = 0
        self.previous_positions.clear()
        self.track_side.clear()
