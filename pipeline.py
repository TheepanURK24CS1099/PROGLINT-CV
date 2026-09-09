import time
import cv2
from collections import deque
from detector import load_detector, detect_persons
from tracker import init_tracker, track_persons
from person_counter import PersonCounter


# =====================================================================
# STEP 1: DRAW BOUNDING BOXES & LABELS ON FRAME
# =====================================================================
def draw_tracks(frame, tracked_persons, person_counter=None):
    """Draws bounding boxes, ByteTrack ID labels, and counting overlay on the frame."""
    annotated = frame.copy()
    box_color = (0, 255, 0)  # Green box in BGR

    for person in tracked_persons:
        x1, y1, x2, y2 = map(int, person['bbox'])
        track_id = person['track_id']
        conf = person['conf']

        # This code draws bounding box around tracked person
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # This code draws text badge displaying ByteTrack track ID and confidence (e.g. ID: 7 | Person | 0.82)
        label = f"ID: {track_id} | Person | {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - h - 8)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + w + 6, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # This code draws horizontal counting line and IN/OUT/INSIDE stats overlay badge
    if person_counter is not None:
        annotated = person_counter.draw_counter_overlay(annotated)

    return annotated


# =====================================================================
# STEP 2: UNIFIED TRACKING PIPELINE CLASS
# =====================================================================
class TrackingPipeline:
    """Combines YOLOv8 detector, ByteTrack tracker, and PersonCounter into a single pipeline."""
    def __init__(self, model_path: str = "models/best.pt", line_position: float = 0.5):
        # This code initializes YOLOv8 detector, ByteTrack tracker, PersonCounter, and FPS history buffer
        self.model = load_detector(model_path)
        self.tracker = init_tracker()
        self.person_counter = PersonCounter(line_position=line_position)
        self.frame_times = deque(maxlen=30)

    def set_line_position(self, line_pos: float):
        # This code dynamically updates counting line position ratio
        self.person_counter.set_line_position(line_pos)

    def process(self, frame, conf_thresh: float = 0.5):
        """Processes a single video frame and returns the annotated frame and metrics."""
        start_time = time.time()

        # Step 1: Detect persons using YOLOv8
        boxes = detect_persons(self.model, frame, conf_thresh=conf_thresh)

        # Step 2: Track persons using ByteTrack
        tracks = track_persons(self.tracker, boxes, frame)

        # Step 3: Update person counter using center-point line crossing logic
        tracks = self.person_counter.update(tracks, frame.shape)

        # Step 4: Draw bounding boxes, track IDs, and counting line
        annotated_frame = draw_tracks(frame, tracks, person_counter=self.person_counter)

        # Step 5: Compute rolling average processing FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            self.frame_times.append(elapsed)
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0
        fps = (1.0 / avg_time) if avg_time > 0 else 0.0

        # Step 6: Compile live statistics dictionary (IN, OUT, INSIDE, Total Unique, FPS)
        counts = self.person_counter.get_counts()
        stats = {
            'active': len(tracks),
            'total': counts['total_unique'],
            'in': counts['in'],
            'out': counts['out'],
            'inside': counts['inside'],
            'fps': fps
        }

        return annotated_frame, stats

    def reset(self):
        # This code resets counter state and FPS timing history
        self.person_counter.reset()
        self.frame_times.clear()

    @property
    def total_unique_count(self) -> int:
        # This code returns total number of unique ByteTrack IDs observed
        return len(self.person_counter.unique_track_ids)
