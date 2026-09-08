import time
import cv2
from detector import load_detector, detect_persons
from tracker import init_tracker, track_persons
from identity_manager import IdentityManager
from metrics import TrackingMetrics
# This code is used to import PersonCounter for person IN/OUT counting
from person_counter import PersonCounter


# =====================================================================
# STEP 4: DRAW BOUNDING BOXES & LABELS ON FRAME
# =====================================================================
def draw_tracks(frame, tracked_persons, person_counter=None):
    """Draws bounding boxes, persistent ID labels, and person IN/OUT counting line overlay."""
    annotated = frame.copy()
    box_color = (0, 255, 0)  # Green in BGR

    for person in tracked_persons:
        x1, y1, x2, y2 = map(int, person['bbox'])
        display_id = person.get('person_id', person['track_id'])
        conf = person['conf']

        # 1. Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # 2. Label badge (Format: ID: P001 | Person | 0.95)
        label = f"ID: {display_id} | Person | {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - h - 8)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + w + 6, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # This code is used to draw the counting line and IN/OUT/INSIDE overlay badge on the frame for person IN/OUT counting
    if person_counter is not None:
        annotated = person_counter.draw_counter_overlay(annotated)

    return annotated


# =====================================================================
# STEP 5: PROCESS SINGLE FRAME (STANDALONE FUNCTION)
# =====================================================================
def process_frame(frame, model, tracker, identity_manager, metrics, person_counter=None, conf_thresh: float = 0.5):
    """
    Step-by-step frame processing:
    1. Detect persons with YOLOv8
    2. Track with ByteTrack
    3. Update persistent person IDs (P001, P002...)
    4. Detect line crossing & update IN/OUT counting
    5. Draw bounding boxes, labels & counting line
    6. Update processing FPS
    """
    start_time = time.time()

    # Step 1: Detect
    boxes = detect_persons(model, frame, conf_thresh=conf_thresh)

    # Step 2: Track
    tracks = track_persons(tracker, boxes, frame)

    # Step 3: Identity Persistence
    if identity_manager is not None:
        tracks = identity_manager.update(tracks, frame)

    # This code is used to update person positions and detect line crossing for person IN/OUT counting
    if person_counter is not None:
        tracks = person_counter.update(tracks, frame.shape)

    # Step 4: Draw bounding boxes, labels, and counting line
    annotated_frame = draw_tracks(frame, tracks, person_counter=person_counter)

    # Step 5: Metrics
    elapsed = time.time() - start_time
    if metrics is not None:
        metrics.update(elapsed)

    # This code is used to compile person IN/OUT counting statistics (IN, OUT, INSIDE) along with FPS and active counts
    counts = person_counter.get_counts() if person_counter is not None else {'in': 0, 'out': 0, 'inside': 0}
    frame_metrics = {
        'active_count': len(tracks),
        'total_unique': identity_manager.get_total_unique_count() if identity_manager else len(tracks),
        'in_count': counts['in'],
        'out_count': counts['out'],
        'inside_count': counts['inside'],
        'fps': metrics.get_fps() if metrics else 0.0
    }
    return annotated_frame, frame_metrics


# =====================================================================
# UNIFIED TRACKING PIPELINE CLASS
# =====================================================================
class TrackingPipeline:
    """Combines detector, tracker, identity manager, person counter, and metrics into one object."""
    def __init__(self, model_path: str = "models/best.pt", *args, line_position: float = 0.5, **kwargs):
        # Handle line_position parameter flexibly whether passed positionally, via keyword, or in kwargs
        if 'line_position' in kwargs:
            line_position = kwargs['line_position']
        elif len(args) > 0 and isinstance(args[0], (int, float)):
            line_position = float(args[0])

        self.model = load_detector(model_path)
        self.tracker = init_tracker()
        self.identity_manager = IdentityManager()
        self.metrics = TrackingMetrics()
        # This code is used to initialize PersonCounter for person IN/OUT counting with configurable line position
        self.person_counter = PersonCounter(line_position=line_position)


    def set_line_position(self, line_pos: float):
        # This code is used to set the counting line position ratio dynamically for person IN/OUT counting
        self.person_counter.set_line_position(line_pos)

    def process(self, frame, conf_thresh: float = 0.5):
        annotated_frame, frame_metrics = process_frame(
            frame, self.model, self.tracker, self.identity_manager, self.metrics,
            person_counter=self.person_counter, conf_thresh=conf_thresh
        )
        # This code is used to populate stats dict with IN, OUT, and INSIDE metrics for person IN/OUT counting
        stats = {
            'active': frame_metrics['active_count'],
            'total': frame_metrics['total_unique'],
            'in': frame_metrics['in_count'],
            'out': frame_metrics['out_count'],
            'inside': frame_metrics['inside_count'],
            'fps': frame_metrics['fps']
        }
        return annotated_frame, stats

    def reset(self):
        # This code is used for IN/OUT person counting to reset state counters when processing a new video
        self.person_counter.reset()
        self.metrics.reset()

    @property
    def total_unique_count(self) -> int:
        return self.identity_manager.get_total_unique_count()


