import time
import cv2
from detector import load_detector, detect_persons
from tracker import init_tracker, track_persons
from identity_manager import IdentityManager
from metrics import TrackingMetrics


# =====================================================================
# STEP 4: DRAW BOUNDING BOXES & LABELS ON FRAME
# =====================================================================
def draw_tracks(frame, tracked_persons):
    """Draws bounding boxes and persistent ID labels (e.g. ID: P001 | Person | 0.95)."""
    annotated = frame.copy()
    box_color = (0, 255, 0)  # Green in BGR

    for person in tracked_persons:
        x1, y1, x2, y2 = map(int, person['bbox'])
        display_id = person.get('person_id', person['track_id'])
        conf = person['conf']

        # 1. Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # 2. Label badge
        label = f"ID: {display_id} | Person | {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - h - 8)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + w + 6, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    return annotated


# =====================================================================
# STEP 5: PROCESS SINGLE FRAME (STANDALONE FUNCTION)
# =====================================================================
def process_frame(frame, model, tracker, identity_manager, metrics, conf_thresh: float = 0.5):
    """
    Step-by-step frame processing:
    1. Detect persons with YOLOv8
    2. Track with ByteTrack
    3. Update persistent person IDs (P001, P002...)
    4. Draw bounding boxes & labels
    5. Update processing FPS
    """
    start_time = time.time()

    # Step 1: Detect
    boxes = detect_persons(model, frame, conf_thresh=conf_thresh)

    # Step 2: Track
    tracks = track_persons(tracker, boxes, frame)

    # Step 3: Identity Persistence
    if identity_manager is not None:
        tracks = identity_manager.update(tracks, frame)

    # Step 4: Draw
    annotated_frame = draw_tracks(frame, tracks)

    # Step 5: Metrics
    elapsed = time.time() - start_time
    if metrics is not None:
        metrics.update(elapsed)

    frame_metrics = {
        'active_count': len(tracks),
        'total_unique': identity_manager.get_total_unique_count() if identity_manager else len(tracks),
        'fps': metrics.get_fps() if metrics else 0.0
    }
    return annotated_frame, frame_metrics


# =====================================================================
# UNIFIED TRACKING PIPELINE CLASS
# =====================================================================
class TrackingPipeline:
    """Combines detector, tracker, identity manager, and metrics into one object."""
    def __init__(self, model_path: str = "models/best.pt"):
        self.model = load_detector(model_path)
        self.tracker = init_tracker()
        self.identity_manager = IdentityManager()
        self.metrics = TrackingMetrics()

    def process(self, frame, conf_thresh: float = 0.5):
        annotated_frame, frame_metrics = process_frame(
            frame, self.model, self.tracker, self.identity_manager, self.metrics, conf_thresh=conf_thresh
        )
        stats = {
            'active': frame_metrics['active_count'],
            'total': frame_metrics['total_unique'],
            'fps': frame_metrics['fps']
        }
        return annotated_frame, stats

    @property
    def total_unique_count(self) -> int:
        return self.identity_manager.get_total_unique_count()
