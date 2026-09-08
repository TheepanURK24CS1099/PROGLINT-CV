import time
import cv2

def draw_tracks(frame, tracked_persons):
    """
    Draws bounding boxes and labels for tracked persons.

    Label format: ID: 12 | Person | 0.91
    """
    annotated_frame = frame.copy()
    box_color = (255, 144, 30)  # BGR color for bounding box (cyan/orange hue)
    text_color = (255, 255, 255) # White text

    for person in tracked_persons:
        bbox = person['bbox']
        track_id = person['track_id']
        conf = person['conf']

        x1, y1, x2, y2 = map(int, bbox)

        # 1. Draw Bounding Box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

        # 2. Prepare Label: ID: X | Person | Y.YY
        label = f"ID: {track_id} | Person | {conf:.2f}"

        # 3. Label Background Box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y1 = max(0, y1 - h - 10)
        label_y2 = max(h + 10, y1)
        cv2.rectangle(annotated_frame, (x1, label_y1), (x1 + w + 10, label_y2), box_color, -1)

        # 4. Text Overlay
        text_y = label_y2 - 5
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            2,
            cv2.LINE_AA
        )

    return annotated_frame


def process_frame(frame, detector, tracker, metrics, conf_threshold: float = 0.5):
    """
    Common frame processing pipeline function used for both Video Upload and Webcam inputs.

    Steps:
    1. Read frame
    2. Run YOLOv8
    3. Keep only person detections
    4. Apply confidence threshold
    5. Send detections to ByteTrack
    6. Receive tracked objects
    7. Draw bounding boxes and Track IDs
    8. Calculate FPS & update metrics
    9. Return annotated frame & metrics

    Returns:
        annotated_frame: Frame with drawn bounding boxes and IDs
        frame_metrics: Dict with current count, total unique count, and processing FPS
    """
    start_time = time.time()

    # STEP 2 & 3 & 4: YOLOv8 Detection & Person Filtering
    person_boxes = detector.detect_persons(frame, conf_threshold=conf_threshold)

    # STEP 5 & 6: ByteTrack Tracking
    tracked_persons = tracker.track_persons(person_boxes, frame)

    # STEP 7: Visualization
    annotated_frame = draw_tracks(frame, tracked_persons)

    end_time = time.time()
    process_duration = end_time - start_time

    # STEP 8 & 9: Calculate FPS and Update Metrics
    metrics.update(
        process_duration,
        tracked_persons,
        tracker.get_total_unique_count()
    )

    frame_metrics = {
        'current_people': len(tracked_persons),
        'total_unique': tracker.get_total_unique_count(),
        'fps': metrics.get_fps()
    }

    return annotated_frame, frame_metrics
