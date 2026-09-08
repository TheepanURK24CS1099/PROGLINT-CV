from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker


# =====================================================================
# STEP 1: INITIALIZE BYTETRACK
# =====================================================================
def init_tracker(track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
    """Initializes the ByteTrack multi-object tracker."""
    args = SimpleNamespace(
        track_thresh=track_thresh,
        track_high_thresh=track_thresh,
        track_low_thresh=0.1,
        new_track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=True
    )
    return BYTETracker(args)


# =====================================================================
# STEP 2: TRACK DETECTED PERSONS ACROSS FRAMES
# =====================================================================
def track_persons(tracker, person_boxes, frame):
    """
    Feeds person bounding boxes into ByteTrack.
    Returns: list of dicts [{'track_id': int, 'bbox': [x1, y1, x2, y2], 'conf': float}]
    """
    if person_boxes is None or len(person_boxes) == 0:
        return []

    tracks_out = tracker.update(person_boxes, frame)
    if tracks_out is None or len(tracks_out) == 0:
        return []

    return [
        {
            'track_id': int(t[4]),
            'bbox': [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
            'conf': float(t[5])
        }
        for t in tracks_out
    ]
