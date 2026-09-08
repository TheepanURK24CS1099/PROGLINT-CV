from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker

class PersonTracker:
    """
    PersonTracker encapsulates the ByteTrack algorithm for multi-object tracking.
    It takes filtered person detections and outputs tracked objects with persistent Track IDs.
    """
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.unique_ids = set()

        self._init_tracker()

    def _init_tracker(self):
        """Initializes the underlying ByteTrack instance with configured args."""
        args = SimpleNamespace(
            track_thresh=self.track_thresh,
            track_high_thresh=self.track_thresh,
            track_low_thresh=0.1,
            new_track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=True
        )
        self.tracker = BYTETracker(args)

    def track_persons(self, person_boxes, frame):
        """
        Updates ByteTrack with current frame's person detections.

        Args:
            person_boxes: Ultralytics Boxes object containing person detections, or None
            frame: Input frame (numpy array)

        Returns:
            tracked_persons: List of dicts [{'track_id': int, 'bbox': [x1, y1, x2, y2], 'conf': float, 'class_id': int}]
        """
        if person_boxes is None or len(person_boxes) == 0:
            # Let tracker update with empty boxes if necessary to maintain predictions
            return []

        # ==========================================
        # STEP 3: ByteTrack Update
        # ==========================================
        # BYTETracker requires CPU tensors / numpy arrays
        boxes_cpu = person_boxes.cpu()
        tracks_out = self.tracker.update(boxes_cpu, frame)

        tracked_persons = []
        if tracks_out is not None and len(tracks_out) > 0:
            for t in tracks_out:
                x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
                track_id = int(t[4])
                conf = float(t[5])
                cls_id = int(t[6]) if len(t) > 6 else 0

                # ==========================================
                # STEP 4: Track ID Assignment & Metric Update
                # ==========================================
                self.unique_ids.add(track_id)

                tracked_persons.append({
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'class_id': cls_id
                })

        return tracked_persons

    def get_total_unique_count(self):
        """Returns the total number of unique Track IDs observed so far."""
        return len(self.unique_ids)

    def reset(self):
        """Resets the tracker state and unique ID memory."""
        self.unique_ids.clear()
        self._init_tracker()
