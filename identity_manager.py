import cv2
import numpy as np


class IdentityManager:
    """
    Maintains persistent Person IDs (P001, P002...) across frames and re-entries
    using color appearance histograms.
    """
    def __init__(self, reid_threshold: float = 0.65):
        self.reid_threshold = reid_threshold
        self.next_id_num = 1
        self.track_to_person = {}      # byte_track_id -> person_id (e.g. 'P001')
        self.person_appearances = {}   # person_id -> HSV histogram
        self.total_unique_people = set()

    # Step 1: Extract color histogram from person crop
    def _extract_appearance(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    # Step 2: Update identities for the current frame
    def update(self, tracked_persons, frame):
        # This code is used to collect all persistent person IDs currently active on visible tracks
        # to prevent duplicate persistent ID assignments (e.g. assigning P003 to two simultaneous tracks).
        active_track_ids = {p['track_id'] for p in tracked_persons}
        occupied_person_ids = {
            self.track_to_person[tid] for tid in active_track_ids if tid in self.track_to_person
        }
        current_frame_person_ids = set(occupied_person_ids)

        for person in tracked_persons:
            track_id = person['track_id']
            bbox = person['bbox']

            # Case A: Track ID is already recognized in consecutive frames
            if track_id in self.track_to_person:
                person_id = self.track_to_person[track_id]
            else:
                # Case B: New Track ID -> Check if person is returning (Re-ID)
                hist = self._extract_appearance(frame, bbox)
                best_match_id = None
                best_similarity = -1.0

                if hist is not None:
                    for pid, stored_hist in self.person_appearances.items():
                        # Prevent assigning an ID that is currently in use by another active track
                        if pid not in current_frame_person_ids:
                            similarity = cv2.compareHist(hist, stored_hist, cv2.HISTCMP_CORREL)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_id = pid

                # If similarity exceeds threshold, restore their previous ID
                if best_similarity >= self.reid_threshold and best_match_id is not None:
                    person_id = best_match_id
                else:
                    # Otherwise, assign a new persistent ID (P001, P002...)
                    person_id = f"P{self.next_id_num:03d}"
                    self.next_id_num += 1

                self.track_to_person[track_id] = person_id
                if hist is not None:
                    self.person_appearances[person_id] = hist

            current_frame_person_ids.add(person_id)
            self.total_unique_people.add(person_id)
            person['person_id'] = person_id

        return tracked_persons

    def get_total_unique_count(self) -> int:
        return len(self.total_unique_people)
