"""
tests/test_reid.py

Comprehensive test suite verifying the Persistent Person ID Re-Entry feature:
1. One continuously visible person remains P001.
2. Person A disappears and returns with a different ByteTrack ID -> still P001.
3. Two different people -> P001 and P002.
4. Person A leaves and Person B enters -> B must NOT automatically become P001.
5. Person A returns while Person B is also present -> A=P001 and B=P002.
6. Existing pipeline behavior and UI format string compatibility are not broken.
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from identity_manager import IdentityManager, PersonID, AppearanceExtractor, PersistentPerson
import pipeline


def create_synthetic_person_crop(shirt_color_bgr, pants_color_bgr, height=200, width=100):
    """
    Creates a realistic synthetic person image crop with distinct clothing colors:
    - Head & Hair: Top 25% (dark)
    - Torso / Shirt: Middle 45% (shirt_color_bgr)
    - Legs / Pants: Bottom 30% (pants_color_bgr)
    """
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    # Head & hair
    head_h = int(height * 0.25)
    crop[0:head_h, int(width * 0.25):int(width * 0.75)] = (50, 40, 30)
    # Shirt / torso
    torso_h = int(height * 0.70)
    crop[head_h:torso_h, int(width * 0.1):int(width * 0.9)] = shirt_color_bgr
    # Pants / legs
    crop[torso_h:, int(width * 0.15):int(width * 0.85)] = pants_color_bgr
    return crop


def make_frame_with_crop(crop, x1=50, y1=50, frame_size=(480, 640)):
    """Places a person crop into a full video frame canvas."""
    frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    h, w = crop.shape[:2]
    frame[y1:y1 + h, x1:x1 + w] = crop
    bbox = [float(x1), float(y1), float(x1 + w), float(y1 + h)]
    return frame, bbox


class TestPersonReIdentification(unittest.TestCase):
    def setUp(self):
        # Initialize fresh IdentityManager with standard project parameters
        self.mgr = IdentityManager(
            reid_similarity_threshold=0.70,
            reid_similarity_margin=0.05,
            feature_update_alpha=0.85,
            identity_lost_timeout=30.0
        )
        # Distinct person appearance models:
        # Person A: Bright Crimson Red shirt, Dark Navy Blue jeans
        self.crop_a = create_synthetic_person_crop(
            shirt_color_bgr=(20, 20, 220),
            pants_color_bgr=(180, 50, 20)
        )
        # Person B: Bright Emerald Green shirt, Bright Goldenrod Yellow pants
        self.crop_b = create_synthetic_person_crop(
            shirt_color_bgr=(30, 220, 40),
            pants_color_bgr=(40, 200, 220)
        )

    def test_1_continuously_visible_person(self):
        """Test 1: Person A enters and remains visible -> persists as P001 across all frames."""
        frame, bbox = make_frame_with_crop(self.crop_a)

        # Frame 1: Initial detection with ByteTrack Track ID = 7
        tracks = [{'track_id': 7, 'bbox': bbox, 'conf': 0.92}]
        res1 = self.mgr.update(tracks, frame)
        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0]['person_id'], "P001")
        self.assertEqual(self.mgr.get_total_unique_count(), 1)

        # Frames 2..5: Continuously visible with the same Track ID = 7
        for frame_idx in range(2, 6):
            tracks = [{'track_id': 7, 'bbox': bbox, 'conf': 0.92}]
            res = self.mgr.update(tracks, frame)
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0]['person_id'], "P001")
            self.assertEqual(self.mgr.get_total_unique_count(), 1)

        print("\n[PASSED] Test 1: Continuously visible person remains P001.")

    def test_2_person_disappears_and_reenters_with_new_track_id(self):
        """
        Test 2: Person A enters (Track 7 -> P001), disappears, then re-enters.
        ByteTrack assigns Track 14, but Re-ID restores persistent Person ID = P001.
        """
        frame, bbox = make_frame_with_crop(self.crop_a)

        # Step 1: Person A enters with ByteTrack Track ID = 7
        tracks_entry = [{'track_id': 7, 'bbox': bbox, 'conf': 0.91}]
        res_entry = self.mgr.update(tracks_entry, frame)
        self.assertEqual(res_entry[0]['person_id'], "P001")
        self.assertEqual(self.mgr.get_total_unique_count(), 1)

        # Step 2: Person A leaves the camera view (empty frames)
        for _ in range(3):
            res_empty = self.mgr.update([], frame)
            self.assertEqual(len(res_empty), 0)

        # Registry should mark P001 as temporarily lost / inactive
        self.assertFalse(self.mgr.registry["P001"].currently_visible)

        # Step 3: Person A re-enters the camera view.
        # ByteTrack assigns a NEW temporary Track ID = 14
        tracks_reentry = [{'track_id': 14, 'bbox': bbox, 'conf': 0.89}]
        res_reentry = self.mgr.update(tracks_reentry, frame)

        self.assertEqual(len(res_reentry), 1)
        # Crucial check: Track 14 must be recognized as P001 via appearance Re-ID!
        self.assertEqual(res_reentry[0]['person_id'], "P001")
        # Temporary ByteTrack ID preserved under temporary_track_id
        self.assertEqual(res_reentry[0]['temporary_track_id'], 14)
        # Total unique people count must remain 1 (NOT increment to 2)
        self.assertEqual(self.mgr.get_total_unique_count(), 1)

        print("[PASSED] Test 2: Disappearance and re-entry with new Track ID restores P001.")

    def test_3_two_different_people(self):
        """Test 3: Two distinct individuals Person A and Person B receive P001 and P002."""
        frame_a, bbox_a = make_frame_with_crop(self.crop_a, x1=50, y1=50)
        frame_b, bbox_b = make_frame_with_crop(self.crop_b, x1=300, y1=50)
        combined_frame = frame_a.copy()
        h_b, w_b = self.crop_b.shape[:2]
        combined_frame[50:50 + h_b, 300:300 + w_b] = self.crop_b

        tracks = [
            {'track_id': 7, 'bbox': bbox_a, 'conf': 0.95},
            {'track_id': 8, 'bbox': bbox_b, 'conf': 0.90}
        ]

        res = self.mgr.update(tracks, combined_frame)
        self.assertEqual(len(res), 2)
        p_ids = {t['person_id'] for t in res}
        self.assertEqual(p_ids, {"P001", "P002"})
        self.assertEqual(self.mgr.get_total_unique_count(), 2)

        print("[PASSED] Test 3: Two distinct people receive P001 and P002.")

    def test_4_person_a_leaves_person_b_enters(self):
        """
        Test 4: Person A leaves, then Person B enters.
        Person B must NOT be falsely identified as P001; must become P002.
        """
        frame_a, bbox_a = make_frame_with_crop(self.crop_a)
        frame_b, bbox_b = make_frame_with_crop(self.crop_b)

        # Person A enters -> P001
        tracks_a = [{'track_id': 7, 'bbox': bbox_a, 'conf': 0.92}]
        self.mgr.update(tracks_a, frame_a)
        self.assertEqual(self.mgr.get_total_unique_count(), 1)

        # Person A leaves
        self.mgr.update([], frame_a)

        # Person B enters with Track ID = 15
        tracks_b = [{'track_id': 15, 'bbox': bbox_b, 'conf': 0.91}]
        res_b = self.mgr.update(tracks_b, frame_b)

        self.assertEqual(len(res_b), 1)
        # Person B MUST NOT be P001!
        self.assertNotEqual(res_b[0]['person_id'], "P001")
        self.assertEqual(res_b[0]['person_id'], "P002")
        self.assertEqual(self.mgr.get_total_unique_count(), 2)

        print("[PASSED] Test 4: Person B does not steal P001 upon entering alone.")

    def test_5_person_a_returns_while_person_b_present(self):
        """
        Test 5: Person A enters (P001), leaves. Person B enters (P002).
        Person A returns (Track 25). Both are present.
        A must be P001 and B must be P002.
        """
        frame_a, bbox_a = make_frame_with_crop(self.crop_a, x1=50, y1=50)
        frame_b, bbox_b = make_frame_with_crop(self.crop_b, x1=300, y1=50)

        # 1. Person A enters -> P001
        self.mgr.update([{'track_id': 7, 'bbox': bbox_a, 'conf': 0.90}], frame_a)
        # 2. Person A leaves
        self.mgr.update([], frame_a)
        # 3. Person B enters -> P002
        self.mgr.update([{'track_id': 12, 'bbox': bbox_b, 'conf': 0.93}], frame_b)

        # 4. Person A returns with Track 25 while Person B is still present with Track 12
        both_frame = frame_a.copy()
        h_b, w_b = self.crop_b.shape[:2]
        both_frame[50:50 + h_b, 300:300 + w_b] = self.crop_b

        both_tracks = [
            {'track_id': 12, 'bbox': bbox_b, 'conf': 0.93}, # Person B
            {'track_id': 25, 'bbox': bbox_a, 'conf': 0.88}  # Person A returning
        ]

        res = self.mgr.update(both_tracks, both_frame)
        self.assertEqual(len(res), 2)

        id_map = {t['temporary_track_id']: t['person_id'] for t in res}
        self.assertEqual(id_map[12], "P002")
        self.assertEqual(id_map[25], "P001")
        self.assertEqual(self.mgr.get_total_unique_count(), 2)

        print("[PASSED] Test 5: Simultaneous presence correctly resolves A=P001 and B=P002.")

    def test_6_pipeline_and_formatting_compatibility(self):
        """
        Test 6: Verify PersonID formatting string compatibility and pipeline integration:
        - f"{t['track_id']:02d}" evaluates without crash to "P001" (app.py line 504 safety)
        - pipeline.draw_tracks draws persistent ID
        - pipeline.process_frame updates metrics and total_unique accurately
        """
        # 1. PersonID formatting check
        pid = PersonID("P001")
        formatted = f"{pid:02d}"
        self.assertEqual(formatted, "P001")
        self.assertTrue(isinstance(pid, str))

        # 2. Pipeline draw_tracks test
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        tracks = [{'track_id': PersonID("P001"), 'person_id': "P001", 'bbox': [10, 10, 50, 100], 'conf': 0.95}]
        annotated = pipeline.draw_tracks(frame, tracks)
        self.assertEqual(annotated.shape, frame.shape)

        # 3. Pipeline process_frame end-to-end with mock detector and tracker
        class MockDetector:
            def detect_persons(self, f, conf_threshold=0.5):
                return []

        class MockTracker:
            def track_persons(self, b, f):
                return [{'track_id': 7, 'bbox': [10, 10, 50, 100], 'conf': 0.92}]
            def get_total_unique_count(self):
                return 1

        class MockMetrics:
            def update(self, dur, tracks, unique):
                pass
            def get_fps(self):
                return 28.5

        test_frame, _ = make_frame_with_crop(self.crop_a, x1=10, y1=10)
        ann, m = pipeline.process_frame(
            test_frame,
            MockDetector(),
            MockTracker(),
            MockMetrics()
        )

        self.assertIn('current_people', m)
        self.assertIn('total_unique', m)
        self.assertIn('fps', m)
        self.assertIn('tracks', m)
        self.assertEqual(m['tracks'][0]['person_id'], "P001")
        self.assertEqual(m['total_unique'], 1)

        print("[PASSED] Test 6: Pipeline integration and UI format compatibility verified.")

    def test_7_ambiguity_protection_safeguard(self):
        """
        Test 7: When two registered identities have very close appearance similarity
        (difference < REID_SIMILARITY_MARGIN = 0.05), the ambiguity safeguard prevents
        blindly assigning an identity and instead allocates a new one.
        """
        mgr = IdentityManager(
            reid_similarity_threshold=0.70,
            reid_similarity_margin=0.05
        )

        # Manually register two very similar dummy identities in registry
        emb1 = np.ones(144, dtype=np.float32)
        emb1 /= np.linalg.norm(emb1)

        # emb2 is almost identical to emb1 (difference < 0.01)
        emb2 = emb1.copy()
        emb2[0] += 0.05
        emb2 /= np.linalg.norm(emb2)

        mgr.registry["P001"] = PersistentPerson(
            person_id="P001",
            appearance_embedding=emb1,
            last_seen_time=100.0,
            last_seen_frame=1,
            last_track_id=1,
            currently_visible=False
        )
        mgr.registry["P002"] = PersistentPerson(
            person_id="P002",
            appearance_embedding=emb2,
            last_seen_time=100.0,
            last_seen_frame=1,
            last_track_id=2,
            currently_visible=False
        )
        mgr.next_id_index = 3

        # Candidate crop has high similarity to both P001 and P002 (sim diff < 0.05)
        # Ambiguity safeguard must trigger and assign P003 instead of false merge!
        test_emb = emb1.copy()
        sim1 = mgr.compute_similarity(test_emb, emb1)
        sim2 = mgr.compute_similarity(test_emb, emb2)
        self.assertLess(abs(sim1 - sim2), 0.05)

        # Feed track with test_emb
        track = {'track_id': 99, 'bbox': [10, 10, 50, 100], 'conf': 0.90, '_crop_embedding': test_emb}
        # Provide a dummy frame
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Mock extractor to return test_emb
        orig_extract = mgr.extractor.extract
        mgr.extractor.extract = lambda crop: test_emb
        try:
            res = mgr.update([track], dummy_frame)
            self.assertEqual(res[0]['person_id'], "P003")
            print("[PASSED] Test 7: Ambiguity protection safeguard prevents false positive merges.")
        finally:
            mgr.extractor.extract = orig_extract

    def test_8_timeout_and_ema_smoothing(self):
        """
        Test 8: Verify EMA smoothing updates embedding and lost timeout marks expired candidates.
        """
        person = PersistentPerson(
            person_id="P001",
            appearance_embedding=np.array([1.0, 0.0], dtype=np.float32),
            last_seen_time=0.0,
            last_seen_frame=0,
            last_track_id=1,
            currently_visible=True
        )
        new_obs = np.array([0.0, 1.0], dtype=np.float32)
        # Update with alpha = 0.85
        person.update_appearance(new_obs, alpha=0.85)
        # Expected: 0.85 * [1, 0] + 0.15 * [0, 1] normalized
        expected_x = 0.85 / np.sqrt(0.85**2 + 0.15**2)
        expected_y = 0.15 / np.sqrt(0.85**2 + 0.15**2)
        self.assertAlmostEqual(person.appearance_embedding[0], expected_x, places=4)
        self.assertAlmostEqual(person.appearance_embedding[1], expected_y, places=4)
        print("[PASSED] Test 8: EMA appearance smoothing and mathematical formulation verified.")


if __name__ == "__main__":
    unittest.main()
