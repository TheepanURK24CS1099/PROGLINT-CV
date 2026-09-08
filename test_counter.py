"""
Comprehensive Unit Test Suite for PersonCounter and IdentityManager in PROGLINT-CV.
Verifies line crossing detection, video start state, inside_ids tracking, duplicate prevention,
multi-track independence, track loss handling, and persistent ID uniqueness.
"""
import numpy as np
from person_counter import PersonCounter
from identity_manager import IdentityManager


def run_all_tests():
    frame_shape = (480, 640, 3)  # counting line at y = 240 (0.5 ratio)

    print("==================================================")
    print("RUNNING REVISED IN/OUT COUNTING & ID UNIT TEST SUITE")
    print("==================================================\n")

    # ----------------------------------------------------
    # TEST 1: Person starts ABOVE, then ABOVE -> BELOW (IN = 1, OUT = 0, INSIDE = 1)
    # ----------------------------------------------------
    counter1 = PersonCounter(line_position=0.5)
    counter1.update([{'track_id': 1, 'bbox': [100, 80, 150, 120]}], frame_shape)
    counter1.update([{'track_id': 1, 'bbox': [100, 280, 150, 320]}], frame_shape)
    c1 = counter1.get_counts()
    print("TEST 1 (Person enters):", c1)
    assert c1['in'] == 1 and c1['out'] == 0 and c1['inside'] == 1, f"Failed TEST 1: {c1}"

    # ----------------------------------------------------
    # TEST 2: Same person remains BELOW for 100 frames (IN remains 1, OUT = 0, INSIDE = 1)
    # ----------------------------------------------------
    for _ in range(100):
        counter1.update([{'track_id': 1, 'bbox': [100, 290, 150, 330]}], frame_shape)
    c2 = counter1.get_counts()
    print("TEST 2 (Remains BELOW 100 frames):", c2)
    assert c2['in'] == 1 and c2['out'] == 0 and c2['inside'] == 1, f"Failed TEST 2: {c2}"

    # ----------------------------------------------------
    # TEST 3: Same person crosses back BELOW -> ABOVE (IN = 1, OUT = 1, INSIDE = 0)
    # ----------------------------------------------------
    counter1.update([{'track_id': 1, 'bbox': [100, 80, 150, 120]}], frame_shape)
    c3 = counter1.get_counts()
    print("TEST 3 (Crosses back OUT):", c3)
    assert c3['in'] == 1 and c3['out'] == 1 and c3['inside'] == 0, f"Failed TEST 3: {c3}"

    # ----------------------------------------------------
    # TEST 4: Two different people enter (IN = 2)
    # ----------------------------------------------------
    counter4 = PersonCounter(line_position=0.5)
    two_above = [
        {'track_id': 10, 'bbox': [100, 80, 150, 120]},
        {'track_id': 20, 'bbox': [300, 80, 350, 120]}
    ]
    two_below = [
        {'track_id': 10, 'bbox': [100, 280, 150, 320]},
        {'track_id': 20, 'bbox': [300, 280, 350, 320]}
    ]
    counter4.update(two_above, frame_shape)
    counter4.update(two_below, frame_shape)
    c4 = counter4.get_counts()
    print("TEST 4 (Two people enter):", c4)
    assert c4['in'] == 2 and c4['inside'] == 2, f"Failed TEST 4: {c4}"

    # ----------------------------------------------------
    # TEST 5: Person already starts BELOW at frame 0 (IN = 0, OUT = 0, INSIDE = 0)
    # ----------------------------------------------------
    counter5 = PersonCounter(line_position=0.5)
    # Track 99 first observed already BELOW line -> must NOT trigger IN count!
    counter5.update([{'track_id': 99, 'bbox': [100, 300, 150, 340]}], frame_shape)
    counter5.update([{'track_id': 99, 'bbox': [100, 310, 150, 350]}], frame_shape)
    c5 = counter5.get_counts()
    print("TEST 5 (Starts BELOW at frame 0):", c5)
    assert c5['in'] == 0 and c5['out'] == 0 and c5['inside'] == 0, f"Failed TEST 5: {c5}"

    # ----------------------------------------------------
    # TEST 6: Person starts ABOVE and exits (ABOVE -> BELOW -> ABOVE => IN = 1, OUT = 1)
    # ----------------------------------------------------
    counter6 = PersonCounter(line_position=0.5)
    counter6.update([{'track_id': 50, 'bbox': [50, 50, 90, 90]}], frame_shape)
    counter6.update([{'track_id': 50, 'bbox': [50, 300, 90, 340]}], frame_shape)
    counter6.update([{'track_id': 50, 'bbox': [50, 50, 90, 90]}], frame_shape)
    c6 = counter6.get_counts()
    print("TEST 6 (Enters then exits):", c6)
    assert c6['in'] == 1 and c6['out'] == 1 and c6['inside'] == 0, f"Failed TEST 6: {c6}"

    # ----------------------------------------------------
    # TEST 7: Temporary track loss (No fake crossing)
    # ----------------------------------------------------
    counter7 = PersonCounter(line_position=0.5, max_disappeared_frames=10)
    counter7.update([{'track_id': 7, 'bbox': [100, 80, 150, 120]}], frame_shape)  # Above
    for _ in range(20):  # Simulate 20 missing frames
        counter7.update([], frame_shape)
    # Returns on the other side after track loss -> must NOT count fake crossing
    counter7.update([{'track_id': 7, 'bbox': [100, 300, 150, 340]}], frame_shape)
    c7 = counter7.get_counts()
    print("TEST 7 (Temporary track loss):", c7)
    assert c7['in'] == 0 and c7['out'] == 0, f"Failed TEST 7: {c7}"

    # ----------------------------------------------------
    # TEST 8: Two people cross simultaneously
    # ----------------------------------------------------
    counter8 = PersonCounter(line_position=0.5)
    counter8.update([
        {'track_id': 101, 'bbox': [10, 50, 50, 90]},
        {'track_id': 102, 'bbox': [200, 50, 240, 90]}
    ], frame_shape)
    counter8.update([
        {'track_id': 101, 'bbox': [10, 300, 50, 340]},
        {'track_id': 102, 'bbox': [200, 300, 240, 340]}
    ], frame_shape)
    c8 = counter8.get_counts()
    print("TEST 8 (Two people cross simultaneously):", c8)
    assert c8['in'] == 2 and c8['inside'] == 2, f"Failed TEST 8: {c8}"

    # ----------------------------------------------------
    # TEST 9: Same track remains on same side (No repeated counting)
    # ----------------------------------------------------
    counter9 = PersonCounter(line_position=0.5)
    for _ in range(30):
        counter9.update([{'track_id': 88, 'bbox': [100, 50, 150, 90]}], frame_shape)
    c9 = counter9.get_counts()
    print("TEST 9 (Same side motion):", c9)
    assert c9['in'] == 0 and c9['out'] == 0, f"Failed TEST 9: {c9}"

    # ----------------------------------------------------
    # TEST 10: Prevention of duplicate persistent IDs in IdentityManager
    # ----------------------------------------------------
    id_mgr = IdentityManager()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    id_mgr.update([{'track_id': 1, 'bbox': [10, 10, 100, 100]}], dummy_frame)
    t_both = [
        {'track_id': 1, 'bbox': [10, 10, 100, 100]},
        {'track_id': 2, 'bbox': [200, 200, 300, 300]}
    ]
    id_mgr.update(t_both, dummy_frame)
    p1 = t_both[0]['person_id']
    p2 = t_both[1]['person_id']
    print(f"TEST 10 (Unique active person IDs): Track 1 -> {p1}, Track 2 -> {p2}")
    assert p1 != p2, f"Failed TEST 10: Simultaneous active tracks shared ID {p1}"

    print("\n[SUCCESS] ALL 10 UNIT TESTS PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_all_tests()
