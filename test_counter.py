"""
Comprehensive Unit Test Suite for PersonCounter and IdentityManager in PROGLINT-CV.
Tests line crossing detection, duplicate prevention, multi-track independence, track loss handling,
and persistent ID uniqueness.
"""
import numpy as np
from person_counter import PersonCounter
from identity_manager import IdentityManager


def run_all_tests():
    frame_shape = (480, 640, 3)  # counting line at y = 240 (0.5 ratio)

    print("==================================================")
    print("RUNNING IN/OUT COUNTING & ID UNIT TEST SUITE")
    print("==================================================\n")

    # ----------------------------------------------------
    # TEST 1: One person enters (ABOVE -> BELOW)
    # ----------------------------------------------------
    counter = PersonCounter(line_position=0.5)
    person1_above = [{'track_id': 1, 'bbox': [100, 80, 150, 120]}]
    person1_below = [{'track_id': 1, 'bbox': [100, 280, 150, 320]}]

    counter.update(person1_above, frame_shape)
    counter.update(person1_below, frame_shape)
    c1 = counter.get_counts()
    print("TEST 1 (One person enters):", c1)
    assert c1['in'] == 1 and c1['out'] == 0 and c1['inside'] == 1, f"Failed TEST 1: {c1}"

    # ----------------------------------------------------
    # TEST 2: Person remains inside for many frames
    # ----------------------------------------------------
    for _ in range(50):
        counter.update(person1_below, frame_shape)
    c2 = counter.get_counts()
    print("TEST 2 (Person remains inside):", c2)
    assert c2['in'] == 1 and c2['out'] == 0 and c2['inside'] == 1, f"Failed TEST 2: {c2}"

    # ----------------------------------------------------
    # TEST 3: Same person exits (BELOW -> ABOVE)
    # ----------------------------------------------------
    counter.update(person1_above, frame_shape)
    c3 = counter.get_counts()
    print("TEST 3 (Same person exits):", c3)
    assert c3['in'] == 1 and c3['out'] == 1 and c3['inside'] == 0, f"Failed TEST 3: {c3}"

    # ----------------------------------------------------
    # TEST 4: Two people enter simultaneously
    # ----------------------------------------------------
    counter_multi = PersonCounter(line_position=0.5)
    two_above = [
        {'track_id': 1, 'bbox': [100, 80, 150, 120]},
        {'track_id': 2, 'bbox': [300, 80, 350, 120]}
    ]
    two_below = [
        {'track_id': 1, 'bbox': [100, 280, 150, 320]},
        {'track_id': 2, 'bbox': [300, 280, 350, 320]}
    ]
    counter_multi.update(two_above, frame_shape)
    counter_multi.update(two_below, frame_shape)
    c4 = counter_multi.get_counts()
    print("TEST 4 (Two people enter simultaneously):", c4)
    assert c4['in'] == 2 and c4['out'] == 0 and c4['inside'] == 2, f"Failed TEST 4: {c4}"

    # ----------------------------------------------------
    # TEST 5: One person enters and stays for 100 frames
    # ----------------------------------------------------
    counter_stay = PersonCounter(line_position=0.5)
    p_above = [{'track_id': 10, 'bbox': [100, 80, 150, 120]}]
    p_below = [{'track_id': 10, 'bbox': [100, 280, 150, 320]}]
    counter_stay.update(p_above, frame_shape)
    for _ in range(100):
        counter_stay.update(p_below, frame_shape)
    c5 = counter_stay.get_counts()
    print("TEST 5 (Person enters & stays 100 frames):", c5)
    assert c5['in'] == 1 and c5['inside'] == 1, f"Failed TEST 5: {c5}"

    # ----------------------------------------------------
    # TEST 6: One person repeatedly remains on the same side
    # ----------------------------------------------------
    counter_same = PersonCounter(line_position=0.5)
    p_stay = [{'track_id': 15, 'bbox': [100, 50, 150, 90]}]
    for _ in range(30):
        counter_same.update(p_stay, frame_shape)
    c6 = counter_same.get_counts()
    print("TEST 6 (Repeatedly remains on same side):", c6)
    assert c6['in'] == 0 and c6['out'] == 0, f"Failed TEST 6: {c6}"

    # ----------------------------------------------------
    # TEST 7: Person enters and then exits
    # ----------------------------------------------------
    counter_in_out = PersonCounter(line_position=0.5)
    counter_in_out.update([{'track_id': 5, 'bbox': [50, 50, 90, 90]}], frame_shape)
    counter_in_out.update([{'track_id': 5, 'bbox': [50, 300, 90, 340]}], frame_shape)
    counter_in_out.update([{'track_id': 5, 'bbox': [50, 50, 90, 90]}], frame_shape)
    c7 = counter_in_out.get_counts()
    print("TEST 7 (Enters then exits):", c7)
    assert c7['in'] == 1 and c7['out'] == 1 and c7['inside'] == 0, f"Failed TEST 7: {c7}"

    # ----------------------------------------------------
    # TEST 8: Two different track IDs cross independently
    # ----------------------------------------------------
    counter_indep = PersonCounter(line_position=0.5)
    counter_indep.update([{'track_id': 100, 'bbox': [10, 10, 50, 50]}], frame_shape)
    counter_indep.update([{'track_id': 200, 'bbox': [200, 400, 240, 440]}], frame_shape)
    counter_indep.update([{'track_id': 100, 'bbox': [10, 300, 50, 340]}], frame_shape)
    counter_indep.update([{'track_id': 200, 'bbox': [200, 50, 240, 90]}], frame_shape)
    c8 = counter_indep.get_counts()
    print("TEST 8 (Two different track IDs cross):", c8)
    assert c8['in'] == 1 and c8['out'] == 1 and c8['inside'] == 0, f"Failed TEST 8: {c8}"

    # ----------------------------------------------------
    # TEST 9: Temporary track disappearance
    # ----------------------------------------------------
    counter_loss = PersonCounter(line_position=0.5, max_disappeared_frames=10)
    counter_loss.update([{'track_id': 7, 'bbox': [100, 80, 150, 120]}], frame_shape)  # Above
    # Simulate track disappearance for 20 frames
    empty_frame = []
    for _ in range(20):
        counter_loss.update(empty_frame, frame_shape)
    # Track returns on the OTHER side after long disappearance
    counter_loss.update([{'track_id': 7, 'bbox': [100, 300, 150, 340]}], frame_shape)
    c9 = counter_loss.get_counts()
    print("TEST 9 (Temporary track disappearance):", c9)
    assert c9['in'] == 0 and c9['out'] == 0, f"Failed TEST 9: {c9}"

    # ----------------------------------------------------
    # TEST 10: Prevention of duplicate persistent IDs in IdentityManager
    # ----------------------------------------------------
    id_mgr = IdentityManager()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Track 1 appears
    t1 = [{'track_id': 1, 'bbox': [10, 10, 100, 100]}]
    id_mgr.update(t1, dummy_frame)
    # Track 2 appears simultaneously with Track 1
    t_both = [
        {'track_id': 1, 'bbox': [10, 10, 100, 100]},
        {'track_id': 2, 'bbox': [200, 200, 300, 300]}
    ]
    id_mgr.update(t_both, dummy_frame)
    p_id_1 = t_both[0]['person_id']
    p_id_2 = t_both[1]['person_id']
    print(f"TEST 10 (Unique active person IDs): Track 1 -> {p_id_1}, Track 2 -> {p_id_2}")
    assert p_id_1 != p_id_2, f"Failed TEST 10: Simultaneous active tracks shared ID {p_id_1}"

    print("\n[SUCCESS] ALL 10 UNIT TESTS PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_all_tests()
