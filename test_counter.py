"""
Unit Test Suite for PersonCounter in PROGLINT-CV.
Verifies line crossing detection, start state initialization, state persistence,
independent multi-person tracking, duplicate count prevention, and INSIDE calculation.
"""
from person_counter import PersonCounter


def run_all_tests():
    frame_shape = (480, 640, 3)  # Counting line at Y = 240 (0.5 ratio)

    print("==================================================")
    print("RUNNING PERSON COUNTER UNIT TEST SUITE")
    print("==================================================\n")

    # ----------------------------------------------------
    # TEST 1: Person enters (ABOVE -> BELOW => IN = 1, OUT = 0, INSIDE = 1)
    # ----------------------------------------------------
    # This code tests a valid entry crossing from ABOVE line (center_y = 100) to BELOW line (center_y = 300)
    counter1 = PersonCounter(line_position=0.5)
    counter1.update([{'track_id': 1, 'bbox': [100, 80, 150, 120]}], frame_shape)
    counter1.update([{'track_id': 1, 'bbox': [100, 280, 150, 320]}], frame_shape)
    c1 = counter1.get_counts()
    print("TEST 1 (Person enters):", c1)
    assert c1['in'] == 1 and c1['out'] == 0 and c1['inside'] == 1, f"Failed TEST 1: {c1}"

    # ----------------------------------------------------
    # TEST 2: Person stays BELOW for 100 frames (IN remains 1, OUT = 0, INSIDE = 1)
    # ----------------------------------------------------
    # This code tests duplicate count prevention when a person stays in the BELOW region across multiple consecutive frames
    for _ in range(100):
        counter1.update([{'track_id': 1, 'bbox': [100, 290, 150, 330]}], frame_shape)
    c2 = counter1.get_counts()
    print("TEST 2 (Person stays BELOW for 100 frames):", c2)
    assert c2['in'] == 1 and c2['out'] == 0 and c2['inside'] == 1, f"Failed TEST 2: {c2}"

    # ----------------------------------------------------
    # TEST 3: Person exits (BELOW -> ABOVE => IN = 1, OUT = 1, INSIDE = 0)
    # ----------------------------------------------------
    # This code tests a valid exit crossing from BELOW line to ABOVE line
    counter1.update([{'track_id': 1, 'bbox': [100, 80, 150, 120]}], frame_shape)
    c3 = counter1.get_counts()
    print("TEST 3 (Person exits):", c3)
    assert c3['in'] == 1 and c3['out'] == 1 and c3['inside'] == 0, f"Failed TEST 3: {c3}"

    # ----------------------------------------------------
    # TEST 4: Person starts BELOW at frame 0 (IN = 0, OUT = 0, INSIDE = 0)
    # ----------------------------------------------------
    # This code tests video start state initialization so pre-existing persons inside at video start do not trigger false IN counts
    counter4 = PersonCounter(line_position=0.5)
    counter4.update([{'track_id': 99, 'bbox': [100, 300, 150, 340]}], frame_shape)
    counter4.update([{'track_id': 99, 'bbox': [100, 310, 150, 350]}], frame_shape)
    c4 = counter4.get_counts()
    print("TEST 4 (Person starts BELOW at frame 0):", c4)
    assert c4['in'] == 0 and c4['out'] == 0 and c4['inside'] == 0, f"Failed TEST 4: {c4}"

    # ----------------------------------------------------
    # TEST 5: Two people enter independently (IN = 2, INSIDE = 2)
    # ----------------------------------------------------
    # This code tests independent state tracking for multiple concurrent track IDs
    counter5 = PersonCounter(line_position=0.5)
    two_above = [
        {'track_id': 10, 'bbox': [100, 80, 150, 120]},
        {'track_id': 20, 'bbox': [300, 80, 350, 120]}
    ]
    two_below = [
        {'track_id': 10, 'bbox': [100, 280, 150, 320]},
        {'track_id': 20, 'bbox': [300, 280, 350, 320]}
    ]
    counter5.update(two_above, frame_shape)
    counter5.update(two_below, frame_shape)
    c5 = counter5.get_counts()
    print("TEST 5 (Two people enter independently):", c5)
    assert c5['in'] == 2 and c5['inside'] == 2, f"Failed TEST 5: {c5}"

    # ----------------------------------------------------
    # TEST 6: Repeated frames (One crossing = One count)
    # ----------------------------------------------------
    # This code tests that repeated frames before and after line crossing generate exactly one count event
    counter6 = PersonCounter(line_position=0.5)
    for _ in range(30):
        counter6.update([{'track_id': 50, 'bbox': [50, 50, 90, 90]}], frame_shape)
    c6_a = counter6.get_counts()
    assert c6_a['in'] == 0 and c6_a['out'] == 0
    counter6.update([{'track_id': 50, 'bbox': [50, 300, 90, 340]}], frame_shape)
    for _ in range(30):
        counter6.update([{'track_id': 50, 'bbox': [50, 310, 90, 350]}], frame_shape)
    c6_b = counter6.get_counts()
    print("TEST 6 (Repeated frames):", c6_b)
    assert c6_b['in'] == 1 and c6_b['out'] == 0 and c6_b['inside'] == 1, f"Failed TEST 6: {c6_b}"

    # ----------------------------------------------------
    # TEST 7: INSIDE calculation (enter -> inside = 1, exit -> inside = 0)
    # ----------------------------------------------------
    # This code tests accurate tracking of current INSIDE count via inside_ids set addition and discarding
    counter7 = PersonCounter(line_position=0.5)
    counter7.update([{'track_id': 7, 'bbox': [100, 80, 150, 120]}], frame_shape)
    counter7.update([{'track_id': 7, 'bbox': [100, 280, 150, 320]}], frame_shape)
    assert counter7.get_counts()['inside'] == 1
    counter7.update([{'track_id': 7, 'bbox': [100, 80, 150, 120]}], frame_shape)
    c7 = counter7.get_counts()
    print("TEST 7 (INSIDE enter and exit):", c7)
    assert c7['inside'] == 0 and c7['in'] == 1 and c7['out'] == 1, f"Failed TEST 7: {c7}"

    print("\n[SUCCESS] ALL UNIT TESTS PASSED PERFECTLY!\n")


if __name__ == "__main__":
    run_all_tests()
