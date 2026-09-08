"""
Unit test for PersonCounter module verifying person IN/OUT counting logic and duplicate prevention.
"""
from person_counter import PersonCounter


def test_person_counter_scenarios():
    counter = PersonCounter(line_position=0.5)
    frame_shape = (480, 640, 3)  # line_y = 240

    print("--- Running Person IN/OUT Counting Unit Test ---")

    # Scenario 1: Person P001 detected ABOVE counting line (cy = 100 < 240)
    person1_f1 = [{'person_id': 'P001', 'track_id': 1, 'bbox': [100, 80, 150, 120]}]
    counter.update(person1_f1, frame_shape)
    counts1 = counter.get_counts()
    print("Frame 1 (P001 above line):", counts1)
    assert counts1['in'] == 0 and counts1['out'] == 0 and counts1['inside'] == 0

    # Scenario 2: Person P001 moves DOWN across counting line (cy = 300 >= 240) -> IN count +1
    person1_f2 = [{'person_id': 'P001', 'track_id': 1, 'bbox': [100, 280, 150, 320]}]
    counter.update(person1_f2, frame_shape)
    counts2 = counter.get_counts()
    print("Frame 2 (P001 crosses DOWN):", counts2)
    assert counts2['in'] == 1 and counts2['out'] == 0 and counts2['inside'] == 1, f"Expected IN=1, OUT=0, INSIDE=1 but got {counts2}"

    # Scenario 3: Person P001 stays on the same side / moves around below line -> IN remains 1
    person1_f3 = [{'person_id': 'P001', 'track_id': 1, 'bbox': [100, 350, 150, 390]}]
    counter.update(person1_f3, frame_shape)
    counts3 = counter.get_counts()
    print("Frame 3 (P001 stays below line):", counts3)
    assert counts3['in'] == 1 and counts3['out'] == 0 and counts3['inside'] == 1, f"Duplicate count detected: {counts3}"

    # Scenario 4: Person P001 moves UP across counting line (cy = 100 < 240) -> OUT count +1
    person1_f4 = [{'person_id': 'P001', 'track_id': 1, 'bbox': [100, 80, 150, 120]}]
    counter.update(person1_f4, frame_shape)
    counts4 = counter.get_counts()
    print("Frame 4 (P001 crosses UP):", counts4)
    assert counts4['in'] == 1 and counts4['out'] == 1 and counts4['inside'] == 0, f"Expected IN=1, OUT=1, INSIDE=0 but got {counts4}"

    # Scenario 5: Another person P002 enters top-to-bottom across line -> IN count +1 (Total IN=2)
    person2_f1 = [{'person_id': 'P002', 'track_id': 2, 'bbox': [200, 50, 250, 90]}]
    counter.update(person2_f1, frame_shape)
    person2_f2 = [{'person_id': 'P002', 'track_id': 2, 'bbox': [200, 280, 250, 320]}]
    counter.update(person2_f2, frame_shape)
    counts5 = counter.get_counts()
    print("Frame 5 & 6 (P002 enters DOWN):", counts5)
    assert counts5['in'] == 2 and counts5['out'] == 1 and counts5['inside'] == 1, f"Expected IN=2, OUT=1, INSIDE=1 but got {counts5}"

    print("\n[SUCCESS] ALL PERSON IN/OUT COUNTING UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_person_counter_scenarios()
