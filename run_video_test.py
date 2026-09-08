"""
Real Video Integration Test for inandout.mp4 running the full TrackingPipeline.
"""
import cv2
from pipeline import TrackingPipeline


def test_real_video(video_path="inandout.mp4", line_position=0.5):
    print(f"==================================================")
    print(f"RUNNING REAL VIDEO TEST ON: {video_path}")
    print(f"==================================================")

    pipeline = TrackingPipeline("models/best.pt", line_position=line_position)
    pipeline.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    fps_sum = 0.0

    print("Processing video frames...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame, stats = pipeline.process(frame)
        fps_sum += stats['fps']

        if frame_count % 30 == 0 or frame_count == 1:
            print(
                f"Frame {frame_count:04d} | Active: {stats['active']} | "
                f"IN: {stats['in']} | OUT: {stats['out']} | INSIDE: {stats['inside']} | "
                f"Total Unique: {stats['total']} | FPS: {stats['fps']:.1f}"
            )

    cap.release()

    avg_fps = fps_sum / max(1, frame_count)

    print("\n--------------------------------------------------")
    print(f"REAL VIDEO TEST RESULTS FOR '{video_path}':")
    print(f"Total Frames Processed : {frame_count}")
    print(f"Final IN Count         : {stats['in']}")
    print(f"Final OUT Count        : {stats['out']}")
    print(f"Final CURRENTLY INSIDE : {stats['inside']}")
    print(f"Total Unique People    : {pipeline.total_unique_count}")
    print(f"Average Processing FPS : {avg_fps:.1f}")
    print("--------------------------------------------------\n")



if __name__ == "__main__":
    test_real_video()
