"""
Multi-Boundary Overhead Walkway Person Counter
===============================================
Counts people entering ('IN') and leaving ('OUT') across three distinct boundaries:

Counting Rules:
1. IN Condition:
   - Horizontal Blue Line (Bottom region).
   - Triggered when tracked person's centroid crosses downward/inward (y < y_blue to y >= y_blue).

2. OUT Conditions (Two Exit Boundaries):
   - Boundary 1 (Top/Cyan Counting Line, Y = 425): Triggered when centroid crosses upward/away (y > y_cyan to y <= y_cyan).
   - Boundary 2 (Slanted Orange Line on Left, (x1, y1) to (x2, y2)): Triggered when centroid crosses from Right to Left.

Requirements & Features:
- Unique tracking per ByteTrack track_id to prevent double counting.
- 2D Cross-Product & Line Segment Intersection geometry math.
- On-screen HUD dashboard: "IN: <count> | OUT: <count> | CURRENT INSIDE: <IN - OUT>".
- Centroid trails, bounding boxes, track IDs, and line labels.
- Modular class design with easily customizable line coordinates.
"""

import os
import cv2
import numpy as np
import argparse
from collections import defaultdict
from ultralytics import YOLO


# =====================================================================
# 1. GEOMETRY & LINE INTERSECTION UTILITIES
# =====================================================================

def cross_product_2d(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Computes the 2D cross product of vector (p2 - p1) and (p3 - p1).
    - Positive value (> 0): p3 lies to the LEFT of vector p1 -> p2.
    - Negative value (< 0): p3 lies to the RIGHT of vector p1 -> p2.
    - Zero (== 0): Points are collinear.
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def is_segment_intersecting(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
    """
    Determines if line segment A-B (person movement vector) intersects line segment C-D (boundary line).
    """
    cp1 = cross_product_2d(c, d, a)
    cp2 = cross_product_2d(c, d, b)
    cp3 = cross_product_2d(a, b, c)
    cp4 = cross_product_2d(a, b, d)

    # Segments intersect if endpoints of each segment lie on opposite sides of the other segment
    if ((cp1 > 0 and cp2 < 0) or (cp1 < 0 and cp2 > 0)) and \
       ((cp3 > 0 and cp4 < 0) or (cp3 < 0 and cp4 > 0)):
        return True
    return False


# =====================================================================
# 2. MULTI-BOUNDARY PERSON COUNTER CLASS
# =====================================================================

class MultiBoundaryCounter:
    """
    Manages multi-boundary line crossing logic for IN and OUT counts.
    """
    def __init__(self, 
                 y_blue: int = 550, 
                 y_cyan: int = 425, 
                 orange_line: tuple = ((100, 200), (250, 500))):
        """
        :param y_blue: Y-coordinate for horizontal Blue Line (IN boundary, bottom region)
        :param y_cyan: Y-coordinate for horizontal Cyan Line (OUT boundary 1, top region)
        :param orange_line: Tuple ((x1, y1), (x2, y2)) for slanted Orange Line (OUT boundary 2, left region)
        """
        self.y_blue = y_blue
        self.y_cyan = y_cyan
        self.orange_line = orange_line

        # Cumulative counters
        self.in_count = 0
        self.out_count = 0

        # Trajectory history per track_id -> list of (cx, cy)
        self.track_history = defaultdict(list)

        # Persistent state tracking to ensure an ID is counted only once per crossing pass
        self.counted_in_ids = set()
        self.counted_out_ids = set()
        self.active_inside_ids = set()

    def set_line_coordinates(self, y_blue: int = None, y_cyan: int = None, orange_line: tuple = None):
        """Allows dynamic updating of boundary line coordinates."""
        if y_blue is not None:
            self.y_blue = y_blue
        if y_cyan is not None:
            self.y_cyan = y_cyan
        if orange_line is not None:
            self.orange_line = orange_line

    def get_centroid(self, bbox: list) -> tuple:
        """Calculates (cx, cy) centroid from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        return cx, cy

    def update(self, tracked_persons: list, frame_width: int):
        """
        Updates line crossing state for all tracked persons.
        
        :param tracked_persons: List of dicts [{'track_id': id, 'bbox': [x1,y1,x2,y2]}, ...]
        :param frame_width: Width of frame in pixels
        """
        for person in tracked_persons:
            track_id = int(person['track_id'])
            bbox = person['bbox']
            curr_pt = self.get_centroid(bbox)

            # Store centroid trajectory history (keep last 30 frames)
            self.track_history[track_id].append(curr_pt)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            # Require at least 2 centroid points to evaluate movement vector
            if len(self.track_history[track_id]) < 2:
                continue

            prev_pt = self.track_history[track_id][-2]
            xa, ya = prev_pt
            xb, yb = curr_pt

            # -----------------------------------------------------------------
            # RULE 1: IN Condition (Horizontal Blue Line, Bottom Region)
            # Moving downward/inward (ya < y_blue to yb >= y_blue)
            # -----------------------------------------------------------------
            if track_id not in self.counted_in_ids:
                blue_segment_start = (0, self.y_blue)
                blue_segment_end = (frame_width, self.y_blue)

                if ya < self.y_blue and yb >= self.y_blue:
                    if is_segment_intersecting(prev_pt, curr_pt, blue_segment_start, blue_segment_end):
                        self.in_count += 1
                        self.counted_in_ids.add(track_id)
                        self.active_inside_ids.add(track_id)

            # -----------------------------------------------------------------
            # RULE 2: OUT Condition - Boundary 1 (Horizontal Cyan Line, Top Region)
            # Moving upward/away (ya > y_cyan to yb <= y_cyan)
            # -----------------------------------------------------------------
            if track_id not in self.counted_out_ids:
                cyan_segment_start = (0, self.y_cyan)
                cyan_segment_end = (frame_width, self.y_cyan)

                if ya > self.y_cyan and yb <= self.y_cyan:
                    if is_segment_intersecting(prev_pt, curr_pt, cyan_segment_start, cyan_segment_end):
                        self.out_count += 1
                        self.counted_out_ids.add(track_id)
                        self.active_inside_ids.discard(track_id)

            # -----------------------------------------------------------------
            # RULE 3: OUT Condition - Boundary 2 (Slanted Orange Line on Left)
            # Moving from Right to Left across slanted line segment (x1, y1) -> (x2, y2)
            # -----------------------------------------------------------------
            if track_id not in self.counted_out_ids:
                (ox1, oy1), (ox2, oy2) = self.orange_line
                
                # Orient line vector from upper Y to lower Y endpoint
                if oy1 <= oy2:
                    p_start, p_end = (ox1, oy1), (ox2, oy2)
                else:
                    p_start, p_end = (ox2, oy2), (ox1, oy1)

                # Compute 2D cross product for previous and current centroid
                cp_prev = cross_product_2d(p_start, p_end, prev_pt)
                cp_curr = cross_product_2d(p_start, p_end, curr_pt)

                # Moving from Right (cp < 0) to Left (cp >= 0) across line
                if cp_prev < 0 and cp_curr >= 0:
                    if is_segment_intersecting(prev_pt, curr_pt, p_start, p_end):
                        self.out_count += 1
                        self.counted_out_ids.add(track_id)
                        self.active_inside_ids.discard(track_id)

    def draw_hud(self, frame: np.ndarray, tracked_persons: list) -> np.ndarray:
        """
        Renders HUD dashboard, boundary lines, bounding boxes, IDs, and centroids.
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # BGR Colors
        COLOR_BLUE = (255, 0, 0)        # Blue Line (IN)
        COLOR_CYAN = (255, 255, 0)      # Cyan Line (OUT Exit 1)
        COLOR_ORANGE = (0, 165, 255)    # Orange Line (OUT Exit 2)
        COLOR_BOX = (0, 255, 0)         # Bounding Box Green
        COLOR_CENTROID = (0, 0, 255)    # Centroid Point Red
        COLOR_TRAIL = (0, 255, 255)     # Trajectory Trail Yellow

        # 1. Draw Horizontal Blue Line (IN)
        cv2.line(annotated, (0, self.y_blue), (width, self.y_blue), COLOR_BLUE, 3, cv2.LINE_AA)
        cv2.putText(annotated, f"IN BOUNDARY (Blue Line Y={self.y_blue}) v", (15, self.y_blue + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLUE, 2, cv2.LINE_AA)

        # 2. Draw Horizontal Cyan Line (OUT Exit 1)
        cv2.line(annotated, (0, self.y_cyan), (width, self.y_cyan), COLOR_CYAN, 3, cv2.LINE_AA)
        cv2.putText(annotated, f"OUT EXIT 1 (Cyan Line Y={self.y_cyan}) ^", (15, self.y_cyan - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CYAN, 2, cv2.LINE_AA)

        # 3. Draw Slanted Orange Line (OUT Exit 2)
        (ox1, oy1), (ox2, oy2) = self.orange_line
        cv2.line(annotated, (ox1, oy1), (ox2, oy2), COLOR_ORANGE, 3, cv2.LINE_AA)
        mid_x, mid_y = int((ox1 + ox2) / 2), int((oy1 + oy2) / 2)
        cv2.putText(annotated, "< OUT EXIT 2 (Orange Line)", (mid_x + 10, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ORANGE, 2, cv2.LINE_AA)

        # 4. Draw Active Track Detections, Trajectories & Centroids
        for person in tracked_persons:
            track_id = int(person['track_id'])
            x1, y1, x2, y2 = map(int, person['bbox'])
            cx, cy = self.get_centroid((x1, y1, x2, y2))

            # Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_BOX, 2)
            cv2.putText(annotated, f"ID: {track_id}", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BOX, 2, cv2.LINE_AA)

            # Centroid Point
            cv2.circle(annotated, (cx, cy), 5, COLOR_CENTROID, -1)

            # Centroid Trajectory Trail
            history = self.track_history[track_id]
            for i in range(1, len(history)):
                cv2.line(annotated, history[i - 1], history[i], COLOR_TRAIL, 2, cv2.LINE_AA)

        # 5. Draw On-Screen Dashboard Box
        current_inside = max(0, self.in_count - self.out_count)
        dashboard_text = f"IN: {self.in_count}  |  OUT: {self.out_count}  |  CURRENT INSIDE: {current_inside}"
        
        (text_w, text_h), _ = cv2.getTextSize(dashboard_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (15, 15), (text_w + 35, text_h + 30), (30, 30, 30), -1)
        cv2.rectangle(annotated, (15, 15), (text_w + 35, text_h + 30), (255, 255, 255), 2)
        cv2.putText(annotated, dashboard_text, (25, 15 + text_h + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated


# =====================================================================
# 3. PIPELINE EXECUTION & MAIN FUNCTION
# =====================================================================

def run_multi_boundary_counter(video_source: str,
                               model_path: str = "yolov8n.pt",
                               y_blue: int = 550,
                               y_cyan: int = 425,
                               orange_line: tuple = ((100, 200), (250, 500)),
                               conf_thresh: float = 0.25,
                               display: bool = True,
                               output_path: str = None):
    """
    Runs YOLO tracking pipeline with MultiBoundaryCounter.
    """
    # 1. Load YOLO model
    if not os.path.exists(model_path):
        if os.path.exists("models/best.pt"):
            model_path = "models/best.pt"
        elif os.path.exists("last.pt"):
            model_path = "last.pt"
    
    print(f"[INFO] Loading YOLO model from '{model_path}'...")
    model = YOLO(model_path)

    # 2. Initialize Video Capture
    cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"[INFO] Video resolution: {width}x{height} @ {fps:.1f} FPS")

    # Initialize Video Writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3. Initialize Counter
    counter = MultiBoundaryCounter(y_blue=y_blue, y_cyan=y_cyan, orange_line=orange_line)

    print("[INFO] Processing video stream. Press 'q' to exit.")

    # 4. Processing Loop using Ultralytics Tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO object tracking (filtering for class 0: person)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_thresh, verbose=False)
        boxes = results[0].boxes

        tracked_persons = []
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxy_boxes = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.int().cpu().tolist()

            for track_id, box, cls_id in zip(track_ids, xyxy_boxes, classes):
                if cls_id == 0:  # Class 0 is 'person'
                    tracked_persons.append({
                        'track_id': track_id,
                        'bbox': box.tolist()
                    })

        # Update multi-boundary counting logic
        counter.update(tracked_persons, frame_width=width)

        # Draw HUD overlay
        annotated_frame = counter.draw_hud(frame, tracked_persons)

        # Write frame to file if requested
        if writer:
            writer.write(annotated_frame)

        # Display output window
        if display:
            cv2.imshow("Multi-Boundary Person Counter", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("[INFO] Processing complete.")
    print(f"[FINAL STATS] IN: {counter.in_count} | OUT: {counter.out_count} | INSIDE: {max(0, counter.in_count - counter.out_count)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Boundary Overhead Person Counter")
    parser.add_argument("--source", type=str, default="0", help="Path to video file or webcam index (default: 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--y_blue", type=int, default=550, help="Y-coordinate for Blue Line (IN boundary)")
    parser.add_argument("--y_cyan", type=int, default=425, help="Y-coordinate for Cyan Line (OUT boundary 1)")
    parser.add_argument("--orange_line", type=int, nargs=4, default=[100, 200, 250, 500],
                        help="Slanted Orange Line endpoints: x1 y1 x2 y2")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display window")

    args = parser.parse_args()

    orange_tuple = ((args.orange_line[0], args.orange_line[1]), (args.orange_line[2], args.orange_line[3]))

    run_multi_boundary_counter(
        video_source=args.source,
        model_path=args.model,
        y_blue=args.y_blue,
        y_cyan=args.y_cyan,
        orange_line=orange_tuple,
        conf_thresh=args.conf,
        display=not args.no_display,
        output_path=args.output
    )
