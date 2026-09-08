# YOLOv8 + ByteTrack Person Detection and Tracking Application

A lightweight, real-time **Person Detection and Tracking Application** combining custom trained **YOLOv8** object detection with the **ByteTrack** multi-object tracking algorithm.

---

## 1. Core Architecture & Algorithm Explanation

```text
Input (Video / Webcam)
          │
          ▼
   YOLOv8 Detection
 (Where is a person?)
          │
          ▼
  Person Filtering
 (Filter class == 'person')
          │
          ▼
      ByteTrack
 (Is this the same person?)
          │
          ▼
   Track ID Mapping
          │
          ▼
    Visualization & Metrics
```

* **YOLOv8 (`detector.py`)**: Responsible for **Person Detection** — locates objects in each frame and produces bounding boxes, class labels, and detection confidence scores. Filters strictly for the `person` class.
* **ByteTrack (`tracker.py`)**: Responsible for **Person Tracking** — assigns persistent unique Track IDs to detected persons across consecutive frames using Kalman filter motion predictions and Hungarian association matching.

---

## 2. Model Location

The application automatically locates and uses your existing trained YOLOv8 `.pt` model:

* Primary path: `models/best.pt`
* Fallback path: `best.pt`

> **Note:** The existing trained `.pt` model is loaded directly without downloading, fine-tuning, or altering model weights.

---

## 3. Quick Start & How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Web UI (Recommended)
Launch the Streamlit user interface:
```bash
streamlit run app.py
```

### Step 3: Alternative OpenCV Window Mode (CLI)
You can also run directly with Python:
```bash
python app.py
```

---

## 4. Key Code Locations (Quick Reference)

| Functionality | File | Class / Method | Line / Location |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Inference** | [`detector.py`](file:///d:/yolov8/detector.py) | `PersonDetector.detect_persons()` | `self.model(frame, conf=conf_threshold)` |
| **Person Class Filtering** | [`detector.py`](file:///d:/yolov8/detector.py) | `PersonDetector.detect_persons()` | `box.cls[0] == self.person_class_id` |
| **ByteTrack Initialization** | [`tracker.py`](file:///d:/yolov8/tracker.py) | `PersonTracker._init_tracker()` | `BYTETracker(args)` |
| **ByteTrack Update** | [`tracker.py`](file:///d:/yolov8/tracker.py) | `PersonTracker.track_persons()` | `self.tracker.update(boxes_cpu, frame)` |
| **Track ID Extraction** | [`tracker.py`](file:///d:/yolov8/tracker.py) | `PersonTracker.track_persons()` | `track_id = int(t[4])` |
| **FPS Calculation** | [`metrics.py`](file:///d:/yolov8/metrics.py) | `TrackingMetrics.get_fps()` | Rolling average `(1.0 / avg_time)` |
| **Unique Person Counting** | [`tracker.py`](file:///d:/yolov8/tracker.py) | `PersonTracker.unique_ids` | `self.unique_ids.add(track_id)` |
| **Frame Pipeline & Visuals** | [`pipeline.py`](file:///d:/yolov8/pipeline.py) | `process_frame()` & `draw_tracks()` | Bounding boxes & `ID: X \| Person \| Y.YY` |

---

## 5. Directory Structure

```text
d:/yolov8/
│
├── models/
│   └── best.pt               # Trained YOLOv8 model weights
├── input/
│   └── sample.mp4            # Input video folder
├── output/                   # Processed outputs directory
├── detector.py               # YOLOv8 detection & person filtering logic
├── tracker.py                # ByteTrack tracking integration
├── metrics.py                # Rolling FPS & unique ID metrics
├── pipeline.py               # Shared frame-by-frame processing pipeline
├── app.py                    # Streamlit UI & OpenCV entry point
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation
```

---

## 6. Testing Procedure

1. **Video File Test**:
   - Run `streamlit run app.py`
   - Choose **Upload Video**
   - Upload any video file (`.mp4`, `.avi`, `.mov`, `.mkv`) or use `input/sample.mp4`
   - Adjust the **Confidence Threshold** slider (default `0.50`)
   - Click **Start Tracking**
   - Verify bounding boxes appear with labels formatted as `ID: X | Person | 0.XX`, processing FPS is displayed, and `Total Unique People` increments correctly.

2. **Webcam Test**:
   - Select **Start Webcam**
   - Click **Start Tracking**
   - Move in front of the camera to verify track ID persistence and detection.
