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

## 4. Modular Code Architecture (Quick Reference)

| File | Lines | Responsibility |
| :--- | :--- | :--- |
| [**`app.py`**](file:///app.py) | ~60 | Minimal Streamlit UI (Upload button + video frame display). Zero HTML/CSS. |
| [**`pipeline.py`**](file:///pipeline.py) | ~55 | Coordinates the chain: Detect ➔ Track ➔ Identity ➔ Draw ➔ FPS. |
| [**`detector.py`**](file:///detector.py) | ~38 | Loads YOLOv8 and filters bounding boxes for `person` class. |
| [**`tracker.py`**](file:///tracker.py) | ~34 | Initializes ByteTrack and performs multi-object frame association. |
| [**`identity_manager.py`**](file:///identity_manager.py) | ~75 | Manages persistent IDs (`P001`, `P002`...) across frames and re-entries. |
| [**`metrics.py`**](file:///metrics.py) | ~24 | Computes rolling average FPS. |

---

## 5. Clean Directory Structure

```text
├── models/
│   └── best.pt               # Trained YOLOv8 model weights
├── detector.py               # Clean YOLOv8 person detection (~40 lines)
├── tracker.py                # Clean ByteTrack tracking (~44 lines)
├── identity_manager.py       # Clean persistent person ID (P001, P002) (~76 lines)
├── metrics.py                # Clean rolling FPS calculation (~22 lines)
├── pipeline.py               # Clean frame processing pipeline (~78 lines)
├── app.py                    # Clean Streamlit UI with Upload + Output only (~53 lines)
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
