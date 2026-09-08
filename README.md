# YOLOv8 + ByteTrack Person Detection, Tracking & IN/OUT Counting Application

A lightweight, real-time **Person Detection, Tracking & IN/OUT Counting Application** combining custom trained **YOLOv8** object detection with **ByteTrack** multi-object tracking and configurable counting-line crossing detection.

---

## 1. Core Architecture & Algorithm Explanation

```text
Input (Video / Webcam)
          │
          ▼
   YOLOv8 Detection (detector.py)
 (Where is a person?)
          │
          ▼
   Person Filtering (Filter class == 'person')
          │
          ▼
       ByteTrack (tracker.py)
 (Is this the same person?)
          │
          ▼
 Identity Management (identity_manager.py)
 (Persistent IDs: P001, P002...)
          │
          ▼
   Person Counter (person_counter.py)
 (Crossed counting line? IN / OUT +1)
          │
          ▼
 Visualization & Streamlit UI (app.py)
 (IN: X  |  OUT: Y  |  INSIDE: Z)
```

* **YOLOv8 (`detector.py`)**: Responsible for **Person Detection** — locates objects in each frame and produces bounding boxes, class labels, and detection confidence scores. Filters strictly for the `person` class.
* **ByteTrack (`tracker.py`)**: Responsible for **Person Tracking** — assigns persistent unique Track IDs to detected persons across consecutive frames using Kalman filter motion predictions and Hungarian association matching.
* **Identity Manager (`identity_manager.py`)**: Manages persistent Person IDs (`P001`, `P002`...) using HSV color histograms.
* **Person Counter (`person_counter.py`)**: Responsible for **Person IN/OUT Counting** — calculates bounding box centers, tracks side state relative to a configurable counting line (`ABOVE` / `BELOW`), and increments `IN` (moving down) or `OUT` (moving up) counts exactly once per crossing event.

---

## 2. Model Location

The application automatically locates and uses your existing trained YOLOv8 `.pt` model:

* Primary path: `models/best.pt`
* Fallback path: `best.pt`

---

## 3. Quick Start & How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Web UI (Recommended)
Launch the Streamlit user interface:
```bash
python -m streamlit run app.py
```

---

## 4. Modular Code Architecture (Quick Reference)

| File | Lines | Responsibility |
| :--- | :--- | :--- |
| [**`app.py`**](file:///app.py) | ~40 | Streamlit UI with file uploader, counting line slider, and IN/OUT/INSIDE metric displays. |
| [**`pipeline.py`**](file:///pipeline.py) | ~110 | Coordinates the chain: Detect ➔ Track ➔ Identity ➔ Counter ➔ Draw ➔ FPS. |
| [**`person_counter.py`**](file:///person_counter.py) | ~130 | Line crossing detection, side state tracking, duplicate count prevention, overlay drawing. |
| [**`detector.py`**](file:///detector.py) | ~40 | Loads YOLOv8 and filters bounding boxes for `person` class. |
| [**`tracker.py`**](file:///tracker.py) | ~45 | Initializes ByteTrack and performs multi-object frame association. |
| [**`identity_manager.py`**](file:///identity_manager.py) | ~77 | Manages persistent IDs (`P001`, `P002`...) across frames and re-entries. |
| [**`metrics.py`**](file:///metrics.py) | ~23 | Computes rolling average FPS. |
| [**`test_counter.py`**](file:///test_counter.py) | ~55 | Unit tests for PersonCounter line crossing and duplicate prevention scenarios. |

---

## 5. Clean Directory Structure

```text
├── models/
│   └── best.pt               # Trained YOLOv8 model weights
├── detector.py               # Clean YOLOv8 person detection (~40 lines)
├── tracker.py                # Clean ByteTrack tracking (~45 lines)
├── identity_manager.py       # Clean persistent person ID (P001, P002) (~77 lines)
├── person_counter.py         # Person IN/OUT counting & line crossing (~130 lines)
├── metrics.py                # Clean rolling FPS calculation (~23 lines)
├── pipeline.py               # Clean frame processing pipeline (~110 lines)
├── app.py                    # Clean Streamlit UI with Upload + Output + Slider (~40 lines)
├── test_counter.py           # Unit tests for IN/OUT counting logic (~55 lines)
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation
```

---

## 6. Testing Procedure

1. **Unit Test Verification**:
   - Run `python test_counter.py` to test line crossing scenarios (IN, OUT, INSIDE, and duplicate prevention).

2. **Video File Test**:
   - Run `python -m streamlit run app.py`
   - Open browser at `http://localhost:8501`
   - Adjust the **Counting Line Position Ratio** slider (default `0.50`)
   - Upload any video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
   - Verify bounding boxes appear with labels formatted as `ID: P001 | Person | 0.95`, the orange counting line is rendered, and live counts (`IN`, `OUT`, `INSIDE`) increment accurately upon crossing the line.

