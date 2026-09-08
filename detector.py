import os
import torch
from ultralytics import YOLO


# =====================================================================
# STEP 1: LOAD YOLOv8 MODEL
# =====================================================================
def load_detector(model_path: str = "models/best.pt"):
    """Loads and returns the trained YOLOv8 model."""
    if not os.path.exists(model_path):
        if os.path.exists("best.pt"):
            model_path = "best.pt"
        else:
            raise FileNotFoundError(f"Model file not found at '{model_path}' or 'best.pt'")
    return YOLO(model_path)


# =====================================================================
# STEP 2: DETECT PERSONS IN FRAME
# =====================================================================
def detect_persons(model, frame, conf_thresh: float = 0.5):
    """
    Runs YOLOv8 on the frame and filters strictly for class ID 0 ('person').
    Returns boxes on CPU, or None if no people are detected.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = model(frame, conf=conf_thresh, device=device, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # Keep only boxes that are 'person' (class 0) above confidence threshold
    person_indices = [
        idx for idx, box in enumerate(boxes)
        if int(box.cls[0].item()) == 0 and float(box.conf[0].item()) >= conf_thresh
    ]

    return boxes[person_indices].cpu() if person_indices else None
