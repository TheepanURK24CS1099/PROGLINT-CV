import os
import torch
from ultralytics import YOLO


# =====================================================================
# STEP 1: LOAD YOLOv11 DETECTOR MODEL
# =====================================================================
def load_detector(model_path: str = "models/mot17_best.pt"):
    """Loads and returns the trained YOLOv11 model."""
    # This code checks if the requested model weight file exists locally with fallback options
    if not os.path.exists(model_path):
        if os.path.exists("mot17_mot20_best.pt"):
            model_path = "mot17_mot20_best.pt"
        elif os.path.exists("mot17_best.pt"):
            model_path = "mot17_best.pt"
        elif os.path.exists("models/best.pt"):
            model_path = "models/best.pt"
        elif os.path.exists("best.pt"):
            model_path = "best.pt"
        else:
            raise FileNotFoundError(f"Model file not found at '{model_path}'")
    
    # This code initializes the YOLO model from Ultralytics
    return YOLO(model_path)


# =====================================================================
# STEP 2: DETECT PERSONS IN FRAME
# =====================================================================
def detect_persons(model, frame, conf_thresh: float = 0.25):
    """
    Runs YOLOv11 on the frame and filters strictly for class ID 0 ('person').
    Returns bounding boxes on CPU, or None if no persons are detected.
    """
    # This code selects CUDA GPU if available, otherwise falls back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # This code runs inference on the frame using YOLOv11
    results = model(frame, conf=conf_thresh, device=device, verbose=False) #controlls log files 
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # This code keeps only boxes corresponding to class 0 ('person') above confidence threshold
    person_indices = [
        idx for idx, box in enumerate(boxes)
        if int(box.cls[0].item()) == 0 and float(box.conf[0].item()) >= conf_thresh
    ]

    # This code returns the filtered person boxes moved to CPU memory
    return boxes[person_indices].cpu() if person_indices else None
