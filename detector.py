import os
import torch
from ultralytics import YOLO

class PersonDetector:
    """
    PersonDetector encapsulates YOLOv8 object detection logic,
    filtering detections strictly for the 'person' class.
    """
    def __init__(self, model_path: str = "models/best.pt"):
        # Fallback check for model location
        if not os.path.exists(model_path):
            if os.path.exists("best.pt"):
                model_path = "best.pt"
            else:
                raise FileNotFoundError(
                    f"YOLOv8 model not found at '{model_path}' or 'best.pt'. Please check the model path."
                )

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device.upper()}")
        print(f"Model: {self.model_path}")
        print("Tracker: ByteTrack")

        # Load existing trained YOLOv8 model
        self.model = YOLO(self.model_path)

        # Identify person class ID dynamically from model class names
        self.person_class_id = 0
        if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
            for cid, cname in self.model.names.items():
                if str(cname).lower() == 'person':
                    self.person_class_id = int(cid)
                    break
        print(f"Person class ID identified as: {self.person_class_id}")

    def detect_persons(self, frame, conf_threshold: float = 0.5):
        """
        Runs YOLOv8 inference on a frame and filters results for the person class.

        Args:
            frame: Input image/frame (numpy array)
            conf_threshold: Minimum detection confidence threshold

        Returns:
            person_boxes: Filtered Ultralytics Boxes object containing only person detections,
                          or None if no persons detected.
        """
        # ==========================================
        # STEP 1: YOLOv8 Inference
        # ==========================================
        results = self.model(frame, conf=conf_threshold, device=self.device, verbose=False)
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None

        # ==========================================
        # STEP 2: Person Class & Confidence Filtering
        # ==========================================
        person_indices = []
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id == self.person_class_id and conf >= conf_threshold:
                person_indices.append(idx)

        if not person_indices:
            return None

        person_boxes = boxes[person_indices]
        return person_boxes

    def get_model_info(self):
        return {
            "model_path": self.model_path,
            "device": self.device.upper(),
            "person_class_id": self.person_class_id,
            "names": getattr(self.model, "names", {0: "person"})
        }
