"""Step 1 - Object detection with a pretrained YOLOv8 model (COCO)."""
from ultralytics import YOLO


class Detector:
    def __init__(self, weights="yolov8n.pt", conf=0.4):
        # First run downloads yolov8n.pt (~6 MB) -> needs internet once.
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame):
        """Return a list of dicts: {label, conf, box=(x1, y1, x2, y2)}."""
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            detections.append({
                "label": self.model.names[int(b.cls[0])],
                "conf": float(b.conf[0]),
                "box": (x1, y1, x2, y2),
            })
        return detections
