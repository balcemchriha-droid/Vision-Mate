"""Step 6 - Log detections to a CSV file (for evaluation / model improvement)."""
import csv
import os
import time
from datetime import datetime


class DetectionLogger:
    def __init__(self, folder="logs", every=1.0):
        os.makedirs(folder, exist_ok=True)
        name = datetime.now().strftime("detections_%Y%m%d_%H%M%S.csv")
        self.path = os.path.join(folder, name)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["time", "class", "confidence", "position", "distance", "closeness"])
        self.every = every
        self._last = 0.0

    def log(self, alerts):
        now = time.time()
        if now - self._last < self.every:  # at most once per second
            return
        self._last = now
        stamp = datetime.now().strftime("%H:%M:%S")
        for a in alerts:
            self._w.writerow([stamp, a["label"], f"{a['conf']:.2f}",
                              a["position"], a["distance"], f"{a['closeness']:.2f}"])
        self._f.flush()

    def close(self):
        self._f.close()
