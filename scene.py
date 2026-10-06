"""Steps 2, 3, 4 - Position (left/ahead/right), rough distance, sentence + priority."""
import math

DISTANCE_ORDER = {"very close": 3, "near": 2, "far": 1}


def get_position(box, frame_w):
    """Split the frame in 3 vertical zones and use the box center."""
    cx = (box[0] + box[2]) / 2
    if cx < frame_w / 3:
        return "on your left"
    if cx > 2 * frame_w / 3:
        return "on your right"
    return "ahead"


def get_closeness(box, frame_w, frame_h):
    """Proxy for distance: the bigger the box in the frame, the closer the object.
    Returns a value between 0 and 1 (to be replaced by MiDaS depth later)."""
    area = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    return math.sqrt(area / (frame_w * frame_h))


def get_distance_label(closeness):
    if closeness > 0.50:
        return "very close"
    if closeness > 0.25:
        return "near"
    return "far"


def analyze(detections, frame_w, frame_h):
    """Add position + distance to each detection and sort: closest first."""
    alerts = []
    for d in detections:
        closeness = get_closeness(d["box"], frame_w, frame_h)
        alerts.append({
            **d,
            "position": get_position(d["box"], frame_w),
            "closeness": closeness,
            "distance": get_distance_label(closeness),
        })
    alerts.sort(key=lambda a: a["closeness"], reverse=True)
    return alerts


def build_message(alerts, max_alerts=2):
    """Turn the top alerts into short spoken sentences."""
    parts = []
    for a in alerts[:max_alerts]:
        sentence = f"{a['label']}, {a['distance']}, {a['position']}"
        if a["distance"] == "very close":
            sentence = "Warning! " + sentence
        parts.append(sentence)
    return ". ".join(parts)
