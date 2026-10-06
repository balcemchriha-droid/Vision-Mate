"""VisionMate - prototype: Camera -> YOLO -> position -> distance -> sentence -> voice.

Usage:
    python main.py                          # webcam
    python main.py --source video.mp4       # video file
    python main.py --source photo.jpg       # one image (press any key to close)
    python main.py --source images_folder   # all images of a folder (any key = next)
    python main.py --no-voice               # without audio
    python main.py --save demo_output.mp4   # record the annotated video
Press 'q' to quit.
"""
import argparse
import os
import time

import cv2

from detector import Detector
from logger import DetectionLogger
from scene import analyze, build_message
from speech import Speaker

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COLORS = {"very close": (0, 0, 255), "near": (0, 165, 255), "far": (0, 200, 0)}  # BGR


def frames(source):
    """Yield (frame, is_static_image)."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif os.path.isdir(source):
        for f in sorted(os.listdir(source)):
            if f.lower().endswith(IMG_EXT):
                img = cv2.imread(os.path.join(source, f))
                if img is not None:
                    yield img, True
        return
    elif source.lower().endswith(IMG_EXT):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(source)
        yield img, True
        return
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, False
    cap.release()


def draw(frame, alerts, message):
    for a in alerts:
        x1, y1, x2, y2 = map(int, a["box"])
        color = COLORS[a["distance"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{a['label']} | {a['distance']} | {a['position']}"
        cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    h, w = frame.shape[:2]
    # 3 zones guide lines
    for x in (w // 3, 2 * w // 3):
        cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)
    if message:
        cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(frame, message, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--cooldown", type=float, default=3.0, help="seconds between voice alerts")
    p.add_argument("--no-voice", action="store_true")
    p.add_argument("--save", default=None, help="path of the output video, e.g. demo.mp4")
    args = p.parse_args()

    detector = Detector(args.weights, args.conf)
    speaker = Speaker(enabled=not args.no_voice)
    logger = DetectionLogger()
    writer = None
    last_msg, last_time = "", 0.0
    shown_msg = ""

    print("VisionMate started - press 'q' to quit")
    for frame, is_static in frames(args.source):
        h, w = frame.shape[:2]
        alerts = analyze(detector.detect(frame), w, h)
        message = build_message(alerts)

        # Voice: cooldown + do not repeat the same sentence too often
        now = time.time()
        if message and now - last_time > args.cooldown and \
                (message != last_msg or now - last_time > 8):
            if speaker.say(message):
                last_msg, last_time = message, now
            if args.no_voice:
                last_msg, last_time = message, now
            print(f"[ALERT] {message}")
        if message:
            shown_msg = message

        logger.log(alerts)
        out = draw(frame, alerts, shown_msg)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(out)

        cv2.imshow("VisionMate", out)
        key = cv2.waitKey(0 if is_static else 1) & 0xFF
        if is_static and message:
            speaker.say(message)
            time.sleep(2)
        if key == ord("q"):
            break

    if writer:
        writer.release()
    logger.close()
    cv2.destroyAllWindows()
    print(f"Detections saved in: {logger.path}")


if __name__ == "__main__":
    main()
