import cv2
import time
from collections import Counter
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────
ALERT_CLASSES = {"person", "cell phone", "laptop"}  # edit these
CONFIDENCE    = 0.5
SAVE_OUTPUT   = True
OUTPUT_FILE   = "output.avi"
# ───────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")
cap   = cv2.VideoCapture(0)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30

writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))
    print(f"Recording to {OUTPUT_FILE}")

prev_time   = time.time()
alert_shown = {}   # tracks cooldown per class so alerts don't spam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, conf=CONFIDENCE, verbose=False)
    counts  = Counter()
    boxes   = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = model.names[int(box.cls[0])]
            counts[cls] += 1
            boxes.append((x1, y1, x2, y2, cls, conf))

    # ── draw boxes ──────────────────────────────────────
    for (x1, y1, x2, y2, cls, conf) in boxes:
        color = (0, 0, 220) if cls in ALERT_CLASSES else (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} {conf:.0%}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── alert banner ────────────────────────────────────
    now = time.time()
    triggered = [c for c in counts if c in ALERT_CLASSES]
    for cls in triggered:
        last = alert_shown.get(cls, 0)
        if now - last > 3:          # 3-second cooldown
            print(f"[ALERT] {cls} detected!")
            alert_shown[cls] = now

    if triggered:
        banner = "ALERT: " + ", ".join(triggered)
        cv2.rectangle(frame, (0, 0), (width, 36), (0, 0, 180), -1)
        cv2.putText(frame, banner, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # ── object counter panel ─────────────────────────────
    panel_y = 50
    cv2.putText(frame, "Detected:", (10, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    for cls, cnt in counts.most_common():
        panel_y += 22
        cv2.putText(frame, f"  {cls}: {cnt}", (10, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # ── FPS ─────────────────────────────────────────────
    curr_time = time.time()
    fps_live  = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps_live:.1f}", (width - 110, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    if writer:
        writer.write(frame)

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer:
    writer.release()
    print(f"Saved to {OUTPUT_FILE}")
cv2.destroyAllWindows()